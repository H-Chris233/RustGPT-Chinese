// ============================================================================
// 模型序列化模块 - 支持二进制和 JSON 两种格式
// ============================================================================
//
// 本模块实现了 RustGPT-Chinese 模型的持久化功能,支持两种序列化格式:
//
// 1. **二进制格式**（推荐用于本仓库的训练恢复与日常保存）：
//    - 使用 bincode 序列化,文件小、速度快
//    - 保存完整的优化器状态(Adam 的 m、v 动量)
//    - 支持断点续训
//    - 文件扩展名: .bin
//
// 2. **JSON 格式**（推荐用于教学检查与调试）：
//    - 人类可读,方便检查权重
//    - 跨语言兼容,可用 Python 读取
//    - 保存完整的优化器状态
//    - 文件扩展名: .json
//
// ============================================================================

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::{Decode, Encode};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    EMBEDDING_DIM, HIDDEN_DIM,
    adam::Adam,
    dropout::Dropout,
    embeddings::Embeddings,
    feed_forward::FeedForward,
    layer_norm::LayerNorm,
    llm::{LLM, Layer},
    output_projection::OutputProjection,
    position_encoding::PositionEncoding,
    self_attention::SelfAttention,
    transformer::TransformerBlock,
    vocab::Vocab,
};

/// 反序列化输入大小上限（防止恶意/损坏文件导致 OOM）。
///
/// 说明：
/// - 这里的限制主要用于“解码阶段的资源上限控制”，并不等同于模型实际大小；
/// - 如需支持更大的模型/检查点，可在后续改为可配置（例如环境变量），但当前按 KISS 先给出保守上限。
const BINCODE_DECODE_LIMIT_BYTES: usize = 512 * 1024 * 1024; // 512MiB
const JSON_DECODE_LIMIT_BYTES: u64 = 256 * 1024 * 1024; // 256MiB

/// 校验序列化字段中是否包含非有限值；若包含，则返回精确错误而不是静默修补。
fn ensure_finite_slice(data: &[f32], field_name: &str) -> Result<(), String> {
    if let Some((idx, value)) = data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "字段 {} 含有非有限值: index={}, value={}",
            field_name, idx, value
        ));
    }
    Ok(())
}

/// 将运行中的二维参数矩阵导出为可序列化向量；若矩阵已损坏，则直接失败。
fn collect_finite_array2_data(matrix: &Array2<f32>, field_name: &str) -> Result<Vec<f32>, String> {
    let data: Vec<f32> = matrix.iter().copied().collect();
    ensure_finite_slice(&data, field_name)?;
    Ok(data)
}

/// 按给定形状严格重建二维参数矩阵。
///
/// 与旧实现不同：
/// - 维度不匹配时直接报错；
/// - 数据包含 NaN/Inf 时直接报错；
/// - 不再静默回退为零矩阵，以免把损坏 checkpoint 伪装成“加载成功”。
fn rebuild_array2(
    shape: (usize, usize),
    data: &[f32],
    field_name: &str,
) -> Result<Array2<f32>, String> {
    let expected_len = shape
        .0
        .checked_mul(shape.1)
        .ok_or_else(|| format!("字段 {} 的形状乘积溢出: {:?}", field_name, shape))?;

    if data.len() != expected_len {
        return Err(format!(
            "字段 {} 的元素数量与形状不匹配: expected={}, actual={}, shape={:?}",
            field_name,
            expected_len,
            data.len(),
            shape
        ));
    }

    ensure_finite_slice(data, field_name)?;
    Array2::from_shape_vec(shape, data.to_vec())
        .map_err(|error| format!("重建字段 {} 失败: {}", field_name, error))
}

fn count_transformer_blocks(network: &[Box<dyn Layer>]) -> usize {
    network
        .iter()
        .filter(|layer| layer.as_any().is::<TransformerBlock>())
        .count()
}

// ============================================================================
// Adam 优化器状态序列化
// ============================================================================

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableAdam {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub timestep: usize,
    pub m_shape: (usize, usize),
    pub m_data: Vec<f32>,
    pub v_shape: (usize, usize),
    pub v_data: Vec<f32>,
}

impl SerializableAdam {
    pub fn from_adam(adam: &Adam) -> Result<Self, String> {
        Ok(Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: adam.timestep,
            m_shape: adam.m.dim(),
            m_data: collect_finite_array2_data(&adam.m, "adam.m")?,
            v_shape: adam.v.dim(),
            v_data: collect_finite_array2_data(&adam.v, "adam.v")?,
        })
    }

    pub fn to_adam(&self) -> Result<Adam, String> {
        if !self.beta1.is_finite() || !self.beta2.is_finite() || !self.epsilon.is_finite() {
            return Err("Adam 超参数含有非有限值".to_string());
        }

        Ok(Adam {
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            timestep: self.timestep,
            m: rebuild_array2(self.m_shape, &self.m_data, "adam.m")?,
            v: rebuild_array2(self.v_shape, &self.v_data, "adam.v")?,
        })
    }
}

// ============================================================================
// 各层的可序列化表示
// ============================================================================

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableEmbeddings {
    pub token_embeddings_shape: (usize, usize),
    pub token_embeddings_data: Vec<f32>,
    pub token_optimizer: SerializableAdam,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableSelfAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q_shape: (usize, usize),
    pub w_q_data: Vec<f32>,
    pub w_k_shape: (usize, usize),
    pub w_k_data: Vec<f32>,
    pub w_v_shape: (usize, usize),
    pub w_v_data: Vec<f32>,
    pub w_o_shape: (usize, usize),
    pub w_o_data: Vec<f32>,
    pub optimizer_w_q: SerializableAdam,
    pub optimizer_w_k: SerializableAdam,
    pub optimizer_w_v: SerializableAdam,
    pub optimizer_w_o: SerializableAdam,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableFeedForward {
    pub w1_shape: (usize, usize),
    pub w1_data: Vec<f32>,
    pub b1_shape: (usize, usize),
    pub b1_data: Vec<f32>,
    pub w2_shape: (usize, usize),
    pub w2_data: Vec<f32>,
    pub b2_shape: (usize, usize),
    pub b2_data: Vec<f32>,
    pub optimizer_w1: SerializableAdam,
    pub optimizer_b1: SerializableAdam,
    pub optimizer_w2: SerializableAdam,
    pub optimizer_b2: SerializableAdam,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableLayerNorm {
    pub epsilon: f32,
    pub gamma_shape: (usize, usize),
    pub gamma_data: Vec<f32>,
    pub beta_shape: (usize, usize),
    pub beta_data: Vec<f32>,
    pub optimizer_gamma: SerializableAdam,
    pub optimizer_beta: SerializableAdam,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableDropout {
    pub dropout_rate: f32,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableOutputProjection {
    pub w_out_shape: (usize, usize),
    pub w_out_data: Vec<f32>,
    pub b_out_shape: (usize, usize),
    pub b_out_data: Vec<f32>,
    pub optimizer: SerializableAdam,
    pub optimizer_bias: SerializableAdam,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableTransformerBlock {
    pub attention: SerializableSelfAttention,
    pub feed_forward: SerializableFeedForward,
    pub dropout1: SerializableDropout,
    pub dropout2: SerializableDropout,
    pub norm1: SerializableLayerNorm,
    pub norm2: SerializableLayerNorm,
}

// ============================================================================
// 层类型枚举
// ============================================================================

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub enum SerializableLayer {
    Embeddings(SerializableEmbeddings),
    TransformerBlock(SerializableTransformerBlock),
    OutputProjection(SerializableOutputProjection),
}

impl SerializableLayer {
    pub fn from_layer(layer: &Box<dyn Layer>) -> Result<Self, String> {
        if let Some(embeddings) = layer.as_any().downcast_ref::<Embeddings>() {
            return Ok(SerializableLayer::Embeddings(Self::serialize_embeddings(
                embeddings,
            )?));
        }

        if let Some(transformer) = layer.as_any().downcast_ref::<TransformerBlock>() {
            return Ok(SerializableLayer::TransformerBlock(
                Self::serialize_transformer_block(transformer)?,
            ));
        }

        if let Some(output_proj) = layer.as_any().downcast_ref::<OutputProjection>() {
            return Ok(SerializableLayer::OutputProjection(
                Self::serialize_output_projection(output_proj)?,
            ));
        }

        Err(format!("Unsupported layer type: {}", layer.layer_type()))
    }

    pub fn to_layer(&self, vocab_size: usize) -> Result<Box<dyn Layer>, String> {
        match self {
            SerializableLayer::Embeddings(s) => Ok(Box::new(Self::deserialize_embeddings(s)?)),
            SerializableLayer::TransformerBlock(s) => {
                Ok(Box::new(Self::deserialize_transformer_block(s)?))
            }
            SerializableLayer::OutputProjection(s) => Ok(Box::new(
                Self::deserialize_output_projection(s, vocab_size)?,
            )),
        }
    }

    fn serialize_embeddings(embeddings: &Embeddings) -> Result<SerializableEmbeddings, String> {
        Ok(SerializableEmbeddings {
            token_embeddings_shape: embeddings.token_embeddings.dim(),
            token_embeddings_data: collect_finite_array2_data(
                &embeddings.token_embeddings,
                "embeddings.token_embeddings",
            )?,
            token_optimizer: SerializableAdam::from_adam(&embeddings.token_optimizer)?,
        })
    }

    fn deserialize_embeddings(s: &SerializableEmbeddings) -> Result<Embeddings, String> {
        let token_embeddings = rebuild_array2(
            s.token_embeddings_shape,
            &s.token_embeddings_data,
            "token_embeddings",
        )?;

        Ok(Embeddings {
            token_embeddings,
            position_encoder: PositionEncoding::new(),
            token_optimizer: s.token_optimizer.to_adam()?,
            token_grads_accum: Array2::<f32>::zeros(s.token_embeddings_shape),
            position_cache: Array2::<f32>::zeros((crate::MAX_SEQ_LEN, crate::EMBEDDING_DIM)),
        })
    }

    fn serialize_self_attention(
        attention: &SelfAttention,
    ) -> Result<SerializableSelfAttention, String> {
        Ok(SerializableSelfAttention {
            embedding_dim: attention.embedding_dim,
            num_heads: attention.num_heads,
            head_dim: attention.head_dim,
            w_q_shape: attention.w_q.dim(),
            w_q_data: collect_finite_array2_data(&attention.w_q, "self_attention.w_q")?,
            w_k_shape: attention.w_k.dim(),
            w_k_data: collect_finite_array2_data(&attention.w_k, "self_attention.w_k")?,
            w_v_shape: attention.w_v.dim(),
            w_v_data: collect_finite_array2_data(&attention.w_v, "self_attention.w_v")?,
            w_o_shape: attention.w_o.dim(),
            w_o_data: collect_finite_array2_data(&attention.w_o, "self_attention.w_o")?,
            optimizer_w_q: SerializableAdam::from_adam(&attention.optimizer_w_q)?,
            optimizer_w_k: SerializableAdam::from_adam(&attention.optimizer_w_k)?,
            optimizer_w_v: SerializableAdam::from_adam(&attention.optimizer_w_v)?,
            optimizer_w_o: SerializableAdam::from_adam(&attention.optimizer_w_o)?,
        })
    }

    fn deserialize_self_attention(s: &SerializableSelfAttention) -> Result<SelfAttention, String> {
        Ok(SelfAttention {
            embedding_dim: s.embedding_dim,
            num_heads: s.num_heads,
            head_dim: s.head_dim,
            w_q: rebuild_array2(s.w_q_shape, &s.w_q_data, "w_q")?,
            w_k: rebuild_array2(s.w_k_shape, &s.w_k_data, "w_k")?,
            w_v: rebuild_array2(s.w_v_shape, &s.w_v_data, "w_v")?,
            w_o: rebuild_array2(s.w_o_shape, &s.w_o_data, "w_o")?,
            kv_cache: None,      // KV缓存初始化为None
            use_kv_cache: false, // 默认不使用KV缓存
            freeze_updates: false,
            causal_mask_cache: std::collections::HashMap::new(), // 初始化掩码缓存
            optimizer_w_q: s.optimizer_w_q.to_adam()?,
            optimizer_w_k: s.optimizer_w_k.to_adam()?,
            optimizer_w_v: s.optimizer_w_v.to_adam()?,
            optimizer_w_o: s.optimizer_w_o.to_adam()?,
            grad_w_q_accum: Array2::zeros(s.w_q_shape),
            grad_w_k_accum: Array2::zeros(s.w_k_shape),
            grad_w_v_accum: Array2::zeros(s.w_v_shape),
            grad_w_o_accum: Array2::zeros(s.w_o_shape),
        })
    }

    fn serialize_feed_forward(ff: &FeedForward) -> Result<SerializableFeedForward, String> {
        Ok(SerializableFeedForward {
            w1_shape: ff.w1.dim(),
            w1_data: collect_finite_array2_data(&ff.w1, "feed_forward.w1")?,
            b1_shape: ff.b1.dim(),
            b1_data: collect_finite_array2_data(&ff.b1, "feed_forward.b1")?,
            w2_shape: ff.w2.dim(),
            w2_data: collect_finite_array2_data(&ff.w2, "feed_forward.w2")?,
            b2_shape: ff.b2.dim(),
            b2_data: collect_finite_array2_data(&ff.b2, "feed_forward.b2")?,
            optimizer_w1: SerializableAdam::from_adam(&ff.optimizer_w1)?,
            optimizer_b1: SerializableAdam::from_adam(&ff.optimizer_b1)?,
            optimizer_w2: SerializableAdam::from_adam(&ff.optimizer_w2)?,
            optimizer_b2: SerializableAdam::from_adam(&ff.optimizer_b2)?,
        })
    }

    fn deserialize_feed_forward(s: &SerializableFeedForward) -> Result<FeedForward, String> {
        Ok(FeedForward {
            w1: rebuild_array2(s.w1_shape, &s.w1_data, "w1")?,
            b1: rebuild_array2(s.b1_shape, &s.b1_data, "b1")?,
            w2: rebuild_array2(s.w2_shape, &s.w2_data, "w2")?,
            b2: rebuild_array2(s.b2_shape, &s.b2_data, "b2")?,
            optimizer_w1: s.optimizer_w1.to_adam()?,
            optimizer_b1: s.optimizer_b1.to_adam()?,
            optimizer_w2: s.optimizer_w2.to_adam()?,
            optimizer_b2: s.optimizer_b2.to_adam()?,
            grad_w1_accum: Array2::zeros(s.w1_shape),
            grad_b1_accum: Array2::zeros(s.b1_shape),
            grad_w2_accum: Array2::zeros(s.w2_shape),
            grad_b2_accum: Array2::zeros(s.b2_shape),
        })
    }

    fn serialize_layer_norm(ln: &LayerNorm) -> Result<SerializableLayerNorm, String> {
        Ok(SerializableLayerNorm {
            epsilon: ln.epsilon,
            gamma_shape: ln.gamma.dim(),
            gamma_data: collect_finite_array2_data(&ln.gamma, "layer_norm.gamma")?,
            beta_shape: ln.beta.dim(),
            beta_data: collect_finite_array2_data(&ln.beta, "layer_norm.beta")?,
            optimizer_gamma: SerializableAdam::from_adam(&ln.optimizer_gamma)?,
            optimizer_beta: SerializableAdam::from_adam(&ln.optimizer_beta)?,
        })
    }

    fn deserialize_layer_norm(s: &SerializableLayerNorm) -> Result<LayerNorm, String> {
        if !s.epsilon.is_finite() {
            return Err("LayerNorm epsilon 含有非有限值".to_string());
        }

        Ok(LayerNorm {
            epsilon: s.epsilon,
            gamma: rebuild_array2(s.gamma_shape, &s.gamma_data, "gamma")?,
            beta: rebuild_array2(s.beta_shape, &s.beta_data, "beta")?,
            optimizer_gamma: s.optimizer_gamma.to_adam()?,
            optimizer_beta: s.optimizer_beta.to_adam()?,
            grad_gamma_accum: Array2::zeros(s.gamma_shape),
            grad_beta_accum: Array2::zeros(s.beta_shape),
        })
    }

    fn serialize_dropout(dropout: &Dropout) -> SerializableDropout {
        SerializableDropout {
            dropout_rate: dropout.dropout_rate,
        }
    }

    fn deserialize_dropout(s: &SerializableDropout) -> Dropout {
        Dropout::new(s.dropout_rate)
    }

    fn serialize_output_projection(
        op: &OutputProjection,
    ) -> Result<SerializableOutputProjection, String> {
        Ok(SerializableOutputProjection {
            w_out_shape: op.w_out.dim(),
            w_out_data: collect_finite_array2_data(&op.w_out, "output_projection.w_out")?,
            b_out_shape: op.b_out.dim(),
            b_out_data: collect_finite_array2_data(&op.b_out, "output_projection.b_out")?,
            optimizer: SerializableAdam::from_adam(&op.optimizer)?,
            optimizer_bias: SerializableAdam::from_adam(&op.optimizer_bias)?,
        })
    }

    fn deserialize_output_projection(
        s: &SerializableOutputProjection,
        _vocab_size: usize,
    ) -> Result<OutputProjection, String> {
        Ok(OutputProjection {
            w_out: rebuild_array2(s.w_out_shape, &s.w_out_data, "w_out")?,
            b_out: rebuild_array2(s.b_out_shape, &s.b_out_data, "b_out")?,
            optimizer: s.optimizer.to_adam()?,
            optimizer_bias: s.optimizer_bias.to_adam()?,
            grad_w_out_accum: Array2::zeros(s.w_out_shape),
            grad_b_out_accum: Array2::zeros(s.b_out_shape),
        })
    }

    fn serialize_transformer_block(
        tb: &TransformerBlock,
    ) -> Result<SerializableTransformerBlock, String> {
        Ok(SerializableTransformerBlock {
            attention: Self::serialize_self_attention(&tb.attention)?,
            feed_forward: Self::serialize_feed_forward(&tb.feed_forward)?,
            dropout1: Self::serialize_dropout(&tb.dropout1),
            dropout2: Self::serialize_dropout(&tb.dropout2),
            norm1: Self::serialize_layer_norm(&tb.norm1)?,
            norm2: Self::serialize_layer_norm(&tb.norm2)?,
        })
    }

    fn deserialize_transformer_block(
        s: &SerializableTransformerBlock,
    ) -> Result<TransformerBlock, String> {
        Ok(TransformerBlock {
            attention: Self::deserialize_self_attention(&s.attention)?,
            feed_forward: Self::deserialize_feed_forward(&s.feed_forward)?,
            dropout1: Self::deserialize_dropout(&s.dropout1),
            dropout2: Self::deserialize_dropout(&s.dropout2),
            norm1: Self::deserialize_layer_norm(&s.norm1)?,
            norm2: Self::deserialize_layer_norm(&s.norm2)?,
        })
    }
}

// ============================================================================
// 完整模型的可序列化表示
// ============================================================================

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct SerializableModel {
    pub version: u32,
    pub vocab: Vocab,
    pub layers: Vec<SerializableLayer>,
    pub context_window: Vec<usize>,
    pub metadata: ModelMetadata,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_transformer_blocks: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub training_info: Option<TrainingInfo>,
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct TrainingInfo {
    pub total_epochs: usize,
    pub last_learning_rate: f32,
    pub total_training_steps: usize,
}

// ============================================================================
// 主要 API
// ============================================================================

/// 先把运行中的 LLM 显式展开成可序列化结构，避免 binary/json 两条保存路径重复拼装模型。
fn build_serializable_model(model: &LLM) -> Result<SerializableModel, std::io::Error> {
    let mut serializable_layers = Vec::with_capacity(model.network.len());
    for (i, layer) in model.network.iter().enumerate() {
        print!("   [{}] 序列化 {} 层...", i + 1, layer.layer_type());
        match SerializableLayer::from_layer(layer) {
            Ok(serialized_layer) => {
                serializable_layers.push(serialized_layer);
                println!(" ✓");
            }
            Err(error) => {
                println!(" ✗");
                return Err(std::io::Error::other(format!(
                    "Failed to serialize layer {}: {}",
                    i, error
                )));
            }
        }
    }

    Ok(SerializableModel {
        version: 1,
        vocab: model.vocab.clone(),
        layers: serializable_layers,
        context_window: model.context_window.clone(),
        metadata: ModelMetadata {
            embedding_dim: EMBEDDING_DIM,
            hidden_dim: HIDDEN_DIM,
            num_transformer_blocks: count_transformer_blocks(&model.network),
            vocab_size: model.vocab.len(),
            max_seq_len: model.max_context_length,
            training_info: None,
        },
    })
}

/// 把磁盘上的 SerializableModel 重建回可运行的 LLM，避免 binary/json 两条加载路径重复回填运行时字段。
fn build_llm_from_serializable(
    serializable_model: SerializableModel,
) -> Result<LLM, std::io::Error> {
    let SerializableModel {
        vocab,
        layers,
        context_window,
        metadata,
        ..
    } = serializable_model;

    let vocab_size = vocab.words.len();
    let vocab_len = vocab.len();
    let mut network: Vec<Box<dyn Layer>> = Vec::with_capacity(layers.len());
    for (i, serializable_layer) in layers.iter().enumerate() {
        print!("   [{}] 重建层...", i + 1);
        let layer = serializable_layer.to_layer(vocab_len).map_err(|error| {
            std::io::Error::other(format!("Failed to rebuild layer {}: {}", i, error))
        })?;
        network.push(layer);
        println!(" ✓");
    }

    LLM::validate_network_topology(&network)
        .map_err(|error| std::io::Error::other(format!("Invalid model topology: {}", error)))?;

    Ok(LLM {
        vocab,
        network,
        context_window,
        max_context_length: metadata.max_seq_len,
        training: false,
        sampling_prob_buffer: Vec::with_capacity(vocab_size),
        sampling_idx_buffer: Vec::with_capacity(vocab_size),
        beam_candidates_buffer: Vec::with_capacity(50),
    })
}

/// 保存模型到二进制文件
pub fn save_model_binary<P: AsRef<Path>>(
    model: &LLM,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 开始保存模型到二进制文件...");
    println!("   路径: {:?}", path.as_ref());

    let serializable_model = build_serializable_model(model)?;

    print!("   写入文件...");
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    let config = bincode::config::standard();
    bincode::encode_into_std_write(&serializable_model, &mut writer, config)?;
    writer.flush()?;
    println!(" ✓");

    let file_size = std::fs::metadata(path.as_ref())?.len();
    println!("   文件大小: {:.2} MB", file_size as f64 / 1_048_576.0);
    println!("✅ 模型保存成功!");

    Ok(())
}

/// 从二进制文件加载模型
pub fn load_model_binary<P: AsRef<Path>>(path: P) -> Result<LLM, Box<dyn std::error::Error>> {
    println!("📂 开始从二进制文件加载模型...");
    println!("   路径: {:?}", path.as_ref());

    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file).take(BINCODE_DECODE_LIMIT_BYTES as u64);

    let config = bincode::config::standard().with_limit::<BINCODE_DECODE_LIMIT_BYTES>();
    let serializable_model: SerializableModel = bincode::decode_from_std_read(&mut reader, config)?;

    println!("   ✓ 文件读取成功");
    println!("   模型版本: {}", serializable_model.version);
    println!("   词汇量: {}", serializable_model.vocab.len());
    println!("   网络层数: {}", serializable_model.layers.len());

    let llm = build_llm_from_serializable(serializable_model)?;

    println!("✅ 模型加载成功!");
    println!("   总参数量: {}", llm.total_parameters());

    Ok(llm)
}

/// 保存模型到 JSON 文件
pub fn save_model_json<P: AsRef<Path>>(
    model: &LLM,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 开始保存模型到 JSON 文件...");
    println!("   路径: {:?}", path.as_ref());

    let serializable_model = build_serializable_model(model)?;

    print!("   写入 JSON 文件...");
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &serializable_model)?;
    writer.flush()?;
    println!(" ✓");

    let file_size = std::fs::metadata(path.as_ref())?.len();
    println!("   文件大小: {:.2} MB", file_size as f64 / 1_048_576.0);
    println!("✅ 模型保存成功!");

    Ok(())
}

/// 从 JSON 文件加载模型
#[allow(dead_code)]
pub fn load_model_json<P: AsRef<Path>>(path: P) -> Result<LLM, Box<dyn std::error::Error>> {
    println!("📂 开始从 JSON 文件加载模型...");
    println!("   路径: {:?}", path.as_ref());

    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file).take(JSON_DECODE_LIMIT_BYTES);
    let serializable_model: SerializableModel = serde_json::from_reader(reader)?;

    println!("   ✓ 文件读取成功");
    println!("   模型版本: {}", serializable_model.version);
    println!("   词汇量: {}", serializable_model.vocab.len());
    println!("   网络层数: {}", serializable_model.layers.len());

    let llm = build_llm_from_serializable(serializable_model)?;

    println!("✅ 模型加载成功!");
    println!("   总参数量: {}", llm.total_parameters());

    Ok(llm)
}

/// 自动选择加载方法
#[allow(dead_code)]
pub fn load_model_auto<P: AsRef<Path>>(path: P) -> Result<LLM, Box<dyn std::error::Error>> {
    let path_str = path.as_ref().to_str().unwrap_or("");

    if path_str.ends_with(".json") {
        load_model_json(path)
    } else {
        load_model_binary(path)
    }
}
