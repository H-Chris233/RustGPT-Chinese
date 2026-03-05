// ============================================================================
// 模型序列化模块 - 支持二进制和 JSON 两种格式
// ============================================================================
//
// 本模块实现了 RustGPT-Chinese 模型的持久化功能,支持两种序列化格式:
//
// 1. **二进制格式** (推荐用于生产):
//    - 使用 bincode 序列化,文件小、速度快
//    - 保存完整的优化器状态(Adam 的 m、v 动量)
//    - 支持断点续训
//    - 文件扩展名: .bin
//
// 2. **JSON 格式** (推荐用于调试):
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
    pub fn from_adam(adam: &Adam) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: adam.timestep,
            m_shape: adam.m.dim(),
            m_data: adam
                .m
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            v_shape: adam.v.dim(),
            v_data: adam
                .v
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
        }
    }

    pub fn to_adam(&self) -> Adam {
        let m = match Array2::from_shape_vec(self.m_shape, self.m_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct m matrix: {}", e);
                Array2::zeros(self.m_shape)
            }
        };
        let v = match Array2::from_shape_vec(self.v_shape, self.v_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct v matrix: {}", e);
                Array2::zeros(self.v_shape)
            }
        };

        Adam {
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            timestep: self.timestep,
            m,
            v,
        }
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
            )));
        }

        if let Some(transformer) = layer.as_any().downcast_ref::<TransformerBlock>() {
            return Ok(SerializableLayer::TransformerBlock(
                Self::serialize_transformer_block(transformer),
            ));
        }

        if let Some(output_proj) = layer.as_any().downcast_ref::<OutputProjection>() {
            return Ok(SerializableLayer::OutputProjection(
                Self::serialize_output_projection(output_proj),
            ));
        }

        Err(format!("Unsupported layer type: {}", layer.layer_type()))
    }

    pub fn to_layer(&self, vocab_size: usize) -> Box<dyn Layer> {
        match self {
            SerializableLayer::Embeddings(s) => Box::new(Self::deserialize_embeddings(s)),
            SerializableLayer::TransformerBlock(s) => {
                Box::new(Self::deserialize_transformer_block(s))
            }
            SerializableLayer::OutputProjection(s) => {
                Box::new(Self::deserialize_output_projection(s, vocab_size))
            }
        }
    }

    fn serialize_embeddings(embeddings: &Embeddings) -> SerializableEmbeddings {
        SerializableEmbeddings {
            token_embeddings_shape: embeddings.token_embeddings.dim(),
            token_embeddings_data: embeddings
                .token_embeddings
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            token_optimizer: SerializableAdam::from_adam(&embeddings.token_optimizer),
        }
    }

    fn deserialize_embeddings(s: &SerializableEmbeddings) -> Embeddings {
        let token_embeddings =
            match Array2::from_shape_vec(s.token_embeddings_shape, s.token_embeddings_data.clone())
            {
                Ok(arr) => arr,
                Err(e) => {
                    log::error!("Failed to reconstruct token_embeddings: {}", e);
                    Array2::zeros(s.token_embeddings_shape)
                }
            };

        Embeddings {
            token_embeddings,
            position_encoder: PositionEncoding::new(),
            cached_input: None,
            cached_input_batch: None,
            token_optimizer: s.token_optimizer.to_adam(),
            token_grads_accum: Array2::<f32>::zeros(s.token_embeddings_shape),
            position_cache: Array2::<f32>::zeros((crate::MAX_SEQ_LEN, crate::EMBEDDING_DIM)),
        }
    }

    fn serialize_self_attention(attention: &SelfAttention) -> SerializableSelfAttention {
        SerializableSelfAttention {
            embedding_dim: attention.embedding_dim,
            num_heads: attention.num_heads,
            head_dim: attention.head_dim,
            w_q_shape: attention.w_q.dim(),
            w_q_data: attention
                .w_q
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            w_k_shape: attention.w_k.dim(),
            w_k_data: attention
                .w_k
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            w_v_shape: attention.w_v.dim(),
            w_v_data: attention
                .w_v
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            w_o_shape: attention.w_o.dim(),
            w_o_data: attention
                .w_o
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            optimizer_w_q: SerializableAdam::from_adam(&attention.optimizer_w_q),
            optimizer_w_k: SerializableAdam::from_adam(&attention.optimizer_w_k),
            optimizer_w_v: SerializableAdam::from_adam(&attention.optimizer_w_v),
            optimizer_w_o: SerializableAdam::from_adam(&attention.optimizer_w_o),
        }
    }

    fn deserialize_self_attention(s: &SerializableSelfAttention) -> SelfAttention {
        let w_q = match Array2::from_shape_vec(s.w_q_shape, s.w_q_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w_q: {}", e);
                Array2::zeros(s.w_q_shape)
            }
        };
        let w_k = match Array2::from_shape_vec(s.w_k_shape, s.w_k_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w_k: {}", e);
                Array2::zeros(s.w_k_shape)
            }
        };
        let w_v = match Array2::from_shape_vec(s.w_v_shape, s.w_v_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w_v: {}", e);
                Array2::zeros(s.w_v_shape)
            }
        };
        let w_o = match Array2::from_shape_vec(s.w_o_shape, s.w_o_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w_o: {}", e);
                Array2::zeros(s.w_o_shape)
            }
        };

        SelfAttention {
            embedding_dim: s.embedding_dim,
            num_heads: s.num_heads,
            head_dim: s.head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attention_weights: None,
            cached_attention_output: None,
            kv_cache: None,      // KV缓存初始化为None
            use_kv_cache: false, // 默认不使用KV缓存
            freeze_updates: false,
            causal_mask_cache: std::collections::HashMap::new(), // 初始化掩码缓存
            optimizer_w_q: s.optimizer_w_q.to_adam(),
            optimizer_w_k: s.optimizer_w_k.to_adam(),
            optimizer_w_v: s.optimizer_w_v.to_adam(),
            optimizer_w_o: s.optimizer_w_o.to_adam(),
            grad_w_q_accum: Array2::zeros(s.w_q_shape),
            grad_w_k_accum: Array2::zeros(s.w_k_shape),
            grad_w_v_accum: Array2::zeros(s.w_v_shape),
            grad_w_o_accum: Array2::zeros(s.w_o_shape),
        }
    }

    fn serialize_feed_forward(ff: &FeedForward) -> SerializableFeedForward {
        SerializableFeedForward {
            w1_shape: ff.w1.dim(),
            w1_data: ff
                .w1
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            b1_shape: ff.b1.dim(),
            b1_data: ff
                .b1
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            w2_shape: ff.w2.dim(),
            w2_data: ff
                .w2
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            b2_shape: ff.b2.dim(),
            b2_data: ff
                .b2
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            optimizer_w1: SerializableAdam::from_adam(&ff.optimizer_w1),
            optimizer_b1: SerializableAdam::from_adam(&ff.optimizer_b1),
            optimizer_w2: SerializableAdam::from_adam(&ff.optimizer_w2),
            optimizer_b2: SerializableAdam::from_adam(&ff.optimizer_b2),
        }
    }

    fn deserialize_feed_forward(s: &SerializableFeedForward) -> FeedForward {
        let w1 = match Array2::from_shape_vec(s.w1_shape, s.w1_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w1: {}", e);
                Array2::zeros(s.w1_shape)
            }
        };
        let b1 = match Array2::from_shape_vec(s.b1_shape, s.b1_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct b1: {}", e);
                Array2::zeros(s.b1_shape)
            }
        };
        let w2 = match Array2::from_shape_vec(s.w2_shape, s.w2_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w2: {}", e);
                Array2::zeros(s.w2_shape)
            }
        };
        let b2 = match Array2::from_shape_vec(s.b2_shape, s.b2_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct b2: {}", e);
                Array2::zeros(s.b2_shape)
            }
        };

        FeedForward {
            w1,
            b1,
            w2,
            b2,
            input: None,
            hidden_pre_activation: None,
            hidden_post_activation: None,
            optimizer_w1: s.optimizer_w1.to_adam(),
            optimizer_b1: s.optimizer_b1.to_adam(),
            optimizer_w2: s.optimizer_w2.to_adam(),
            optimizer_b2: s.optimizer_b2.to_adam(),
            grad_w1_accum: Array2::zeros(s.w1_shape),
            grad_b1_accum: Array2::zeros(s.b1_shape),
            grad_w2_accum: Array2::zeros(s.w2_shape),
            grad_b2_accum: Array2::zeros(s.b2_shape),
        }
    }

    fn serialize_layer_norm(ln: &LayerNorm) -> SerializableLayerNorm {
        SerializableLayerNorm {
            epsilon: ln.epsilon,
            gamma_shape: ln.gamma.dim(),
            gamma_data: ln
                .gamma
                .iter()
                .map(|&x| if x.is_finite() { x } else { 1.0 })
                .collect(),
            beta_shape: ln.beta.dim(),
            beta_data: ln
                .beta
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            optimizer_gamma: SerializableAdam::from_adam(&ln.optimizer_gamma),
            optimizer_beta: SerializableAdam::from_adam(&ln.optimizer_beta),
        }
    }

    fn deserialize_layer_norm(s: &SerializableLayerNorm) -> LayerNorm {
        let gamma = match Array2::from_shape_vec(s.gamma_shape, s.gamma_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct gamma: {}", e);
                Array2::zeros(s.gamma_shape)
            }
        };
        let beta = match Array2::from_shape_vec(s.beta_shape, s.beta_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct beta: {}", e);
                Array2::zeros(s.beta_shape)
            }
        };

        LayerNorm {
            epsilon: s.epsilon,
            gamma,
            beta,
            cached_input: None,
            cached_mean: None,
            cached_inv_std: None,
            optimizer_gamma: s.optimizer_gamma.to_adam(),
            optimizer_beta: s.optimizer_beta.to_adam(),
            grad_gamma_accum: Array2::zeros(s.gamma_shape),
            grad_beta_accum: Array2::zeros(s.beta_shape),
        }
    }

    fn serialize_dropout(dropout: &Dropout) -> SerializableDropout {
        SerializableDropout {
            dropout_rate: dropout.dropout_rate,
        }
    }

    fn deserialize_dropout(s: &SerializableDropout) -> Dropout {
        Dropout::new(s.dropout_rate)
    }

    fn serialize_output_projection(op: &OutputProjection) -> SerializableOutputProjection {
        SerializableOutputProjection {
            w_out_shape: op.w_out.dim(),
            w_out_data: op
                .w_out
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            b_out_shape: op.b_out.dim(),
            b_out_data: op
                .b_out
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            optimizer: SerializableAdam::from_adam(&op.optimizer),
        }
    }

    fn deserialize_output_projection(
        s: &SerializableOutputProjection,
        _vocab_size: usize,
    ) -> OutputProjection {
        let w_out = match Array2::from_shape_vec(s.w_out_shape, s.w_out_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct w_out: {}", e);
                Array2::zeros(s.w_out_shape)
            }
        };
        let b_out = match Array2::from_shape_vec(s.b_out_shape, s.b_out_data.clone()) {
            Ok(arr) => arr,
            Err(e) => {
                log::error!("Failed to reconstruct b_out: {}", e);
                Array2::zeros(s.b_out_shape)
            }
        };

        OutputProjection {
            w_out,
            b_out,
            optimizer: s.optimizer.to_adam(),
            cached_input: None,
            grad_w_out_accum: Array2::zeros(s.w_out_shape),
            grad_b_out_accum: Array2::zeros(s.b_out_shape),
        }
    }

    fn serialize_transformer_block(tb: &TransformerBlock) -> SerializableTransformerBlock {
        SerializableTransformerBlock {
            attention: Self::serialize_self_attention(&tb.attention),
            feed_forward: Self::serialize_feed_forward(&tb.feed_forward),
            dropout1: Self::serialize_dropout(&tb.dropout1),
            dropout2: Self::serialize_dropout(&tb.dropout2),
            norm1: Self::serialize_layer_norm(&tb.norm1),
            norm2: Self::serialize_layer_norm(&tb.norm2),
        }
    }

    fn deserialize_transformer_block(s: &SerializableTransformerBlock) -> TransformerBlock {
        TransformerBlock {
            attention: Self::deserialize_self_attention(&s.attention),
            feed_forward: Self::deserialize_feed_forward(&s.feed_forward),
            dropout1: Self::deserialize_dropout(&s.dropout1),
            dropout2: Self::deserialize_dropout(&s.dropout2),
            norm1: Self::deserialize_layer_norm(&s.norm1),
            norm2: Self::deserialize_layer_norm(&s.norm2),
        }
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

/// 保存模型到二进制文件
pub fn save_model_binary<P: AsRef<Path>>(
    model: &LLM,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 开始保存模型到二进制文件...");
    println!("   路径: {:?}", path.as_ref());

    let mut serializable_layers = Vec::new();
    for (i, layer) in model.network.iter().enumerate() {
        print!("   [{}] 序列化 {} 层...", i + 1, layer.layer_type());
        match SerializableLayer::from_layer(layer) {
            Ok(s_layer) => {
                serializable_layers.push(s_layer);
                println!(" ✓");
            }
            Err(e) => {
                println!(" ✗");
                return Err(format!("Failed to serialize layer {}: {}", i, e).into());
            }
        }
    }

    let serializable_model = SerializableModel {
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
    };

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

    let mut network: Vec<Box<dyn Layer>> = Vec::new();
    for (i, s_layer) in serializable_model.layers.iter().enumerate() {
        print!("   [{}] 重建层...", i + 1);
        let layer = s_layer.to_layer(serializable_model.vocab.len());
        network.push(layer);
        println!(" ✓");
    }

    let vocab_size = serializable_model.vocab.words.len();
    let llm = LLM {
        vocab: serializable_model.vocab,
        network,
        context_window: serializable_model.context_window,
        max_context_length: serializable_model.metadata.max_seq_len,
        training: false,
        parallel_training: true,
        sampling_prob_buffer: Vec::with_capacity(vocab_size),
        sampling_idx_buffer: Vec::with_capacity(vocab_size),
        beam_candidates_buffer: Vec::with_capacity(50),
    };

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

    let mut serializable_layers = Vec::new();
    for (i, layer) in model.network.iter().enumerate() {
        print!("   [{}] 序列化 {} 层...", i + 1, layer.layer_type());
        match SerializableLayer::from_layer(layer) {
            Ok(s_layer) => {
                serializable_layers.push(s_layer);
                println!(" ✓");
            }
            Err(e) => {
                println!(" ✗");
                return Err(format!("Failed to serialize layer {}: {}", i, e).into());
            }
        }
    }

    let serializable_model = SerializableModel {
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
    };

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

    let mut network: Vec<Box<dyn Layer>> = Vec::new();
    for (i, s_layer) in serializable_model.layers.iter().enumerate() {
        print!("   [{}] 重建层...", i + 1);
        let layer = s_layer.to_layer(serializable_model.vocab.len());
        network.push(layer);
        println!(" ✓");
    }

    let vocab_size = serializable_model.vocab.words.len();
    let llm = LLM {
        vocab: serializable_model.vocab,
        network,
        context_window: serializable_model.context_window,
        max_context_length: serializable_model.metadata.max_seq_len,
        training: false,
        parallel_training: true,
        sampling_prob_buffer: Vec::with_capacity(vocab_size),
        sampling_idx_buffer: Vec::with_capacity(vocab_size),
        beam_candidates_buffer: Vec::with_capacity(50),
    };

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
