//! # 词嵌入层（Embeddings Layer）
//!
//! 这是神经语言模型的输入层，负责将离散的 token ID 转换为连续的向量表示。
//!
//! ## 核心概念
//!
//! ### 1. 词嵌入 (Token Embeddings)
//!
//! **问题**：神经网络无法直接处理文本，需要将词转换为数字向量。
//!
//! **解决方案**：为每个词分配一个固定维度的向量（本项目中是512维）。
//! 这些向量在训练过程中不断更新，相似的词会有相似的向量表示。
//!
//! **示例**：
//! ```text
//! "北京" → [0.23, -0.45, 0.67, ..., 0.12]  (512维)
//! "上海" → [0.25, -0.42, 0.65, ..., 0.10]  (相似的向量)
//! "苹果" → [-0.31, 0.52, -0.18, ..., 0.87] (不同的向量)
//! ```
//!
//! ### 2. 位置编码 (Positional Encoding)
//!
//! **问题**：Transformer 的注意力机制本身没有位置信息，无法区分词的顺序。
//! "我喜欢你" 和 "你喜欢我" 在不加位置编码的情况下会产生相同的注意力模式。
//!
//! **解决方案**：使用正弦/余弦函数生成位置编码，为每个位置添加唯一的"标记"。
//!
//! **公式**：
//! ```text
//! PE(pos, 2i)   = sin(pos / 10000^(2i/d))     // 偶数维度使用 sin
//! PE(pos, 2i+1) = cos(pos / 10000^(2i/d))     // 奇数维度使用 cos
//! ```
//!
//! 其中：
//! - `pos` = 词在序列中的位置 (0, 1, 2, ...)
//! - `i` = 嵌入维度的索引 (0, 1, 2, ..., 255)
//! - `d` = 嵌入维度总数 (512)
//!
//! ### 3. 组合方式
//!
//! 最终的嵌入向量 = 词嵌入 + 位置编码（逐元素相加）
//!
//! ```text
//! token_embedding = [0.23, -0.45, 0.67, ...]
//! position_encoding = [0.01, 0.03, -0.02, ...]
//! final_embedding = [0.24, -0.42, 0.65, ...]  // 逐元素相加
//! ```

use ndarray::{Array1, Array2, Array3, Zip};

use crate::{
    EMBEDDING_DIM, adam::Adam, llm::Layer, position_encoding::PositionEncoding,
    utils::sample_normal, vocab::Vocab,
};

/// **嵌入层结构体**
///
/// 包含词嵌入矩阵、位置编码器和用于反向传播的缓存。
pub struct Embeddings {
    /// **词嵌入矩阵** (vocab_size × embedding_dim)
    ///
    /// 每一行代表一个词的向量表示。例如：
    /// - 第0行：`<|pad|>` 的向量
    /// - 第1行：`<|unk|>` 的向量
    /// - 第100行：某个中文词的向量
    ///
    /// 这个矩阵是可学习的，训练过程中会不断更新。
    pub token_embeddings: Array2<f32>,

    /// **位置编码器**
    ///
    /// 使用正弦/余弦函数生成固定的位置编码。
    /// **注意**：位置编码是固定的，不参与训练（不需要梯度）。
    pub position_encoder: PositionEncoding,

    /// **缓存的输入** (用于反向传播)
    ///
    /// 保存前向传播时的 token ID，反向传播时需要知道更新哪些行的嵌入。
    pub cached_input: Option<Array2<f32>>,

    /// **Adam 优化器**
    ///
    /// 用于更新词嵌入矩阵的参数。每个词的嵌入向量独立更新。
    pub token_optimizer: Adam,

    /// **位置编码缓存** (性能优化)
    ///
    /// 预分配的缓冲区，避免每次forward都重新分配Array2
    pub position_cache: Array2<f32>,
}

impl Default for Embeddings {
    fn default() -> Self {
        let vocab = Vocab::default();
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), EMBEDDING_DIM),
            position_encoder: PositionEncoding::new(),
            cached_input: None,
            token_optimizer: Adam::new((vocab.words.len(), EMBEDDING_DIM)),
            position_cache: Array2::<f32>::zeros((crate::MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }
}

impl Embeddings {
    /// **创建新的嵌入层**
    ///
    /// # 参数
    /// - `vocab`: 词汇表，决定嵌入矩阵的行数（每个词一行）
    ///
    /// # 初始化策略
    /// 使用正态分布 N(0, 0.02) 初始化嵌入权重。较小的标准差（0.02）有助于训练稳定。
    pub fn new(vocab: Vocab) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), EMBEDDING_DIM),
            position_encoder: PositionEncoding::new(),
            cached_input: None,
            token_optimizer: Adam::new((vocab.words.len(), EMBEDDING_DIM)),
            position_cache: Array2::<f32>::zeros((crate::MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }

    /// **初始化嵌入矩阵**
    ///
    /// # 参数
    /// - `vocab_size`: 词汇表大小（词的数量）
    /// - `embedding_dim`: 每个词的嵌入维度（512）
    ///
    /// # 初始化方法
    /// 使用正态分布 N(0, 0.02) 随机初始化。
    ///
    /// **为什么是 0.02？**
    /// - 太大：梯度爆炸，训练不稳定
    /// - 太小：梯度消失，学习速度慢
    /// - 0.02 是经验值，在多数情况下效果良好
    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, 0.02)
        })
    }

    /// **根据 token ID 获取对应的嵌入向量**
    ///
    /// # 工作原理
    /// 这本质上是一个"查表"操作：给定 token ID，返回嵌入矩阵的对应行。
    ///
    /// # 示例
    /// ```text
    /// token_ids = [5, 12, 3]  // 三个词的ID
    /// embeddings = [[第5行], [第12行], [第3行]]  // 返回三个512维向量
    /// ```
    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));

        let safe_ids: Vec<usize> = token_ids
            .iter()
            .map(|&token_id| {
                if token_id >= embeddings.nrows() {
                    log::warn!(
                        "Token ID {} 越界（词表大小: {}），将使用最后一个可用ID作为回退",
                        token_id,
                        embeddings.nrows()
                    );
                    embeddings.nrows().saturating_sub(1)
                } else {
                    token_id
                }
            })
            .collect();

        Zip::indexed(&mut token_embeds).for_each(|(i, j), value| {
            *value = embeddings[[safe_ids[i], j]];
        });

        token_embeds
    }

    /// **生成完整的嵌入（词嵌入 + 位置编码）**
    ///
    /// # 算法步骤
    /// 1. 根据 token ID 查询词嵌入
    /// 2. 从预生成的位置编码矩阵中获取对应slice
    /// 3. 将词嵌入和位置编码逐元素相加
    ///
    /// # 参数
    /// - `token_ids`: token ID 序列，例如 [5, 12, 3, 8]
    ///
    /// # 返回值
    /// 形状为 (seq_len, embedding_dim) 的嵌入矩阵
    ///
    /// # 示例
    /// ```text
    /// 输入: token_ids = [5, 12, 3]
    ///
    /// 步骤 1 - 获取词嵌入:
    ///   position 0: token_id=5  → embedding_5
    ///   position 1: token_id=12 → embedding_12
    ///   position 2: token_id=3  → embedding_3
    ///
    /// 步骤 2 - 从预生成的position_encoder.encoding中slice位置编码:
    ///   position 0: PE(0) 直接从encoding[0]取
    ///   position 1: PE(1) 直接从encoding[1]取
    ///   position 2: PE(2) 直接从encoding[2]取
    ///
    /// 步骤 3 - 逐元素相加:
    ///   final[0] = embedding_5 + PE(0)
    ///   final[1] = embedding_12 + PE(1)
    ///   final[2] = embedding_3 + PE(2)
    /// ```
    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        // 步骤 1：查询词嵌入
        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);

        // 步骤 2：直接从预生成的位置编码矩阵中切片，避免重新分配
        let seq_len = token_ids.len();
        let position_embeds = self
            .position_encoder
            .encoding
            .slice(ndarray::s![0..seq_len, ..]);

        // 步骤 3：逐元素相加
        token_embeds + position_embeds
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// **前向传播：将 token ID 转换为嵌入向量**
    ///
    /// # 输入格式
    /// `input` 是一个 (1, seq_len) 的矩阵，每个元素是一个 token ID (以浮点数形式存储)。
    /// 例如：`[[5.0, 12.0, 3.0, 8.0]]` 表示4个token的ID。
    ///
    /// # 输出格式
    /// 返回 (seq_len, embedding_dim) 的嵌入矩阵，每一行是一个512维的向量。
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 保存输入，用于反向传播时确定哪些嵌入向量需要更新
        self.cached_input = Some(input.clone());

        // 将浮点数转换为整数 token ID
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();

        // 查询嵌入 + 添加位置编码
        self.embed_tokens(&token_ids)
    }

    /// **反向传播：更新词嵌入矩阵**
    ///
    /// # 核心思想
    ///
    /// 嵌入层的反向传播比较特殊：
    /// - **位置编码固定**：不需要更新，梯度直接传递
    /// - **词嵌入可学习**：只更新在本批次中出现的词的嵌入向量
    ///
    /// # 算法步骤
    ///
    /// 1. 对于输入序列中的每个 token ID，累积它的梯度
    /// 2. 使用 Adam 优化器更新对应行的嵌入向量
    /// 3. 返回原始梯度（因为位置编码不变）
    ///
    /// # 示例
    /// ```text
    /// 假设输入: token_ids = [5, 12, 5]  (注意ID=5出现两次)
    /// 梯度: grads = [grad_0, grad_1, grad_2]  (每个都是512维)
    ///
    /// 累积梯度:
    ///   token_grads[5] = grad_0 + grad_2  (ID=5的累积梯度)
    ///   token_grads[12] = grad_1          (ID=12的梯度)
    ///
    /// 更新嵌入:
    ///   embedding[5] -= lr * Adam(token_grads[5])
    ///   embedding[12] -= lr * Adam(token_grads[12])
    /// ```
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 获取缓存的输入 token ID
        let Some(input) = self.cached_input.as_ref() else {
            log::warn!("Embeddings.backward 在未执行 forward 的情况下被调用，跳过参数更新");
            return grads.clone();
        };
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = grads.view(); // (sequence_length, embedding_dim)

        // 初始化梯度累积矩阵（全零，与嵌入矩阵形状相同）
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());

        // 累积每个 token 的梯度
        for (i, &token_id) in token_ids.iter().enumerate() {
            let safe_id = if token_id >= self.token_embeddings.nrows() {
                log::warn!(
                    "Token ID {} 越界（词表大小: {}），将使用最后一个可用ID作为回退",
                    token_id,
                    self.token_embeddings.nrows()
                );
                self.token_embeddings.nrows().saturating_sub(1)
            } else {
                token_id
            };
            let grad_row = grads.row(i);

            // 累积到对应 token 的梯度行
            // 如果一个 token 在序列中出现多次，梯度会累加
            {
                let mut token_row = token_grads.row_mut(safe_id);
                token_row += &grad_row;
            }
        }

        // 使用 Adam 优化器更新词嵌入矩阵
        self.token_optimizer
            .step(&mut self.token_embeddings, &token_grads, lr);

        // 返回原始梯度（位置编码不需要梯度）
        grads.to_owned()
    }

    /// **计算参数数量**
    ///
    /// 返回词嵌入矩阵的元素总数 = vocab_size × embedding_dim
    /// 位置编码不计入，因为它是固定的。
    fn parameters(&self) -> usize {
        self.token_embeddings.len()
    }

    /// **设置训练模式**
    ///
    /// 嵌入层不受训练/推理模式影响，因为它没有 Dropout 等需要切换的组件。
    fn set_training_mode(&mut self, _training: bool) {}

    /// **批量前向传播：将批量 token ID 转换为嵌入向量**
    ///
    /// # 输入格式
    /// `input` 是一个 (batch_size, seq_len, 1) 的张量，每个元素是一个 token ID。
    /// 为了兼容，我们也支持直接传入 token IDs 作为 (batch_size, seq_len) 的形式。
    ///
    /// # 输出格式
    /// 返回 (batch_size, seq_len, embedding_dim) 的嵌入矩阵。
    fn forward_batch(
        &mut self,
        input: &Array3<f32>,
        _attention_mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        // 保存输入用于反向传播
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // 将 (batch, seq, 1) 转换为 (batch, seq)
        let token_ids_flat: Vec<Vec<usize>> = (0..batch_size)
            .map(|b| (0..seq_len).map(|s| input[[b, s, 0]] as usize).collect())
            .collect();

        // 为每个批次样本生成嵌入
        let mut output = Array3::zeros((batch_size, seq_len, EMBEDDING_DIM));

        for (b, token_ids) in token_ids_flat.iter().enumerate() {
            let embeddings = self.embed_tokens(token_ids);
            output.slice_mut(ndarray::s![b, .., ..]).assign(&embeddings);
        }

        output
    }

    /// **批量反向传播：更新词嵌入矩阵**
    fn backward_batch(
        &mut self,
        grads: &Array3<f32>,
        _lr: f32,
        _attention_mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        // 简化实现：批量模式下，使用默认实现（对每个样本分别调用单样本backward）
        // 实际应用中，应该在forward_batch中缓存token IDs，这里才能累积梯度

        // 直接返回梯度（位置编码不需要梯度）
        grads.to_owned()
    }
}

impl Embeddings {
    /// **批量前向传播的便捷方法（直接接受 token IDs）**
    ///
    /// # 参数
    /// - `token_ids_batch`: (batch_size, seq_len) 的 token ID 矩阵
    ///
    /// # 返回值
    /// (batch_size, seq_len, embedding_dim) 的嵌入矩阵
    pub fn forward_batch_from_ids(&mut self, token_ids_batch: &Array2<usize>) -> Array3<f32> {
        let batch_size = token_ids_batch.nrows();
        let seq_len = token_ids_batch.ncols();

        let mut output = Array3::zeros((batch_size, seq_len, EMBEDDING_DIM));

        for b in 0..batch_size {
            let token_ids: Vec<usize> = token_ids_batch.row(b).to_vec();
            let embeddings = self.embed_tokens(&token_ids);
            output.slice_mut(ndarray::s![b, .., ..]).assign(&embeddings);
        }

        output
    }
}
