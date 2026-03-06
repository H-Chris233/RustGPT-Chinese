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
//! "北京" → [0.23, -0.45, 0.67, ..., 0.12]  (EMBEDDING_DIM 维)
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
//! - `d` = 嵌入维度总数 (EMBEDDING_DIM)
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

use ndarray::{Array2, Array3, Zip};

use crate::{
    EMBEDDING_DIM, adam::Adam, llm::Layer, position_encoding::PositionEncoding,
    utils::sample_normal, vocab::Vocab,
};

/// Embeddings 层的前向/反向上下文。
///
/// 该上下文保存了本次 forward 的 token_id 序列，backward 需要它将梯度累积到
/// `token_embeddings` 的对应行。
#[derive(Clone)]
struct EmbeddingsContext {
    token_ids: Vec<usize>,
}

/// **嵌入层结构体**
///
/// 包含词嵌入矩阵、位置编码器，以及训练所需的优化器/梯度累积 buffer。
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

    /// **Adam 优化器**
    ///
    /// 用于更新词嵌入矩阵的参数。每个词的嵌入向量独立更新。
    pub token_optimizer: Adam,

    // =====================================================================
    // 梯度累积支持（Gradient Accumulation）
    // =====================================================================
    //
    // 与 OutputProjection 同理：为了实现“正确的梯度累积”，我们需要在每个 micro-batch
    // 立刻执行 backward（保证“本次 forward 的 ctx/输入”与 grads 一一对应），但把参数梯度累加起来，
    // 等到累积步结束再更新一次。
    pub token_grads_accum: Array2<f32>,

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
            token_optimizer: Adam::new((vocab.words.len(), EMBEDDING_DIM)),
            token_grads_accum: Array2::<f32>::zeros((vocab.words.len(), EMBEDDING_DIM)),
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
        let vocab_size = vocab.words.len();
        Self {
            token_embeddings: Self::init_embeddings(vocab_size, EMBEDDING_DIM),
            position_encoder: PositionEncoding::new(),
            token_optimizer: Adam::new((vocab_size, EMBEDDING_DIM)),
            token_grads_accum: Array2::<f32>::zeros((vocab_size, EMBEDDING_DIM)),
            position_cache: Array2::<f32>::zeros((crate::MAX_SEQ_LEN, EMBEDDING_DIM)),
        }
    }

    /// 清空梯度累积 buffer。
    pub fn zero_grad_accum(&mut self) {
        self.token_grads_accum.fill(0.0);
    }

    /// 将本次样本的梯度累积到 `token_grads_accum`（核心逻辑，ctx 驱动）。
    ///
    /// 为什么需要把这段逻辑抽出来？
    /// - Embeddings 的参数是一个大矩阵：`token_embeddings[vocab_size, d_model]`；
    /// - backward 时我们只更新“本次序列里出现过的 token 对应的行”；
    /// - 因此只要我们能拿到 token_id 序列，就能完成累积；
    /// - 新版推荐从 `ctx` 取 token_ids，避免依赖层内 `cached_*` 字段（批量/并发时会被覆盖）。
    fn accumulate_token_grads(&mut self, token_ids: &[usize], grads: &Array2<f32>) {
        let grads = grads.view(); // (seq_len, embedding_dim)

        if token_ids.len() != grads.nrows() {
            log::warn!(
                "Embeddings.accumulate_token_grads: token_ids.len()={} 与 grads.nrows()={} 不一致，将按较短长度截断",
                token_ids.len(),
                grads.nrows()
            );
        }

        let seq_len = token_ids.len().min(grads.nrows());
        for (i, &token_id) in token_ids.iter().take(seq_len).enumerate() {
            let safe_id = if token_id >= self.token_embeddings.nrows() {
                self.token_embeddings.nrows().saturating_sub(1)
            } else {
                token_id
            };

            let grad_row = grads.row(i);
            let mut acc_row = self.token_grads_accum.row_mut(safe_id);
            acc_row += &grad_row;
        }
    }

    /// 用于梯度累积：旧接口（**已废弃**）。
    ///
    /// 历史原因：
    /// - 旧版 accumulate 依赖 `self.cached_input` 从层内部取回 token_id；
    /// - 这会让“正确性”绑定到 `cached_*`，从而阻碍我们删除缓存字段，也会在 batch 场景埋雷。
    ///
    /// 本轮重构已经删除了 `cached_input` 字段，因此该接口将 fail-fast，避免静默错误。
    /// 新代码请使用：`backward_accumulate_with_ctx(ctx, grads)`。
    #[deprecated(note = "已迁移到 ctx：请改用 backward_accumulate_with_ctx(ctx, grads)")]
    pub fn backward_accumulate(&mut self, _grads: &Array2<f32>) -> Array2<f32> {
        panic!("Embeddings.backward_accumulate 已废弃：请改用 backward_accumulate_with_ctx(ctx, grads)")
    }

    /// 用于梯度累积：只累加梯度，不更新参数（ctx 驱动，不依赖 cached_*）。
    ///
    /// 教学说明：
    /// - 这就是“第二轮重构”的关键：让 accumulate 接口也显式接收 ctx；
    /// - 这样训练循环就可以逐步移除层内 `cached_*` 字段，而不会破坏梯度累积语义。
    pub fn backward_accumulate_with_ctx(
        &mut self,
        ctx: &crate::llm::LayerContext,
        grads: &Array2<f32>,
    ) -> Array2<f32> {
        let Some(ctx) = ctx.downcast_ref::<EmbeddingsContext>() else {
            log::warn!("Embeddings.backward_accumulate_with_ctx 收到未知 ctx，跳过累积");
            return grads.clone();
        };

        self.accumulate_token_grads(&ctx.token_ids, grads);
        grads.to_owned()
    }

    /// 对累积的梯度执行一次参数更新，并清空累积 buffer。
    pub fn step_accumulated(&mut self, lr: f32, scale: f32) {
        let grad_scaled = &self.token_grads_accum * scale;
        self.token_optimizer
            .step(&mut self.token_embeddings, &grad_scaled, lr);
        self.zero_grad_accum();
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

    /// 计算带位移的嵌入表示，用于增量推理场景
    ///
    /// # 参数
    /// - `token_ids`: 待嵌入的 token 序列
    /// - `start_position`: 序列起始绝对位置（用于正确添加位置编码）
    pub fn embed_tokens_with_offset(
        &mut self,
        token_ids: &[usize],
        start_position: usize,
    ) -> Array2<f32> {
        if token_ids.is_empty() {
            return Array2::zeros((0, EMBEDDING_DIM));
        }

        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
        let seq_len = token_ids.len();
        let max_position = self.position_encoder.encoding.nrows();

        {
            use ndarray::s;
            let mut cache_slice = self.position_cache.slice_mut(s![0..seq_len, ..]);
            for (idx, mut row) in cache_slice.outer_iter_mut().enumerate() {
                let absolute_pos = start_position + idx;
                let clamped_pos = if absolute_pos < max_position {
                    absolute_pos
                } else {
                    log::warn!(
                        "位置编码越界: absolute_pos={} (max={}), 自动使用最后一行位置编码",
                        absolute_pos,
                        max_position
                    );
                    max_position.saturating_sub(1)
                };
                row.assign(&self.position_encoder.encoding.row(clamped_pos));
            }
        }

        let position_slice = self
            .position_cache
            .slice(ndarray::s![0..seq_len, ..])
            .to_owned();

        token_embeds + position_slice
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
    ///
    /// # 关于“显式上下文（ctx）”
    /// 在旧实现中，Embedding 层曾依赖 `self.cached_input` 在 backward 时取回 token_id。
    /// 这种“缓存写在 self 里”的设计在 batch 场景会踩坑：同一个层实例连续 forward 多个样本，
    /// 缓存会被覆盖，导致 backward 用错样本的 token_id。
    ///
    /// 新版 `Layer` trait 要求 `forward()` 返回 `(output, ctx)`：
    /// - `output`: 当前样本的前向输出
    /// - `ctx`: 当前样本 backward 所需的中间量（这里就是 token_id 序列）
    ///
    /// 这样 batch 时每个样本都有自己的 ctx，不会互相覆盖。
    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, crate::llm::LayerContext) {
        // 将浮点数转换为整数 token ID
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();

        // 查询嵌入 + 添加位置编码
        let out = self.embed_tokens(&token_ids);

        (out, Box::new(EmbeddingsContext { token_ids }))
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
    ///
    /// # ctx 的使用
    /// Embedding 的 backward 只需要 token_id 来把梯度累积到对应的词向量行。
    /// 因此我们从 `ctx` 里取回 token_ids，而不是依赖层内 `cached_*` 字段。
    fn backward(
        &mut self,
        ctx: &crate::llm::LayerContext,
        grads: &Array2<f32>,
        lr: f32,
    ) -> Array2<f32> {
        // 注意：ctx 是 `Box<dyn Any>`，需要 downcast 到本层定义的 EmbeddingsContext。
        // 如果 downcast 失败，说明调用方传错 ctx（通常是 bug），这里选择保守处理：不更新参数，直接传递梯度。
        let Some(ctx) = ctx.downcast_ref::<EmbeddingsContext>() else {
            log::warn!("Embeddings.backward 收到未知 ctx，跳过参数更新");
            return grads.clone();
        };

        let token_ids = &ctx.token_ids;
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
    ) -> (Array3<f32>, Vec<crate::llm::LayerContext>) {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let token_dim = input.shape()[2];

        // 兼容两种输入格式：
        // 1) (batch, seq, 1)：最后一维是 token_id 的标量（推荐）
        // 2) (batch, seq, d)：历史遗留/误用情况下也可能出现，我们仅使用第 0 维作为 token_id
        if token_dim != 1 {
            log::warn!(
                "Embeddings.forward_batch 收到非标量 token 维度：shape={:?}，将仅使用最后一维索引 0 的值作为 token_id（兼容模式）",
                input.shape()
            );
        }

        let token_ids_mat: Array2<usize> = Array2::from_shape_fn((batch_size, seq_len), |(b, s)| {
            input[[b, s, 0.min(token_dim.saturating_sub(1))]] as usize
        });

        // 为每个批次样本生成嵌入
        let mut output = Array3::zeros((batch_size, seq_len, EMBEDDING_DIM));
        let mut ctxs: Vec<crate::llm::LayerContext> = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let token_ids: Vec<usize> = token_ids_mat.row(b).to_vec();
            let embeddings = self.embed_tokens(&token_ids);
            output.slice_mut(ndarray::s![b, .., ..]).assign(&embeddings);
            ctxs.push(Box::new(EmbeddingsContext { token_ids }));
        }

        (output, ctxs)
    }

    /// **批量反向传播：更新词嵌入矩阵**
    fn backward_batch(
        &mut self,
        ctxs: &[crate::llm::LayerContext],
        grads: &Array3<f32>,
        lr: f32,
        attention_mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        if ctxs.len() != grads.shape()[0] {
            log::warn!(
                "Embeddings.backward_batch ctxs.len() 与 batch_size 不一致：ctxs={}, batch={}",
                ctxs.len(),
                grads.shape()[0]
            );
            return grads.to_owned();
        }

        let batch_size = grads.shape()[0];
        let seq_len = grads.shape()[1];

        // 累积 token 梯度（dense 版本，教学清晰优先）
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());

        for b in 0..batch_size {
            let Some(ctx) = ctxs[b].downcast_ref::<EmbeddingsContext>() else {
                log::warn!("Embeddings.backward_batch 收到未知 ctx，跳过该样本更新");
                continue;
            };
            for s in 0..seq_len {
                if let Some(mask) = attention_mask {
                    if mask[[b, s]] < 0.5 {
                        continue; // PAD 位置不更新嵌入
                    }
                }

                let token_id = ctx.token_ids.get(s).copied().unwrap_or(0);
                let safe_id = if token_id >= self.token_embeddings.nrows() {
                    self.token_embeddings.nrows().saturating_sub(1)
                } else {
                    token_id
                };

                let grad_vec = grads.slice(ndarray::s![b, s, ..]);
                {
                    let mut acc_row = token_grads.row_mut(safe_id);
                    acc_row += &grad_vec;
                }
            }
        }

        self.token_optimizer
            .step(&mut self.token_embeddings, &token_grads, lr);

        // 位置编码不需要梯度，直接把输入梯度传回上一层。
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
