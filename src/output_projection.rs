//! # 输出投影层（Output Projection Layer）
//!
//! 这是语言模型的最后一层，将隐藏状态投影到词汇表空间，预测下一个词。
//!
//! ## 作用
//!
//! 将 Transformer 的输出（512维向量）转换为词汇表大小的 logits（概率分数）：
//!
//! ```text
//! 输入: (seq_len, EMBEDDING_DIM) - Transformer 的隐藏状态
//! 输出: (seq_len, vocab_size) - 每个词的未归一化概率
//! ```
//!
//! ## 与 Softmax 的关系
//!
//! ```text
//! 完整的预测流程:
//! 1. 输出投影: hidden → logits (未归一化分数)
//! 2. Softmax: logits → probs (概率分布，总和为1)
//! 3. 采样/解码: probs → token_id (选择下一个词)
//! ```
//!
//! ## 参数规模
//!
//! 这是模型中参数最多的层之一：
//! - **权重**: EMBEDDING_DIM × vocab_size
//! - **偏置**: vocab_size ≈ 10,000 参数
//! - **总计**: 约 512 万参数
//!
//! ## 权重共享（Weight Tying）
//!
//! 在许多大型语言模型中，输出投影层的权重与词嵌入层共享：
//! - **优势**: 减少参数量，提高训练效率
//! - **本项目**: 未实现权重共享（教育目的，保持独立性）

use ndarray::{Array1, Array2, Axis};

use crate::{adam::Adam, llm::Layer, utils::sample_normal};

/// **输出投影层结构体**
pub struct OutputProjection {
    /// **权重矩阵** W: (embedding_dim, vocab_size) = (512, ~10000)
    /// 将隐藏状态映射到词汇表空间
    pub w_out: Array2<f32>,

    /// **偏置向量** b: (1, vocab_size)
    /// 为每个词添加偏置项
    pub b_out: Array2<f32>,

    /// **Adam 优化器**: 用于更新权重
    pub optimizer: Adam,

    /// **缓存输入**: 用于反向传播计算梯度
    pub cached_input: Option<Array2<f32>>,

    // =====================================================================
    // 梯度累积支持（Gradient Accumulation）
    // =====================================================================
    //
    // 背景：
    // - 本项目的训练代码支持 `accumulation_steps`（梯度累积），用于在显存/内存有限时
    //   “用多个 micro-batch 模拟一个大 batch”。
    // - 早期实现曾经尝试“只累积 logits 梯度，最后统一 backward 一次”，但由于各层的
    //   backward 依赖 forward 缓存（cached_input 等），会导致缓存被覆盖，从而使梯度
    //   与激活不匹配（数学错误）。
    //
    // 解决方案：
    // - 每个 micro-batch 都立刻执行一次完整 backward（保证缓存正确）；
    // - 但不立刻更新参数，而是把参数梯度累加到下面的 buffer 中；
    // - 当累积步数到达阈值时，再把累加梯度做平均（scale=1/steps）并执行一次 Adam 更新。
    //
    // 说明：
    // - 这里的 buffer **不参与序列化**（checkpoint 不保存它们），因为它们只在一次
    //   训练 step 内有意义；恢复训练时从 0 开始累积是正确且预期的。
    pub grad_w_out_accum: Array2<f32>,
    pub grad_b_out_accum: Array2<f32>,
}

impl OutputProjection {
    /// **创建新的输出投影层**
    ///
    /// # 参数
    /// - `embedding_dim`: 输入维度（512）
    /// - `vocab_size`: 词汇表大小（动态，通常5000-15000）
    ///
    /// # 初始化策略
    /// - **权重**: He 初始化 std = sqrt(2 / embedding_dim)
    /// - **偏置**: 全零初始化
    ///
    /// # 参数规模示例
    /// ```text
    /// vocab_size = 10,000:
    ///   权重: 512 × 10,000 = 5,120,000 参数
    ///   偏置: 10,000 参数
    ///   总计: 5,130,000 参数 (约占整个模型的一半！)
    /// ```
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        // He 初始化：std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();

        let w_out = Array2::from_shape_fn((embedding_dim, vocab_size), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        OutputProjection {
            w_out,
            b_out: Array2::zeros((1, vocab_size)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
            grad_w_out_accum: Array2::zeros((embedding_dim, vocab_size)),
            grad_b_out_accum: Array2::zeros((1, vocab_size)),
        }
    }

    /// 清空梯度累积 buffer。
    ///
    /// 教学要点：
    /// - 梯度累积的本质是“把多个 micro-batch 的梯度求和/平均后再更新一次参数”。因此每个
    ///   累积周期开始前必须把累积 buffer 清零。
    pub fn zero_grad_accum(&mut self) {
        self.grad_w_out_accum.fill(0.0);
        self.grad_b_out_accum.fill(0.0);
    }

    /// 仅做“梯度计算 + 累加”，不更新参数（用于梯度累积）。
    ///
    /// 返回值：传递给前一层的梯度（dL/dInput）。
    pub fn backward_accumulate(&mut self, grads: &Array2<f32>) -> Array2<f32> {
        let Some(input) = self.cached_input.as_ref() else {
            log::warn!("OutputProjection.backward_accumulate 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        let (grad_input, grad_w_out, grad_b_out) = Self::compute_grads(input, &self.w_out, grads);

        self.grad_w_out_accum += &grad_w_out;
        self.grad_b_out_accum += &grad_b_out;

        grad_input
    }

    /// 对累积的梯度执行一次参数更新，并清空累积 buffer。
    ///
    /// # 参数
    /// - `lr`: 学习率
    /// - `scale`: 梯度缩放系数（通常为 `1.0 / accumulation_steps`，表示取平均梯度）
    pub fn step_accumulated(&mut self, lr: f32, scale: f32) {
        let grad_w_scaled = &self.grad_w_out_accum * scale;
        let grad_b_scaled = &self.grad_b_out_accum * scale;

        self.optimizer.step(&mut self.w_out, &grad_w_scaled, lr);

        // bias 采用“纯 SGD 更新”（不走 Adam），以保持与历史 checkpoint 格式兼容。
        // 同时我们在这里使用 **sum**（而不是 mean）来聚合 bias 梯度，避免多除一次 seq_len。
        self.b_out -= &(lr * grad_b_scaled);

        self.zero_grad_accum();
    }

    fn compute_grads(
        input: &Array2<f32>,
        w_out: &Array2<f32>,
        grads: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // 计算权重梯度: grad_W = input^T · grads
        let grad_w_out = input.t().dot(grads);

        // 计算偏置梯度：对 token 位置维度求和（sum），而不是 mean。
        //
        // 原因（非常关键）：
        // - 本项目的 `compute_gradients_step()` 已经对 token 数做了平均（除以 target.len()）；
        // - 如果此处再对 token 维度取 mean，相当于又额外除一次 seq_len，导致 bias 学习过慢。
        let grad_b_vec: Array1<f32> = grads.sum_axis(Axis(0));
        let grad_b_out = grad_b_vec.insert_axis(Axis(0));

        // 计算输入梯度: grad_input = grads · W^T
        let grad_input = grads.dot(&w_out.t());

        (grad_input, grad_w_out, grad_b_out)
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// **前向传播：将隐藏状态投影到词汇表空间**
    ///
    /// # 计算公式
    /// ```text
    /// logits = input · W + b
    /// ```
    ///
    /// # 参数
    /// - `input`: (seq_len, 512) 隐藏状态
    ///
    /// # 返回值
    /// - `logits`: (seq_len, vocab_size) 未归一化的分数
    ///
    /// # 示例
    /// ```text
    /// 输入: (4, 512) - 4个token的隐藏状态
    /// 输出: (4, 10000) - 4个token，每个对10000个词的预测分数
    ///
    /// 输出[0]的含义：
    ///   [3.2, -1.5, 0.8, ...]  // 10000个分数
    ///   ↑     ↑     ↑
    ///   词0   词1   词2
    ///   高分  低分  中等
    ///
    /// 经过 softmax 后变为概率：
    ///   [0.45, 0.01, 0.15, ...]  // 总和为1
    /// ```
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) + &self.b_out
    }

    /// **反向传播：计算梯度并更新参数**
    ///
    /// # 梯度计算
    /// ```text
    /// 前向: logits = input · W + b
    ///
    /// 反向:
    ///   grad_W = input^T · grads
    ///   grad_b = mean(grads, axis=0)
    ///   grad_input = grads · W^T
    /// ```
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let Some(input) = self.cached_input.as_ref() else {
            log::warn!("OutputProjection.backward 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        let (grad_input, grad_w_out, grad_b_out) = Self::compute_grads(input, &self.w_out, grads);

        // 更新参数
        self.optimizer.step(&mut self.w_out, &grad_w_out, lr);
        self.b_out -= &(lr * &grad_b_out);

        grad_input
    }

    /// **参数总数**
    ///
    /// 返回: embedding_dim × vocab_size + vocab_size
    ///
    /// 例如: 512 × 10000 + 10000 = 5,130,000 参数
    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }

    fn set_training_mode(&mut self, _training: bool) {}
}
