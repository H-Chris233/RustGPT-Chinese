//! # 输出投影层（Output Projection Layer）
//!
//! 这是语言模型的最后一层，将隐藏状态投影到词汇表空间，预测下一个词。
//!
//! ## 作用
//!
//! 将 Transformer 的输出（512维向量）转换为词汇表大小的 logits（概率分数）：
//!
//! ```text
//! 输入: (seq_len, 512) - Transformer 的隐藏状态
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
//! - **权重**: 512 × vocab_size ≈ 512 × 10,000 = 5,120,000 参数
//! - **偏置**: vocab_size ≈ 10,000 参数
//! - **总计**: 约 512 万参数
//!
//! ## 权重共享（Weight Tying）
//!
//! 在许多大型语言模型中，输出投影层的权重与词嵌入层共享：
//! - **优势**: 减少参数量，提高训练效率
//! - **本项目**: 未实现权重共享（教育目的，保持独立性）

use ndarray::{Array2, Axis};

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
        }
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

        // 计算权重梯度: grad_W = input^T · grads
        let grad_w_out = input.t().dot(grads);

        // 计算偏置梯度: grad_b = mean(grads)
        let grad_b_out = grads
            .mean_axis(Axis(0))
            .unwrap_or_else(|| ndarray::Array1::zeros(grads.shape()[1]));

        // 计算输入梯度: grad_input = grads · W^T
        let grad_input = grads.dot(&self.w_out.t());

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
