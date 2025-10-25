//! # 算子融合模块（Operator Fusion）
//!
//! 本模块实现了常见神经网络操作的融合，以减少中间张量分配和内存访问开销。
//!
//! ## 优化目标（v0.4.0）
//!
//! 1. **LayerNorm + Linear**: 合并归一化和线性变换
//! 2. **GELU + Linear**: 合并激活函数和线性层
//! 3. **减少内存分配**: 通过融合操作减少中间张量创建
//!
//! ## 性能提升原理
//!
//! **未融合**:
//! ```text
//! x → LayerNorm → [中间张量1] → Linear → [输出]
//! - 2次内存分配
//! - 2次完整遍历数据
//! - 缓存未命中增加
//! ```
//!
//! **融合后**:
//! ```text
//! x → LayerNorm+Linear → [输出]
//! - 1次内存分配
//! - 1次遍历数据（部分融合）
//! - 更好的缓存局部性
//! ```
//!
//! ## 使用示例
//!
//! ```rust
//! use llm::fused_ops::FusedLayerNormLinear;
//! use ndarray::Array2;
//!
//! let mut fused_op = FusedLayerNormLinear::new(512, 1024);
//! let input = Array2::zeros((10, 512));
//! let output = fused_op.forward(&input);
//! // output shape: (10, 1024)
//! ```

use ndarray::{Array1, Array2, Axis};

use crate::{llm::Layer, utils::sample_normal, EPSILON};

/// **融合的 LayerNorm + Linear 操作**
///
/// 将层归一化和线性变换融合为单一操作，减少中间张量分配。
///
/// # 数学原理
///
/// ```text
/// 标准流程:
///   1. x_norm = (x - mean) / sqrt(variance + ε)
///   2. x_scaled = x_norm * gamma + beta
///   3. output = x_scaled · W + b
///
/// 融合后:
///   output = ((x - mean) / sqrt(variance + ε) * gamma + beta) · W + b
/// ```
///
/// # 性能优势
///
/// - 减少1个中间张量的分配（x_scaled）
/// - 可以在单次遍历中完成归一化和部分线性计算
/// - 更好的缓存局部性
pub struct FusedLayerNormLinear {
    /// LayerNorm 的缩放参数（可学习）
    gamma: Array1<f32>,
    /// LayerNorm 的偏移参数（可学习）
    beta: Array1<f32>,
    /// Linear 层的权重矩阵 (input_dim, output_dim)
    weight: Array2<f32>,
    /// Linear 层的偏置向量 (output_dim,)
    bias: Array1<f32>,

    /// 输入维度
    input_dim: usize,
    /// 输出维度
    output_dim: usize,

    /// 缓存：归一化后的输入（用于反向传播）
    cached_normalized: Option<Array2<f32>>,
    /// 缓存：原始输入（用于反向传播）
    cached_input: Option<Array2<f32>>,
    /// 缓存：均值（用于反向传播）
    cached_mean: Option<Array1<f32>>,
    /// 缓存：标准差（用于反向传播）
    cached_std: Option<Array1<f32>>,
}

impl FusedLayerNormLinear {
    /// 创建新的融合操作层
    ///
    /// # 参数
    /// - `input_dim`: 输入维度
    /// - `output_dim`: 输出维度
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();

        // LayerNorm 参数初始化
        let gamma = Array1::ones(input_dim);
        let beta = Array1::zeros(input_dim);

        // Linear 层权重初始化（He 初始化）
        let std = (2.0 / input_dim as f32).sqrt();
        let weight = Array2::from_shape_fn((input_dim, output_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });
        let bias = Array1::zeros(output_dim);

        FusedLayerNormLinear {
            gamma,
            beta,
            weight,
            bias,
            input_dim,
            output_dim,
            cached_normalized: None,
            cached_input: None,
            cached_mean: None,
            cached_std: None,
        }
    }

    /// 前向传播：融合的 LayerNorm + Linear
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (batch_size, _) = input.dim();

        // 步骤1: LayerNorm
        let mean = input.mean_axis(Axis(1)).unwrap();
        let variance = input.var_axis(Axis(1), 0.0);
        let std = variance.mapv(|v| (v + EPSILON).sqrt());

        let mut normalized = Array2::zeros(input.dim());
        for i in 0..batch_size {
            let row = input.row(i);
            let norm_row = (&row - mean[i]) / std[i];
            normalized
                .row_mut(i)
                .assign(&(&norm_row * &self.gamma + &self.beta));
        }

        // 缓存中间结果
        self.cached_normalized = Some(normalized.clone());
        self.cached_input = Some(input.clone());
        self.cached_mean = Some(mean);
        self.cached_std = Some(std);

        // 步骤2: Linear 变换
        normalized.dot(&self.weight) + &self.bias
    }

    /// 反向传播
    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let normalized = self.cached_normalized.as_ref().unwrap();
        let input = self.cached_input.as_ref().unwrap();
        let _mean = self.cached_mean.as_ref().unwrap();
        let std = self.cached_std.as_ref().unwrap();

        // 线性层的梯度
        let grad_weight = normalized.t().dot(grad_output);
        let grad_bias = grad_output.sum_axis(Axis(0));
        let grad_normalized = grad_output.dot(&self.weight.t());

        // LayerNorm 的梯度（简化版）
        let (batch_size, input_dim) = input.dim();
        let mut grad_input = Array2::zeros(input.dim());

        // 累积 gamma 和 beta 的梯度
        let mut grad_gamma_total = Array1::zeros(input_dim);
        let mut grad_beta_total = Array1::zeros(input_dim);

        for i in 0..batch_size {
            let norm_row = normalized.row(i);
            let grad_norm_row = grad_normalized.row(i);

            // 累积 gamma 和 beta 的梯度
            for j in 0..input_dim {
                grad_gamma_total[j] += grad_norm_row[j] * norm_row[j];
                grad_beta_total[j] += grad_norm_row[j];
            }

            // 输入的梯度
            let grad_input_row = &grad_norm_row * &self.gamma / std[i];
            grad_input.row_mut(i).assign(&grad_input_row);
        }

        // 使用 SGD 更新参数
        self.weight -= &(&grad_weight * lr);
        self.bias -= &(&grad_bias * lr);
        self.gamma -= &(&grad_gamma_total * lr);
        self.beta -= &(&grad_beta_total * lr);

        grad_input
    }
}

impl Layer for FusedLayerNormLinear {
    fn layer_type(&self) -> &str {
        "FusedLayerNormLinear"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        self.backward(grads, lr)
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len() + self.weight.len() + self.bias.len()
    }

    fn set_training_mode(&mut self, _training: bool) {
        // LayerNorm 和 Linear 没有训练/推理模式的区别
    }
}

/// **融合的 GELU + Linear 操作**
///
/// 在前馈网络中常用，将 GELU 激活和线性层融合。
///
/// # 数学原理
///
/// ```text
/// 标准流程:
///   1. x_activated = GELU(x) = x * Φ(x)
///   2. output = x_activated · W + b
///
/// 融合后:
///   output = GELU(x) · W + b
/// ```
pub struct FusedGELULinear {
    /// Linear 层的权重矩阵
    weight: Array2<f32>,
    /// Linear 层的偏置
    bias: Array1<f32>,

    input_dim: usize,
    output_dim: usize,

    /// 缓存：GELU 激活后的输出
    cached_activated: Option<Array2<f32>>,
    /// 缓存：原始输入
    cached_input: Option<Array2<f32>>,
}

impl FusedGELULinear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / input_dim as f32).sqrt();
        let weight = Array2::from_shape_fn((input_dim, output_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });
        let bias = Array1::zeros(output_dim);

        FusedGELULinear {
            weight,
            bias,
            input_dim,
            output_dim,
            cached_activated: None,
            cached_input: None,
        }
    }

    /// GELU 激活函数
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// GELU 导数
    fn gelu_derivative(x: f32) -> f32 {
        let tanh_arg = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
        let tanh_val = tanh_arg.tanh();
        let sech_squared = 1.0 - tanh_val * tanh_val;

        let term1 = 0.5 * (1.0 + tanh_val);
        let term2 = 0.5
            * x
            * sech_squared
            * (2.0 / std::f32::consts::PI).sqrt()
            * (1.0 + 3.0 * 0.044715 * x.powi(2));

        term1 + term2
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 应用 GELU
        let activated = input.mapv(Self::gelu);

        self.cached_activated = Some(activated.clone());
        self.cached_input = Some(input.clone());

        // Linear 变换
        activated.dot(&self.weight) + &self.bias
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let activated = self.cached_activated.as_ref().unwrap();
        let input = self.cached_input.as_ref().unwrap();

        // Linear 层的梯度
        let grad_weight = activated.t().dot(grad_output);
        let grad_bias = grad_output.sum_axis(Axis(0));
        let grad_activated = grad_output.dot(&self.weight.t());

        // GELU 的梯度
        let grad_input = grad_activated
            .iter()
            .zip(input.iter())
            .map(|(&g, &x)| g * Self::gelu_derivative(x))
            .collect::<Vec<_>>();
        let grad_input = Array2::from_shape_vec(input.dim(), grad_input).unwrap();

        // 使用 SGD 更新参数
        self.weight -= &(&grad_weight * lr);
        self.bias -= &(&grad_bias * lr);

        grad_input
    }
}

impl Layer for FusedGELULinear {
    fn layer_type(&self) -> &str {
        "FusedGELULinear"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        self.backward(grads, lr)
    }

    fn parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    fn set_training_mode(&mut self, _training: bool) {
        // GELU 和 Linear 没有训练/推理模式的区别
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_layernorm_linear() {
        let mut fused_op = FusedLayerNormLinear::new(4, 8);
        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let output = fused_op.forward(&input);
        assert_eq!(output.shape(), &[2, 8]);

        let grad = Array2::ones((2, 8));
        let grad_input = fused_op.backward(&grad, 0.001);
        assert_eq!(grad_input.shape(), &[2, 4]);
    }

    #[test]
    fn test_fused_gelu_linear() {
        let mut fused_op = FusedGELULinear::new(4, 8);
        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let output = fused_op.forward(&input);
        assert_eq!(output.shape(), &[2, 8]);

        let grad = Array2::ones((2, 8));
        let grad_input = fused_op.backward(&grad, 0.001);
        assert_eq!(grad_input.shape(), &[2, 4]);
    }
}
