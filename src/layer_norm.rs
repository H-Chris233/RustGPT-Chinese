//! # 层归一化（Layer Normalization）
//!
//! 层归一化是稳定神经网络训练的关键技术，在 Transformer 中广泛使用。
//!
//! ## 为什么需要归一化？
//!
//! **问题**：深度网络训练时，激活值的分布会逐层发生变化（Internal Covariate Shift）：
//! - 某些神经元的值过大 → 梯度爆炸
//! - 某些神经元的值过小 → 梯度消失
//! - 学习率难以调整（对不同层需要不同的学习率）
//!
//! **解决方案**：归一化将每层的激活值标准化到均值0、方差1。
//!
//! ## Layer Norm vs Batch Norm
//!
//! | 特性 | Batch Norm | Layer Norm |
//! |------|-----------|-----------|
//! | **归一化维度** | 跨样本（batch 维度） | 跨特征（feature 维度） |
//! | **依赖性** | 依赖 batch size | 不依赖 batch size |
//! | **适用场景** | CNN（图像） | RNN/Transformer（序列） |
//! | **推理时** | 需要统计量 | 不需要额外统计 |
//!
//! ## 数学公式
//!
//! ```text
//! 1. 计算均值和标准差（对特征维度）:
//!    μ = mean(x)
//!    σ = std(x)
//!
//! 2. 标准化:
//!    x_norm = (x - μ) / (σ + ε)
//!
//! 3. 缩放和偏移（可学习参数）:
//!    y = γ * x_norm + β
//! ```
//!
//! 其中：
//! - `ε (epsilon)`: 防止除零的小常数（1e-8）
//! - `γ (gamma)`: 可学习的缩放参数（初始化为1）
//! - `β (beta)`: 可学习的偏移参数（初始化为0）
//!
//! ## 为什么需要 γ 和 β？
//!
//! 标准化会破坏网络学到的特征分布。γ 和 β 让网络可以：
//! - **恢复原始分布**：如果 γ=σ, β=μ，可以完全还原
//! - **学习最优分布**：网络自己决定需要什么样的分布
//!
//! ## 示例
//!
//! ```text
//! 输入: x = [[1.0, 2.0, 3.0],
//!            [4.0, 5.0, 6.0]]
//!
//! 步骤 1 - 计算统计量（每行独立）:
//!   行1: μ=2.0, σ=0.816
//!   行2: μ=5.0, σ=0.816
//!
//! 步骤 2 - 标准化:
//!   行1: [(-1.224), 0, 1.224]
//!   行2: [(-1.224), 0, 1.224]
//!
//! 步骤 3 - 应用 γ 和 β (假设 γ=1, β=0):
//!   输出与标准化相同
//! ```

use ndarray::{Array2, Axis};

use crate::{EPSILON, adam::Adam, llm::Layer};

/// **层归一化结构体**
pub struct LayerNorm {
    /// **数值稳定性常数**: 防止除零错误（1e-8）
    pub epsilon: f32,

    /// **缩放参数 γ**: (1, embedding_dim)，可学习，初始化为1
    /// 控制标准化后特征的尺度
    pub gamma: Array2<f32>,

    /// **偏移参数 β**: (1, embedding_dim)，可学习，初始化为0
    /// 控制标准化后特征的中心位置
    pub beta: Array2<f32>,

    // ========== 前向传播缓存（用于反向传播） ==========
    /// **缓存输入**: 原始输入值
    pub cached_input: Option<Array2<f32>>,

    /// **缓存均值**: 每个样本在特征维度上的均值
    pub cached_mean: Option<Array2<f32>>,

    /// **缓存 inv_std**: 每个样本在特征维度上的 `1 / sqrt(var + ε)`
    ///
    /// 说明：
    /// - 早期版本缓存的是 `std`，并采用 `(std + ε)` 的非标准归一化；
    /// - 为保证前向/反向数学一致性（标准 LayerNorm），这里缓存 `inv_std`。
    pub cached_inv_std: Option<Array2<f32>>,

    // ========== Adam 优化器 ==========
    pub optimizer_gamma: Adam,
    pub optimizer_beta: Adam,

    // =====================================================================
    // 梯度累积支持（Gradient Accumulation）
    // =====================================================================
    //
    // 说明：
    // - LayerNorm 的可训练参数只有 γ/β，两者的梯度形状都是 (1, embedding_dim)。
    // - 为了支持“正确的梯度累积”，我们在每个 micro-batch 的 backward 时计算并累加梯度，
    //   而不是立刻更新参数；累积周期结束后再统一做一次 Adam step。
    pub grad_gamma_accum: Array2<f32>,
    pub grad_beta_accum: Array2<f32>,
}

impl LayerNorm {
    /// **创建新的层归一化层**
    ///
    /// # 参数
    /// - `embedding_dim`: 特征维度（与 EMBEDDING_DIM 一致）
    ///
    /// # 初始化策略
    /// - **γ (gamma)**: 初始化为全1（保持原始尺度）
    /// - **β (beta)**: 初始化为全0（保持原始中心）
    /// - **ε (epsilon)**: 使用全局常量 EPSILON (1e-8)
    ///
    /// 这样初始化确保层归一化在训练初期不改变数据分布。
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: EPSILON,                        // 使用统一的 EPSILON 常量
            gamma: Array2::ones((1, embedding_dim)), // γ 初始化为 1
            beta: Array2::zeros((1, embedding_dim)), // β 初始化为 0
            cached_input: None,
            cached_mean: None,
            cached_inv_std: None,
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta: Adam::new((1, embedding_dim)),
            grad_gamma_accum: Array2::zeros((1, embedding_dim)),
            grad_beta_accum: Array2::zeros((1, embedding_dim)),
        }
    }

    /// 清空梯度累积 buffer。
    pub fn zero_grad_accum(&mut self) {
        self.grad_gamma_accum.fill(0.0);
        self.grad_beta_accum.fill(0.0);
    }

    /// 仅计算并累积参数梯度，不更新参数（用于梯度累积）。
    pub fn backward_accumulate(&mut self, grads: &Array2<f32>) -> Array2<f32> {
        let (Some(input), Some(mean), Some(inv_std)) = (
            self.cached_input.as_ref(),
            self.cached_mean.as_ref(),
            self.cached_inv_std.as_ref(),
        ) else {
            log::warn!("LayerNorm.backward_accumulate 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        let (grad_input, grad_gamma, grad_beta) =
            Self::compute_grads(input, mean, inv_std, grads, &self.gamma, self.epsilon);

        self.grad_gamma_accum += &grad_gamma;
        self.grad_beta_accum += &grad_beta;

        grad_input
    }

    /// 对累积的梯度执行一次参数更新，并清空累积 buffer。
    pub fn step_accumulated(&mut self, lr: f32, scale: f32) {
        let grad_gamma_scaled = &self.grad_gamma_accum * scale;
        let grad_beta_scaled = &self.grad_beta_accum * scale;

        self.optimizer_gamma
            .step(&mut self.gamma, &grad_gamma_scaled, lr);
        self.optimizer_beta
            .step(&mut self.beta, &grad_beta_scaled, lr);

        self.zero_grad_accum();
    }

    fn compute_grads(
        input: &Array2<f32>,
        mean: &Array2<f32>,
        inv_std: &Array2<f32>,
        grads: &Array2<f32>,
        gamma: &Array2<f32>,
        _epsilon: f32,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // 标准 LayerNorm 定义：
        //   var = mean((x - mean)^2)
        //   inv_std = 1 / sqrt(var + ε)
        //   x_hat = (x - mean) * inv_std
        //   y = gamma * x_hat + beta
        //
        // 反向传播（对每一行独立计算）：
        //   dy = grads
        //   dgamma = sum(dy * x_hat)
        //   dbeta  = sum(dy)
        //   dx = inv_std * (dy*gamma - mean(dy*gamma) - x_hat*mean((dy*gamma)*x_hat))
        //
        // 参考：LayerNorm 的常见推导形式（等价于 “batchnorm-style” 的简化向量化公式）。
        let x_hat = (input - mean) * inv_std;
        let n_features = input.shape()[1] as f32;

        let grad_gamma = (&x_hat * grads).sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_beta = grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let dy = gamma * grads;
        let mean_dy = dy.sum_axis(Axis(1)).insert_axis(Axis(1)) / n_features;
        let mean_dy_xhat = (&dy * &x_hat).sum_axis(Axis(1)).insert_axis(Axis(1)) / n_features;
        let grad_input = (dy - &mean_dy - &x_hat * &mean_dy_xhat) * inv_std;

        (grad_input, grad_gamma, grad_beta)
    }

    /// **执行层归一化**
    ///
    /// # 算法步骤
    ///
    /// 1. **计算统计量**（对每个样本的特征维度）：
    ///    ```text
    ///    μ = mean(x, axis=1)  // 每行的均值
    ///    var = mean((x-μ)^2)  // 每行的方差（无偏差修正）
    ///    ```
    ///
    /// 2. **标准化**：
    ///    ```text
    ///    x_hat = (x - μ) / sqrt(var + ε)
    ///    ```
    ///
    /// 3. **缩放和偏移**：
    ///    ```text
    ///    output = γ * x_hat + β
    ///    ```
    ///
    /// # 参数
    /// - `input`: (seq_len, embedding_dim) 输入张量
    ///
    /// # 返回值
    /// (seq_len, embedding_dim) 归一化后的张量
    ///
    /// # 示例
    /// ```text
    /// 输入: [[1.0, 2.0, 3.0],    // 均值=2.0, 标准差≈0.816
    ///       [10.0, 20.0, 30.0]]  // 均值=20.0, 标准差≈8.165
    ///
    /// 标准化后（假设 γ=1, β=0）:
    /// [[-1.224, 0.0, 1.224],     // 均值=0, 标准差=1
    ///  [-1.224, 0.0, 1.224]]     // 均值=0, 标准差=1
    /// ```
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 步骤 1: 计算每个样本的均值和方差
        let Some(mean) = input.mean_axis(Axis(1)).map(|m| m.insert_axis(Axis(1))) else {
            log::warn!("LayerNorm.normalize 收到空输入，直接返回");
            return input.clone();
        }; // (seq_len, 1)
        let centered = input - &mean;
        let Some(var) = centered
            .mapv(|x| x * x)
            .mean_axis(Axis(1))
            .map(|v| v.insert_axis(Axis(1)))
        else {
            log::warn!("LayerNorm.normalize 收到空输入（无法计算方差），直接返回");
            return input.clone();
        }; // (seq_len, 1)
        let inv_std = (var + self.epsilon).mapv(|x| 1.0 / x.sqrt()); // (seq_len, 1)

        // 缓存值用于反向传播
        self.cached_input = Some(input.clone());
        self.cached_mean = Some(mean.clone());
        self.cached_inv_std = Some(inv_std.clone());

        // 步骤 2: 标准化（均值0，方差1）
        let normalized = centered * &inv_std;

        // 步骤 3: 缩放和偏移
        &self.gamma * &normalized + &self.beta
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (Some(input), Some(mean), Some(inv_std)) = (
            self.cached_input.as_ref(),
            self.cached_mean.as_ref(),
            self.cached_inv_std.as_ref(),
        ) else {
            log::warn!("LayerNorm.backward 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        let (grad_input, grad_gamma, grad_beta) =
            Self::compute_grads(input, mean, inv_std, grads, &self.gamma, self.epsilon);

        self.optimizer_gamma.step(&mut self.gamma, &grad_gamma, lr);
        self.optimizer_beta.step(&mut self.beta, &grad_beta, lr);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn set_training_mode(&mut self, _training: bool) {}
}
