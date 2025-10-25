//! # 前馈神经网络（Feed-Forward Network, FFN）
//!
//! 这是 Transformer 块中的第二个核心组件，负责对每个位置的特征进行非线性变换。
//!
//! ## 网络结构
//!
//! FFN 是一个简单的两层全连接网络，包含：
//! 1. **扩展层**：512 → 1024（embedding_dim → hidden_dim）
//! 2. **ReLU 激活**：引入非线性
//! 3. **压缩层**：1024 → 512（hidden_dim → embedding_dim）
//!
//! ## 数学表示
//!
//! ```text
//! FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
//! ```
//!
//! 其中：
//! - `x`: 输入 (seq_len, 512)
//! - `W₁`: 第一层权重 (512, 1024)
//! - `b₁`: 第一层偏置 (1, 1024)
//! - `ReLU`: 激活函数，ReLU(x) = max(0, x)
//! - `W₂`: 第二层权重 (1024, 512)
//! - `b₂`: 第二层偏置 (1, 512)
//!
//! ## 为什么需要 FFN？
//!
//! 1. **增加模型容量**：自注意力是线性操作（加权求和），FFN 引入非线性
//! 2. **特征变换**：在更高维度空间（1024维）中进行特征提取
//! 3. **位置独立**：对每个位置独立处理，不同于注意力的全局依赖
//!
//! ## "瓶颈"设计的优势
//!
//! 512 → 1024 → 512 的结构类似自编码器：
//! - **扩展阶段**：学习丰富的中间表示
//! - **压缩阶段**：提取最重要的特征
//! - 这种设计有助于模型学习抽象的、压缩的特征表示

use ndarray::{Array2, Axis};

use crate::{adam::Adam, llm::Layer, utils::sample_normal};

/// **前馈神经网络结构体**
pub struct FeedForward {
    /// **第一层权重** W₁: (embedding_dim, hidden_dim) = (512, 1024)
    pub w1: Array2<f32>,

    /// **第一层偏置** b₁: (1, hidden_dim) = (1, 1024)
    pub b1: Array2<f32>,

    /// **第二层权重** W₂: (hidden_dim, embedding_dim) = (1024, 512)
    pub w2: Array2<f32>,

    /// **第二层偏置** b₂: (1, embedding_dim) = (1, 512)
    pub b2: Array2<f32>,

    // ========== 缓存变量（用于反向传播） ==========
    /// **缓存的输入** x: (seq_len, 512)
    pub input: Option<Array2<f32>>,

    /// **ReLU 激活前的隐藏层** (seq_len, 1024)
    /// 保存 W₁·x + b₁ 的结果，反向传播时需要计算 ReLU 梯度
    pub hidden_pre_activation: Option<Array2<f32>>,

    /// **ReLU 激活后的隐藏层** (seq_len, 1024)
    /// 保存 ReLU(W₁·x + b₁) 的结果，反向传播时需要计算 W₂ 的梯度
    pub hidden_post_activation: Option<Array2<f32>>,

    // ========== Adam 优化器（每个参数一个） ==========
    /// W₁ 的优化器
    pub optimizer_w1: Adam,

    /// b₁ 的优化器
    pub optimizer_b1: Adam,

    /// W₂ 的优化器
    pub optimizer_w2: Adam,

    /// b₂ 的优化器
    pub optimizer_b2: Adam,
}

impl FeedForward {
    /// **创建新的前馈神经网络**
    ///
    /// # 参数
    /// - `embedding_dim`: 输入/输出维度（512）
    /// - `hidden_dim`: 中间层维度（1024）
    ///
    /// # 权重初始化：He 初始化
    ///
    /// 使用 He 初始化（也称为 Kaiming 初始化），特别适合 ReLU 激活函数：
    ///
    /// ```text
    /// std_w1 = sqrt(2 / embedding_dim) = sqrt(2 / 512) ≈ 0.0625
    /// std_w2 = sqrt(2 / hidden_dim) = sqrt(2 / 1024) ≈ 0.0442
    /// ```
    ///
    /// **为什么用 He 初始化？**
    /// - **Xavier 初始化**：适合 tanh/sigmoid，但对 ReLU 来说太小
    /// - **He 初始化**：专为 ReLU 设计，系数为 sqrt(2/n) 而非 sqrt(1/n)
    /// - ReLU 会将一半的神经元置零，需要更大的初始权重来维持方差
    ///
    /// **偏置初始化**：全零初始化（标准做法）
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        // He 初始化 W₁
        let std_w1 = (2.0 / embedding_dim as f32).sqrt();

        // He 初始化 W₂
        let std_w2 = (2.0 / hidden_dim as f32).sqrt();

        let w1 = Array2::from_shape_fn((embedding_dim, hidden_dim), |_| {
            sample_normal(&mut rng, 0.0, std_w1)
        });

        let w2 = Array2::from_shape_fn((hidden_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std_w2)
        });

        FeedForward {
            w1,
            b1: Array2::zeros((1, hidden_dim)), // 偏置初始化为0
            w2,
            b2: Array2::zeros((1, embedding_dim)), // 偏置初始化为0
            input: None,
            hidden_pre_activation: None,
            hidden_post_activation: None,
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_b1: Adam::new((1, hidden_dim)),
            optimizer_w2: Adam::new((hidden_dim, embedding_dim)),
            optimizer_b2: Adam::new((1, embedding_dim)),
        }
    }
}

impl Layer for FeedForward {
    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// **反向传播：计算梯度并更新参数**
    ///
    /// # 反向传播推导
    ///
    /// 前向传播：
    /// ```text
    /// h = W₁·x + b₁                  // 线性变换
    /// h_activated = ReLU(h)          // 激活
    /// output = W₂·h_activated + b₂   // 输出层
    /// ```
    ///
    /// 反向传播（链式法则）：
    /// ```text
    /// grad_W₂ = h_activated^T · grad_output
    /// grad_b₂ = sum(grad_output, axis=0)
    /// grad_h_activated = grad_output · W₂^T
    /// grad_h = grad_h_activated * ReLU'(h)    // ReLU导数
    /// grad_W₁ = x^T · grad_h
    /// grad_b₁ = sum(grad_h, axis=0)
    /// grad_input = grad_h · W₁^T
    /// ```
    ///
    /// # ReLU 导数
    ///
    /// ReLU'(x) = 1 if x > 0 else 0
    ///
    /// 这意味着只有激活的神经元（值>0）会传播梯度，其他神经元梯度为0。
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 从缓存中获取前向传播的中间结果
        let (Some(input), Some(hidden_pre_activation), Some(hidden_post_activation)) = (
            self.input.as_ref(),
            self.hidden_pre_activation.as_ref(),
            self.hidden_post_activation.as_ref(),
        ) else {
            log::warn!("FeedForward.backward 在未执行 forward 的情况下被调用，跳过参数更新");
            return grads.clone();
        };

        // ========== 反向传播第二层（输出层） ==========
        // grad_W₂ = h_activated^T · grad_output
        let grad_w2 = hidden_post_activation.t().dot(grads);

        // grad_b₂ = sum(grad_output, axis=0)
        let grad_b2 = grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        // grad_h_activated = grad_output · W₂^T
        let grad_hidden_post_activation = grads.dot(&self.w2.t());

        // ========== 反向传播 ReLU 激活函数 ==========
        // ReLU 的导数：x > 0 ? 1 : 0
        let mut relu_grad = hidden_pre_activation.clone();
        relu_grad.map_inplace(|x| {
            *x = if *x > 0.0 { 1.0 } else { 0.0 };
        });

        // grad_h = grad_h_activated * ReLU'(h) (逐元素相乘)
        let grad_hidden_pre_activation = grad_hidden_post_activation * relu_grad;

        // ========== 反向传播第一层 ==========
        // grad_W₁ = input^T · grad_h
        let grad_w1 = input.t().dot(&grad_hidden_pre_activation);

        // grad_b₁ = sum(grad_h, axis=0)
        let grad_b1 = grad_hidden_pre_activation
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));

        // grad_input = grad_h · W₁^T（传递给上一层）
        let grad_input_feedforward = grad_hidden_pre_activation.dot(&self.w1.t());

        // ========== 使用 Adam 优化器更新所有参数 ==========
        self.optimizer_w2.step(&mut self.w2, &grad_w2, lr);
        self.optimizer_b2.step(&mut self.b2, &grad_b2, lr);
        self.optimizer_w1.step(&mut self.w1, &grad_w1, lr);
        self.optimizer_b1.step(&mut self.b1, &grad_b1, lr);

        grad_input_feedforward
    }

    /// **前向传播：两层全连接网络**
    ///
    /// # 计算步骤
    ///
    /// 1. **第一层线性变换**： h = x·W₁ + b₁
    ///    - 输入：(seq_len, 512)
    ///    - W₁：(512, 1024)
    ///    - 输出：(seq_len, 1024)
    ///
    /// 2. **ReLU 激活**： h_activated = max(0, h)
    ///    - 将负值置零，保留正值
    ///
    /// 3. **第二层线性变换**： output = h_activated·W₂ + b₂
    ///    - 输入：(seq_len, 1024)
    ///    - W₂：(1024, 512)
    ///    - 输出：(seq_len, 512)
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 第一层：线性变换
        let hidden_pre_activation = input.dot(&self.w1) + &self.b1;

        // ReLU 激活：max(0, x)
        let mut hidden_post_activation = hidden_pre_activation.clone();
        hidden_post_activation.map_inplace(|x| {
            *x = x.max(0.0);
        });

        // 第二层：线性变换到输出
        let output = hidden_post_activation.dot(&self.w2) + &self.b2;

        // 缓存所有中间结果，供反向传播使用
        self.input = Some(input.clone());
        self.hidden_pre_activation = Some(hidden_pre_activation);
        self.hidden_post_activation = Some(hidden_post_activation);

        output
    }

    /// **计算参数总数**
    ///
    /// 包括：
    /// - W₁: 512 × 1024 = 524,288
    /// - b₁: 1 × 1024 = 1,024
    /// - W₂: 1024 × 512 = 524,288
    /// - b₂: 1 × 512 = 512
    /// - **总计**: 约 105万 参数
    fn parameters(&self) -> usize {
        self.b1.len() + self.b2.len() + self.w1.len() + self.w2.len()
    }

    /// **设置训练模式**
    ///
    /// FFN 没有 Dropout，所以训练/推理模式没有区别。
    fn set_training_mode(&mut self, _training: bool) {}
}
