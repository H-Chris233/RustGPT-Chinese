//! # Transformer 块实现
//!
//! 这是 Transformer 架构的核心构建单元。每个 Transformer 块包含：
//! 1. 多头自注意力机制（Multi-Head Self-Attention）
//! 2. 前馈神经网络（Feed-Forward Network）
//! 3. 层归一化（Layer Normalization）
//! 4. 残差连接（Residual Connections）
//! 5. Dropout 正则化
//!
//! ## 架构设计：Pre-LN Transformer
//!
//! 本实现采用 **Pre-LN (Pre-Layer Normalization)** 架构，这是 GPT-2 之后的标准做法。
//! 相比原始 Transformer 的 Post-LN，Pre-LN 具有更好的训练稳定性。
//!
//! ### Pre-LN vs Post-LN 对比
//!
//! **Post-LN（原始 Transformer）**：
//! ```text
//! x = x + Attention(x)
//! x = LayerNorm(x)
//! x = x + FFN(x)
//! x = LayerNorm(x)
//! ```
//!
//! **Pre-LN（本实现，GPT-2/GPT-3 使用）**：
//! ```text
//! x = x + Attention(LayerNorm(x))
//! x = x + FFN(LayerNorm(x))
//! ```
//!
//! **Pre-LN 的优势**：
//! - 训练更稳定：归一化在残差路径之外，梯度流动更顺畅
//! - 不需要学习率预热（Warmup）
//! - 更容易训练深层网络
//!
//! ## 数据流示意图
//!
//! ```text
//! 输入 (seq_len, 512)
//!   │
//!   ├─────────────────┐  [残差连接分支]
//!   │                 │
//!   v                 │
//! LayerNorm          │
//!   │                 │
//!   v                 │
//! MultiHeadAttention │
//!   │                 │
//!   v                 │
//! Dropout (10%)      │
//!   │                 │
//!   └─────────(+)─────┘  [残差相加]
//!   │
//!   ├─────────────────┐  [残差连接分支]
//!   │                 │
//!   v                 │
//! LayerNorm          │
//!   │                 │
//!   v                 │
//! Feed-Forward       │
//!   │                 │
//!   v                 │
//! Dropout (10%)      │
//!   │                 │
//!   └─────────(+)─────┘  [残差相加]
//!   │
//!   v
//! 输出 (seq_len, 512)
//! ```

use ndarray::Array2;

use crate::{
    dropout::Dropout, feed_forward::FeedForward, layer_norm::LayerNorm, llm::Layer,
    self_attention::SelfAttention,
};

/// **Transformer 块结构体**
///
/// 一个完整的 Transformer 层，包含注意力和前馈网络两个子层。
///
/// ## 组件说明
///
/// - **attention**: 8头自注意力机制，负责捕捉序列中不同位置之间的依赖关系
/// - **feed_forward**: 两层全连接网络 (512→1024→512)，负责特征变换
/// - **norm1, norm2**: 层归一化，稳定训练过程
/// - **dropout1, dropout2**: 10% 的 dropout 率，防止过拟合
pub struct TransformerBlock {
    /// 多头自注意力层：学习序列内的相互关系
    pub attention: SelfAttention,

    /// 前馈神经网络：对每个位置独立进行非线性变换
    pub feed_forward: FeedForward,

    /// 第一个 Dropout 层：在注意力之后应用
    pub dropout1: Dropout,

    /// 第二个 Dropout 层：在前馈网络之后应用
    pub dropout2: Dropout,

    /// 第一个归一化层：在注意力之前应用（Pre-LN）
    pub norm1: LayerNorm,

    /// 第二个归一化层：在前馈网络之前应用（Pre-LN）
    pub norm2: LayerNorm,
}

impl TransformerBlock {
    /// **创建新的 Transformer 块**
    ///
    /// # 参数
    /// - `embedding_dim`: 嵌入维度（通常为512），决定输入输出的大小
    /// - `hidden_dim`: 前馈网络的隐藏层维度（通常为1024，是 embedding_dim 的2倍）
    ///
    /// # 初始化细节
    /// 1. **SelfAttention**: 8个注意力头，每个头处理 embedding_dim/8 = 64 维
    /// 2. **FeedForward**: 512 → 1024 → 512 的两层网络，使用 ReLU 激活
    /// 3. **LayerNorm**: 对特征维度进行归一化，使用可学习的 scale/shift 参数
    /// 4. **Dropout**: 10% 的丢弃率，只在训练时生效
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            dropout1: Dropout::new(0.1), // 10% dropout，防止注意力过拟合
            dropout2: Dropout::new(0.1), // 10% dropout，防止FFN过拟合
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }

    /// 推理路径：在 KV 缓存开启时使用增量前向传播
    pub fn forward_inference(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let norm1_out = self.norm1.normalize(input);
        let attention_out = if self.attention.use_kv_cache {
            self.attention.forward_with_kv_cache(&norm1_out)
        } else {
            self.attention.forward(&norm1_out)
        };
        let dropout1_out = self.dropout1.forward(&attention_out);

        let x = input + &dropout1_out;

        let norm2_out = self.norm2.normalize(&x);
        let feed_forward_out = self.feed_forward.forward(&norm2_out);
        let dropout2_out = self.dropout2.forward(&feed_forward_out);

        &x + &dropout2_out
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// **前向传播：Pre-LN Transformer 的数据流**
    ///
    /// # 算法步骤
    ///
    /// ## 第一个子层：多头自注意力
    /// ```text
    /// 1. norm1_out = LayerNorm(input)              // 先归一化
    /// 2. attention_out = SelfAttention(norm1_out)   // 计算注意力
    /// 3. dropout1_out = Dropout(attention_out)      // 随机丢弃
    /// 4. x = input + dropout1_out                   // 残差连接
    /// ```
    ///
    /// ## 第二个子层：前馈神经网络
    /// ```text
    /// 5. norm2_out = LayerNorm(x)                   // 再次归一化
    /// 6. ffn_out = FeedForward(norm2_out)          // 非线性变换
    /// 7. dropout2_out = Dropout(ffn_out)           // 再次丢弃
    /// 8. output = x + dropout2_out                 // 最终残差连接
    /// ```
    ///
    /// # 残差连接的作用
    ///
    /// 残差连接（Residual Connection）让梯度能直接流回输入层，解决深度网络的梯度消失问题。
    /// 可以理解为：output = 原始特征 + 学习到的新特征
    ///
    /// # 参数
    /// - `input`: 输入张量，形状 (seq_len, embedding_dim)
    ///
    /// # 返回值
    /// 输出张量，形状与输入相同 (seq_len, embedding_dim)
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // ========== 第一个子层：多头自注意力 ==========
        // Pre-LN 架构：先归一化，再应用注意力
        let norm1_out = self.norm1.normalize(input);
        let attention_out = self.attention.forward(&norm1_out);
        let dropout1_out = self.dropout1.forward(&attention_out);

        // 残差连接：保留原始信息，避免信息丢失
        let x = input + &dropout1_out;

        // ========== 第二个子层：前馈神经网络 ==========
        // 同样采用 Pre-LN：先归一化，再应用FFN
        let norm2_out = self.norm2.normalize(&x);
        let feed_forward_out = self.feed_forward.forward(&norm2_out);
        let dropout2_out = self.dropout2.forward(&feed_forward_out);

        // 最终残差连接：整合注意力和前馈的输出
        &x + &dropout2_out
    }

    /// **反向传播：计算梯度并更新参数**
    ///
    /// # 反向传播顺序
    ///
    /// 反向传播是前向传播的镜像，但顺序相反。对于每个残差连接 `y = x + f(x)`，
    /// 梯度分为两路：
    /// 1. 直接路径：`grad_x = grad_y`（残差连接直接传递梯度）
    /// 2. 变换路径：`grad_x += backward(f, grad_y)`（通过子层反向传播）
    ///
    /// ## 详细步骤
    ///
    /// ```text
    /// 输入: grad_output (来自上一层的梯度)
    ///
    /// 1. 处理第二个残差连接（FFN部分）:
    ///    grad_x = grad_output                      // 残差路径
    ///    grad_ffn = Dropout2.backward(grad_output)
    ///    grad_ffn = FFN.backward(grad_ffn)
    ///    grad_ffn = Norm2.backward(grad_ffn)
    ///    grad_x += grad_ffn                         // 累加梯度
    ///
    /// 2. 处理第一个残差连接（注意力部分）:
    ///    grad_input = grad_x                        // 残差路径
    ///    grad_attn = Dropout1.backward(grad_x)
    ///    grad_attn = Attention.backward(grad_attn)
    ///    grad_attn = Norm1.backward(grad_attn)
    ///    grad_input += grad_attn                    // 最终输入梯度
    /// ```
    ///
    /// # 参数
    /// - `grads`: 来自后续层的梯度 (seq_len, embedding_dim)
    /// - `lr`: 当前学习率
    ///
    /// # 返回值
    /// 传递给前一层的梯度 (seq_len, embedding_dim)
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // ========== 反向传播第二个残差连接 ==========
        // 梯度分为两路：残差路径 + 变换路径

        // 残差路径：梯度直接传递
        let grad_dropout2 = grads;
        let grad_x_from_residual2 = grads;

        // 变换路径：依次通过 Dropout2 → FFN → Norm2
        let grad_dropout2_out = self.dropout2.backward(grad_dropout2, lr);
        let grad_ffn = self.feed_forward.backward(&grad_dropout2_out, lr);
        let grad_norm2 = self.norm2.backward(&grad_ffn, lr);

        // 累积两路梯度
        let grad_x = &grad_norm2 + grad_x_from_residual2;

        // ========== 反向传播第一个残差连接 ==========

        // 残差路径
        let grad_dropout1 = &grad_x;
        let grad_input_from_residual1 = &grad_x;

        // 变换路径：依次通过 Dropout1 → Attention → Norm1
        let grad_dropout1_out = self.dropout1.backward(grad_dropout1, lr);
        let grad_attention = self.attention.backward(&grad_dropout1_out, lr);
        let grad_norm1 = self.norm1.backward(&grad_attention, lr);

        // 累积两路梯度，得到最终输入梯度
        &grad_norm1 + grad_input_from_residual1
    }

    /// **设置训练/推理模式**
    ///
    /// 训练模式和推理模式的主要区别在于 Dropout：
    /// - **训练模式 (training=true)**: Dropout 生效，随机丢弃神经元
    /// - **推理模式 (training=false)**: Dropout 关闭，使用全部神经元
    ///
    /// 这个方法会递归设置所有子层的模式。
    fn set_training_mode(&mut self, training: bool) {
        self.dropout1.set_training_mode(training);
        self.dropout2.set_training_mode(training);
        self.attention.set_training_mode(training);
        self.feed_forward.set_training_mode(training);
        self.norm1.set_training_mode(training);
        self.norm2.set_training_mode(training);
    }

    /// **计算参数总数**
    ///
    /// 统计这个 Transformer 块中所有可训练参数的数量。
    /// 包括：
    /// - 自注意力的 Q/K/V/O 权重矩阵
    /// - 前馈网络的两层权重和偏置
    /// - 两个 LayerNorm 的 scale/shift 参数
    ///
    /// Dropout 没有可训练参数，所以返回0。
    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.dropout1.parameters()  // 返回 0
            + self.dropout2.parameters()  // 返回 0
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}
