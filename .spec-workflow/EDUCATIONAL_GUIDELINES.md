# 教育性代码指南 - RustGPT-Chinese

## 🎓 项目教育目标

本项目的首要目标是**教学**，而非生产级性能。每一行代码都应当：
- ✅ 易于理解和学习
- ✅ 展示核心原理和算法
- ✅ 保持最小依赖，从零实现
- ✅ 通过注释阐明设计决策

---

## 📝 注释原则

### 1. 必须注释的内容

#### 数学公式和算法
```rust
/// 计算缩放点积注意力（Scaled Dot-Product Attention）
///
/// # 算法
/// ```
/// scores = (Q @ K^T) / sqrt(d_k)
/// attention = softmax(scores) @ V
/// ```
///
/// 缩放因子 sqrt(d_k) 用于防止 softmax 梯度消失
fn compute_attention(query: &Array2<f32>, key: &Array2<f32>, value: &Array2<f32>) -> Array2<f32> {
    // 1. 计算注意力分数
    let d_k = (key.ncols() as f32).sqrt();
    let scores = query.dot(&key.t()) / d_k;  // Q @ K^T / √d_k
    
    // 2. Softmax 归一化（数值稳定性处理）
    // 减去最大值防止 exp() 溢出
    let max_score = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores = (scores - max_score).mapv(f32::exp);
    let sum_exp = exp_scores.sum_axis(Axis(1)).insert_axis(Axis(1));
    let attention_weights = &exp_scores / &sum_exp;
    
    // 3. 加权求和
    attention_weights.dot(value)
}
```

#### 非显而易见的优化
```rust
// 使用 KV-Cache 避免推理时重复计算历史 token 的 Key 和 Value
// 原理：自回归生成时，每次只需计算新 token 的注意力
if self.kv_cache_enabled {
    self.cached_keys.push(current_key);
    self.cached_values.push(current_value);
    
    // 拼接历史和当前 Key/Value
    let all_keys = concatenate_cached(&self.cached_keys);
    let all_values = concatenate_cached(&self.cached_values);
}
```

#### 关键设计决策
```rust
// 使用 Pre-LN 架构（LayerNorm 在 Attention 前）
// 理由：训练稳定性更好，收敛速度快 20%（相比 Post-LN）
// 参考: GPT-2/3, BERT 后期模型的标准做法
let normalized = self.layer_norm.forward(input);
let attention_out = self.attention.forward(&normalized);
let residual = input + &attention_out;  // 残差连接
```

#### 数值稳定性技巧
```rust
// Softmax 数值稳定性：减去最大值防止 exp() 溢出
// 数学上: softmax(x - max(x)) = softmax(x)（指数差不变）
let max_val = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
let exp_logits = (logits - max_val).mapv(f32::exp);
let sum_exp = exp_logits.sum();
let probabilities = &exp_logits / sum_exp;
```

### 2. 不需要注释的内容

#### 自解释的代码
```rust
// ❌ 不好：注释重复代码内容
// 计算嵌入维度
let embedding_dim = 256;

// ✅ 好：代码已经很清晰
let embedding_dim = 256;
```

#### 显而易见的操作
```rust
// ❌ 不好
// 调用 forward 方法
let output = layer.forward(&input);

// ✅ 好：无需注释
let output = layer.forward(&input);
```

### 3. 注释风格指南

#### 函数文档注释（使用 Rust Doc 格式）
```rust
/// 对输入张量应用 GELU 激活函数
///
/// GELU（Gaussian Error Linear Unit）是一种平滑的 ReLU 变体，
/// 在 Transformer 模型中广泛使用。
///
/// # 数学定义
/// ```
/// GELU(x) = x * Φ(x)
/// 其中 Φ(x) 是标准正态分布的累积分布函数
/// ```
///
/// # 近似实现
/// 使用 tanh 近似:
/// ```
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// # 参数
/// - `x`: 输入值
///
/// # 返回
/// 激活后的值
///
/// # 示例
/// ```
/// let activated = gelu(2.0);
/// assert!(activated > 1.9 && activated < 2.1);
/// ```
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;  // √(2/π)
    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}
```

#### 行内注释（解释"为什么"而非"是什么"）
```rust
// ✅ 好：解释原因
// 梯度裁剪防止梯度爆炸，阈值 5.0 是 Transformer 的经验值
if grad_norm > 5.0 {
    gradients = gradients * (5.0 / grad_norm);
}

// ❌ 不好：重复代码
// 如果梯度范数大于 5.0，则裁剪梯度
if grad_norm > 5.0 {
    gradients = gradients * (5.0 / grad_norm);
}
```

---

## 🏗️ 代码结构原则

### 1. 单一职责原则

每个文件/模块只做一件事：

```rust
// ✅ 好：单一职责
// src/layer_norm.rs - 只实现 LayerNorm
pub struct LayerNorm { ... }
impl Layer for LayerNorm { ... }

// ❌ 不好：多个不相关功能
// src/utils.rs - 混杂各种功能
pub struct LayerNorm { ... }
pub struct Dropout { ... }
pub fn load_data() { ... }
```

### 2. 显式优于隐式

明确展示每一步操作，避免"魔法"：

```rust
// ✅ 好：显式的矩阵乘法步骤
let query = input.dot(&self.w_q);  // X @ W_q
let key = input.dot(&self.w_k);    // X @ W_k
let value = input.dot(&self.w_v);  // X @ W_v

// ❌ 不好：隐藏细节的高级抽象
let qkv = self.qkv_projection(input);  // 不清楚内部做了什么
```

### 3. 教育友好的变量名

使用描述性名称，即使较长：

```rust
// ✅ 好：清晰表达意图
let attention_weights = softmax(&attention_scores);
let weighted_values = attention_weights.dot(&value_matrix);

// ❌ 不好：数学符号（除非在注释中解释）
let a = softmax(&s);
let w = a.dot(&v);
```

### 4. 分步骤实现复杂算法

将复杂算法分解为多个清晰的步骤：

```rust
/// 多头自注意力的前向传播
///
/// 分为三个主要步骤：
/// 1. 线性投影生成 Q, K, V
/// 2. 分割多头并计算注意力
/// 3. 拼接多头并输出投影
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // ========== 步骤 1: 线性投影 ==========
    let query = input.dot(&self.w_q);
    let key = input.dot(&self.w_k);
    let value = input.dot(&self.w_v);
    
    // ========== 步骤 2: 多头注意力 ==========
    let mut head_outputs = Vec::new();
    for head_idx in 0..self.num_heads {
        let q_head = self.split_head(&query, head_idx);
        let k_head = self.split_head(&key, head_idx);
        let v_head = self.split_head(&value, head_idx);
        
        let attention_out = self.compute_single_head_attention(q_head, k_head, v_head);
        head_outputs.push(attention_out);
    }
    
    // ========== 步骤 3: 拼接和输出投影 ==========
    let concatenated = self.concatenate_heads(&head_outputs);
    concatenated.dot(&self.w_o)
}
```

---

## 🧮 数学实现规范

### 1. 公式注释格式

使用 Markdown 代码块标注数学公式：

```rust
/// # 算法：Adam 优化器参数更新
/// ```
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        // 一阶矩（动量）
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       // 二阶矩（RMSProp）
/// m̂_t = m_t / (1 - β₁^t)                     // 偏差修正
/// v̂_t = v_t / (1 - β₂^t)
/// θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)      // 参数更新
/// ```
```

### 2. 变量与公式对应

代码变量名与数学符号对应清晰：

```rust
// 论文公式: Attention(Q, K, V) = softmax(Q K^T / √d_k) V
fn scaled_dot_product_attention(
    query: &Array2<f32>,   // Q
    key: &Array2<f32>,     // K
    value: &Array2<f32>,   // V
    d_k: f32,              // 键的维度
) -> Array2<f32> {
    // QK^T
    let scores = query.dot(&key.t());
    
    // QK^T / √d_k
    let scaled_scores = scores / d_k.sqrt();
    
    // softmax(QK^T / √d_k)
    let attention_weights = softmax(&scaled_scores);
    
    // softmax(...) V
    attention_weights.dot(value)
}
```

### 3. 梯度推导注释

反向传播必须注释梯度推导：

```rust
/// # 反向传播：LayerNorm 梯度推导
///
/// ## 前向公式
/// ```
/// μ = mean(x)
/// σ² = var(x)
/// x̂ = (x - μ) / √(σ² + ε)
/// y = γ * x̂ + β
/// ```
///
/// ## 反向梯度（链式法则）
/// ```
/// ∂L/∂γ = Σ (∂L/∂y * x̂)
/// ∂L/∂β = Σ (∂L/∂y)
/// ∂L/∂x = (∂L/∂y * γ / √(σ² + ε)) * (1 - 1/N - (x̂)² / N)
/// ```
fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
    // 1. 计算 ∂L/∂γ 和 ∂L/∂β
    let grad_gamma = (grad_output * &self.normalized_input).sum_axis(Axis(0));
    let grad_beta = grad_output.sum_axis(Axis(0));
    
    // 2. 计算 ∂L/∂x̂
    let grad_normalized = grad_output * &self.gamma.view().insert_axis(Axis(0));
    
    // 3. 计算 ∂L/∂x（反向传播通过归一化）
    let grad_input = self.compute_input_gradient(&grad_normalized);
    
    // 4. 更新参数
    self.gamma = &self.gamma - lr * &grad_gamma;
    self.beta = &self.beta - lr * &grad_beta;
    
    grad_input
}
```

---

## 🧪 测试规范

### 1. 测试驱动的教学

每个功能必须有对应测试：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 测试：验证 GELU 激活函数在 x=0 时的值
    /// 预期: GELU(0) = 0
    #[test]
    fn test_gelu_at_zero() {
        let result = gelu(0.0);
        assert_eq!(result, 0.0, "GELU(0) 应该等于 0");
    }

    /// 测试：验证 GELU 的平滑性质
    /// GELU 应该在负数区域有小的梯度（不像 ReLU 直接截断为 0）
    #[test]
    fn test_gelu_smoothness() {
        let negative_result = gelu(-1.0);
        assert!(
            negative_result < 0.0 && negative_result > -1.0,
            "GELU 应该在负数区域有非零输出，体现平滑性"
        );
    }

    /// 测试：验证正数区域接近线性
    /// 当 x 较大时，GELU(x) ≈ x
    #[test]
    fn test_gelu_linearity_at_large_values() {
        let x = 3.0;
        let result = gelu(x);
        assert!(
            (result - x).abs() < 0.1,
            "GELU 在大正数时应接近 x"
        );
    }
}
```

### 2. 测试作为文档

测试用例展示如何使用模块：

```rust
/// 示例：如何使用 Vocab 构建词汇表并编码文本
#[test]
fn test_vocab_usage_example() {
    // 1. 准备训练文本
    let texts = vec![
        "深度学习很有趣".to_string(),
        "Transformer 是一种神经网络".to_string(),
    ];
    
    // 2. 构建词汇表（自动使用 Jieba 分词）
    let vocab = Vocab::build_from_texts(&texts);
    
    // 3. 编码文本为 token IDs
    let text = "深度学习";
    let token_ids = vocab.encode_sequence(text);
    
    // 4. 解码 token IDs 回文本
    let decoded = vocab.decode_sequence(&token_ids);
    
    assert!(decoded.contains("深度"));
    assert!(decoded.contains("学习"));
}
```

### 3. 边界测试（教学价值高）

测试边界情况帮助理解算法限制：

```rust
/// 测试：空输入
#[test]
fn test_forward_with_empty_input() {
    let layer = FeedForward::new(256, 512);
    let empty_input = Array2::zeros((0, 256));
    let output = layer.forward(&empty_input);
    assert_eq!(output.shape(), &[0, 256]);
}

/// 测试：超长序列截断
#[test]
fn test_context_truncation_at_max_length() {
    let mut model = LLM::new(vocab);
    
    // 添加超过 MAX_SEQ_LEN 的 token
    for _ in 0..200 {
        model.context.push(1);  // 添加 200 个 token
    }
    
    // 应该只保留最近的 128 个（MAX_SEQ_LEN）
    assert_eq!(model.context.len(), 128);
}
```

---

## 📚 依赖管理哲学

### 只添加无法简单实现的依赖

#### ✅ 可接受的依赖
```rust
// ndarray - 张量计算是核心，手写效率低且易错
use ndarray::{Array2, Axis};

// jieba-rs - 中文分词是专门领域，需要词典和统计模型
use jieba_rs::Jieba;

// serde - 序列化是通用需求，标准库未提供
use serde::{Serialize, Deserialize};
```

#### ❌ 应避免的依赖
```rust
// PyTorch 绑定 - 隐藏实现细节，违背教学目标
// use tch::{Tensor, nn};  ❌

// 自动微分库 - 手写梯度有教学价值
// use autograd::{grad, backward};  ❌

// CSV 解析 - 简单格式可自己实现
// use csv::Reader;  ❌
```

### 最小化依赖原则

```toml
# ✅ 好：只依赖必需的核心功能
[dependencies]
ndarray = "0.16"
jieba-rs = "0.7"

# ❌ 不好：添加"可能有用"的依赖
[dependencies]
ndarray = "0.16"
jieba-rs = "0.7"
tokio = "1.0"       # 异步运行时（本项目不需要）
reqwest = "0.11"    # HTTP 客户端（本项目不需要）
clap = "4.0"        # 命令行解析（当前简单交互足够）
```

---

## 🎨 代码美学

### 1. 对齐和格式

使用一致的对齐提高可读性：

```rust
// ✅ 好：对齐的结构体字段
pub struct TransformerConfig {
    pub max_seq_len:    usize,  // 128
    pub embedding_dim:  usize,  // 256
    pub hidden_dim:     usize,  // 512
    pub num_heads:      usize,  // 8
    pub num_layers:     usize,  // 2
    pub dropout_rate:   f32,    // 0.1
}

// ✅ 好：对齐的参数赋值
let config = TransformerConfig {
    max_seq_len:    128,
    embedding_dim:  256,
    hidden_dim:     512,
    num_heads:      8,
    num_layers:     2,
    dropout_rate:   0.1,
};
```

### 2. 分隔复杂逻辑

使用空行和注释分隔逻辑块：

```rust
fn train_epoch(&mut self, dataset: &[String], lr: f32) {
    // ========== 阶段 1: 数据预处理 ==========
    let tokenized_data = self.preprocess_dataset(dataset);
    let total_samples = tokenized_data.len();
    
    println!("开始训练，共 {} 个样本", total_samples);
    
    // ========== 阶段 2: 前向传播和损失计算 ==========
    let mut total_loss = 0.0;
    for sample in &tokenized_data {
        let output = self.forward(sample, true);  // training=true
        let loss = self.compute_loss(&output, sample);
        total_loss += loss;
    }
    
    // ========== 阶段 3: 反向传播和参数更新 ==========
    let avg_loss = total_loss / total_samples as f32;
    self.backward(&loss_gradient, lr);
    
    // ========== 阶段 4: 指标记录 ==========
    self.metrics.push(avg_loss);
    println!("Epoch 完成，平均 Loss: {:.4}", avg_loss);
}
```

### 3. 一致的命名风格

```rust
// ✅ 好：一致的命名模式
pub struct Embeddings { ... }      // 名词，复数
pub struct LayerNorm { ... }       // 名词组合
pub struct SelfAttention { ... }   // 名词组合

fn forward(&mut self, ...) { ... }      // 动词
fn compute_loss(...) { ... }            // 动词 + 名词
fn build_vocabulary(...) { ... }        // 动词 + 名词

// ❌ 不好：混乱的命名
pub struct Embed { ... }           // 动词形式
pub struct NormLayer { ... }       // 顺序不一致

fn go_forward(...) { ... }         // 冗余的 "go"
fn loss_computation(...) { ... }   // 名词形式
```

---

## 🚫 反模式（应避免）

### 1. 过度抽象

```rust
// ❌ 不好：为了抽象而抽象（增加理解难度）
trait Computable {
    fn compute(&self) -> Box<dyn Any>;
}

struct Layer {
    operation: Box<dyn Computable>,
}

// ✅ 好：直接实现（清晰易懂）
pub struct FeedForward {
    w1: Array2<f32>,
    b1: Array1<f32>,
}

impl FeedForward {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // 直接的矩阵运算
        input.dot(&self.w1) + &self.b1
    }
}
```

### 2. 隐藏的魔法数字

```rust
// ❌ 不好：没有解释的魔法数字
let output = input * 0.044715;

// ✅ 好：常量 + 注释说明
const GELU_CUBIC_COEFF: f32 = 0.044715;  // GELU 激活函数的三次项系数
let output = input * GELU_CUBIC_COEFF;
```

### 3. 过度使用宏

```rust
// ❌ 不好：复杂的宏隐藏逻辑
macro_rules! define_layer {
    ($name:ident, $dim:expr) => {
        // 复杂的宏展开逻辑...
    };
}

// ✅ 好：显式的结构体定义
pub struct FeedForward {
    input_dim: usize,
    hidden_dim: usize,
}

impl FeedForward {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self { ... }
}
```

### 4. 过度优化（以牺牲可读性为代价）

```rust
// ❌ 不好：过度优化导致难以理解
let result = input
    .axis_iter(Axis(0))
    .zip(weights.axis_iter(Axis(1)))
    .flat_map(|(i, w)| i.iter().zip(w.iter()))
    .map(|(x, w)| x * w)
    .sum::<f32>();

// ✅ 好：清晰的矩阵乘法（ndarray 内部已优化）
let result = input.dot(&weights);
```

---

## 📖 学习路径建议

### 新手入门顺序

对于想要学习本项目的开发者，建议按以下顺序阅读代码：

1. **lib.rs** - 理解全局配置和 Layer trait
2. **main.rs** - 理解整体训练流程
3. **vocab.rs** - 理解中文分词和词汇表
4. **embeddings.rs** - 第一个简单的层
5. **layer_norm.rs** - 理解归一化
6. **feed_forward.rs** - 理解前馈网络
7. **self_attention.rs** - 理解注意力机制（核心）
8. **transformer.rs** - 理解层的组合
9. **llm.rs** - 理解前向/反向传播
10. **tests/** - 通过测试理解每个模块

### 进阶学习建议

#### 修改实验
1. **调整超参数**: 修改 `lib.rs` 中的配置，观察效果
2. **更换激活函数**: 将 GELU 替换为 ReLU，对比收敛速度
3. **添加新层**: 实现 Batch Normalization 或 Group Normalization
4. **改进采样**: 实现 Temperature Annealing 或 Contrastive Search

#### 深入理解
1. **手动推导梯度**: 对比代码中的反向传播实现
2. **可视化注意力**: 输出注意力权重矩阵并可视化
3. **分析性能**: 使用 `cargo flamegraph` 找出性能瓶颈
4. **阅读论文**: 对照 "Attention Is All You Need" 论文

---

## 🎯 代码审查 Checklist（教育视角）

### 提交代码前自查

- [ ] **注释充分**: 复杂算法有公式和推导说明
- [ ] **变量命名**: 描述性名称，避免单字母（除数学惯例）
- [ ] **逻辑分步**: 复杂函数分解为多个清晰步骤
- [ ] **测试完整**: 至少有基本的功能测试
- [ ] **文档更新**: 新功能更新了 CLAUDE.md 或 README
- [ ] **最小依赖**: 没有引入不必要的依赖
- [ ] **无 unsafe**: 没有使用 unsafe 代码（除依赖库内部）
- [ ] **无魔法数字**: 常量有清晰命名和注释
- [ ] **格式规范**: 通过 `cargo fmt` 检查
- [ ] **无警告**: 通过 `cargo clippy` 检查

### 代码审查问题清单

审查他人代码时应问：

1. **教育价值**: 这段代码是否帮助学习者理解算法原理？
2. **可读性**: 没有相关知识的人能否通过注释理解？
3. **必要性**: 这个功能/依赖是否必需？
4. **简洁性**: 能否用更简单的方式实现？
5. **测试性**: 是否有测试用例验证功能？

---

## 💡 示例：教育友好的代码

### 完整示例：实现一个新层

```rust
//! # Dropout 正则化层
//!
//! Dropout 是一种防止过拟合的正则化技术，通过在训练时随机丢弃部分神经元。
//!
//! ## 原理
//! - **训练时**: 以概率 p 随机将神经元输出置为 0
//! - **推理时**: 不丢弃，但输出缩放为 (1 - p) 倍
//!
//! ## 为什么有效
//! 迫使网络不依赖特定神经元，学习更鲁棒的特征。
//!
//! ## 参考
//! - 论文: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., 2014)

use ndarray::Array2;
use rand::Rng;

use crate::Layer;

/// Dropout 层结构体
///
/// # 字段
/// - `dropout_rate`: 丢弃概率（通常 0.1 ~ 0.5）
/// - `training`: 训练模式开关
/// - `mask`: 缓存的 Dropout 掩码（反向传播需要）
pub struct Dropout {
    dropout_rate: f32,
    training: bool,
    mask: Option<Array2<f32>>,
}

impl Dropout {
    /// 创建新的 Dropout 层
    ///
    /// # 参数
    /// - `dropout_rate`: 丢弃概率（0.0 ~ 1.0）
    ///
    /// # 示例
    /// ```
    /// let dropout = Dropout::new(0.1);  // 10% 的神经元被丢弃
    /// ```
    pub fn new(dropout_rate: f32) -> Self {
        assert!(
            dropout_rate >= 0.0 && dropout_rate < 1.0,
            "Dropout rate 必须在 [0, 1) 范围内"
        );
        
        Self {
            dropout_rate,
            training: true,
            mask: None,
        }
    }
}

impl Layer for Dropout {
    fn layer_type(&self) -> &str {
        "Dropout"
    }
    
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if !self.training {
            // 推理模式：直接返回输入
            return input.clone();
        }
        
        // ========== 训练模式：应用 Dropout ==========
        
        // 1. 生成随机掩码：0 或 1
        //    如果 rand() > dropout_rate，则保留（mask = 1）
        //    否则丢弃（mask = 0）
        let mut rng = rand::thread_rng();
        let mask = input.mapv(|_| {
            if rng.gen::<f32>() > self.dropout_rate {
                1.0
            } else {
                0.0
            }
        });
        
        // 2. 应用掩码并缩放
        //    缩放因子: 1 / (1 - p)
        //    原因：保持期望值不变（E[output] = E[input]）
        let keep_prob = 1.0 - self.dropout_rate;
        let output = (input * &mask) / keep_prob;
        
        // 3. 缓存掩码（反向传播需要）
        self.mask = Some(mask);
        
        output
    }
    
    fn backward(&mut self, grad_output: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // Dropout 没有可学习参数，直接传递梯度
        
        if !self.training {
            // 推理模式：直接传递梯度
            return grad_output.clone();
        }
        
        // 训练模式：应用相同的掩码和缩放
        let mask = self.mask.as_ref().expect("Forward 必须先于 Backward 调用");
        let keep_prob = 1.0 - self.dropout_rate;
        
        (grad_output * mask) / keep_prob
    }
    
    fn parameters(&self) -> usize {
        0  // Dropout 没有可学习参数
    }
    
    fn set_training_mode(&mut self, training: bool) {
        self.training = training;
    }
}

// ========== 单元测试 ==========

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    /// 测试：推理模式不应修改输入
    #[test]
    fn test_inference_mode_no_dropout() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training_mode(false);  // 推理模式
        
        let input = Array::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let output = dropout.forward(&input);
        
        assert_eq!(input, output, "推理模式不应修改输入");
    }

    /// 测试：训练模式应有部分元素为 0
    #[test]
    fn test_training_mode_drops_elements() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training_mode(true);  // 训练模式
        
        let input = Array::from_shape_vec((100, 100), vec![1.0; 10000]).unwrap();
        let output = dropout.forward(&input);
        
        // 统计为 0 的元素数量
        let zero_count = output.iter().filter(|&&x| x == 0.0).count();
        
        // 50% dropout 应该有约一半元素为 0（允许 10% 误差）
        let expected = 5000;
        let tolerance = 500;
        assert!(
            (zero_count as i32 - expected).abs() < tolerance,
            "Dropout 应丢弃约 50% 的元素，实际丢弃 {}%",
            zero_count as f32 / 10000.0 * 100.0
        );
    }

    /// 测试：期望值保持不变
    #[test]
    fn test_expectation_invariance() {
        let mut dropout = Dropout::new(0.3);
        dropout.set_training_mode(true);
        
        let input = Array::from_shape_vec((1000, 100), vec![2.0; 100000]).unwrap();
        let output = dropout.forward(&input);
        
        // 计算平均值（应该接近 2.0）
        let mean = output.mean().unwrap();
        assert!(
            (mean - 2.0).abs() < 0.1,
            "Dropout 应保持期望值不变，输入均值 2.0，输出均值 {}",
            mean
        );
    }
}
```

---

*本指南最后更新: 2024-10-25 | 版本: v0.4.0*
