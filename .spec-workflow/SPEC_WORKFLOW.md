# RustGPT-Chinese 项目规范与工作流程

## 📋 项目概述

**项目名称**: RustGPT-Chinese  
**版本**: v0.4.0  
**项目类型**: 教育性深度学习框架  
**核心目标**: 从零实现中文 Transformer 语言模型，用于教学和理解 LLM 内部机制

### 项目定位

这是一个**教育优先**的项目，旨在通过清晰、可读的 Rust 代码展示：
- Transformer 架构的完整实现细节
- 反向传播和梯度计算的工作原理
- 中文 NLP 处理的技术挑战
- 现代优化算法（Adam）的实际应用

**核心原则**:
1. ✅ **代码清晰度优先** - 可读性 > 性能优化
2. ✅ **最小依赖原则** - 只依赖基础库（ndarray, jieba-rs）
3. ✅ **从零实现** - 不使用 PyTorch/TensorFlow/Candle
4. ✅ **注释充分** - 关键算法必须有详细解释
5. ✅ **模块化设计** - 每个组件独立可测试

---

## 🏗️ 架构规范

### 1. 模块层次结构

```
RustGPT-Chinese
│
├── Core Neural Network Layers (核心神经网络层)
│   ├── embeddings.rs          # Token 嵌入 + 位置编码
│   ├── self_attention.rs      # 多头自注意力机制
│   ├── feed_forward.rs        # 前馈神经网络（GELU激活）
│   ├── layer_norm.rs          # 层归一化
│   ├── dropout.rs             # Dropout 正则化
│   ├── output_projection.rs   # 输出投影层（vocab_size）
│   └── transformer.rs         # Transformer Block（组合层）
│
├── Model Orchestration (模型编排)
│   ├── llm.rs                 # LLM 主类（前向/反向传播）
│   └── lib.rs                 # 全局配置和 Layer trait
│
├── Training Infrastructure (训练基础设施)
│   ├── adam.rs                # Adam 优化器
│   ├── training_optimizations.rs  # 学习率调度、早停、梯度累积
│   ├── batch_loader.rs        # 批量数据加载和缓存
│   └── checkpoint_manager.rs  # 模型检查点管理
│
├── Data Processing (数据处理)
│   ├── vocab.rs               # 词汇表构建（Jieba分词 + LRU缓存）
│   ├── dataset_loader.rs      # JSON 数据加载
│   └── utils.rs               # 通用工具函数
│
├── Performance Optimizations (性能优化)
│   ├── fused_ops.rs           # 融合算子（LayerNorm+Linear, GELU+Linear）
│   ├── position_encoding.rs   # 正弦位置编码（预计算缓存）
│   └── performance_monitor.rs # 性能指标监控
│
└── Serialization (序列化)
    └── model_serialization.rs # 模型保存/加载（二进制 + JSON）
```

### 2. Layer Trait 规范

所有神经网络层必须实现统一的 `Layer` trait：

```rust
pub trait Layer: Send + Sync {
    /// 返回层的类型名称（用于调试和日志）
    fn layer_type(&self) -> &str;
    
    /// 前向传播：输入 → 输出
    /// - input: (batch_size, seq_len, feature_dim)
    /// - output: (batch_size, seq_len, output_dim)
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    
    /// 反向传播：梯度 → 梯度（返回传递给上一层的梯度）
    /// - grads: 来自下一层的梯度
    /// - lr: 学习率
    /// - 返回: 传递给上一层的梯度
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
    
    /// 返回层的可训练参数数量
    fn parameters(&self) -> usize;
    
    /// 设置训练/推理模式（影响 Dropout 行为）
    fn set_training_mode(&mut self, training: bool);
}
```

**设计原则**:
- ✅ 统一接口便于组合和替换层
- ✅ `&mut self` 允许缓存中间结果（反向传播需要）
- ✅ 返回梯度实现链式反向传播

### 3. 数据流规范

#### 训练流程
```
Raw Text (原始文本)
    ↓
Jieba 分词 (vocab.rs + LRU 缓存)
    ↓
Token IDs (整数序列)
    ↓
Embeddings (token + position) [256维]
    ↓
Transformer Block 1
    ├─ LayerNorm → MultiHeadAttention (8 heads) → Dropout (10%) → Residual
    └─ LayerNorm → FeedForward (512d) → Dropout (10%) → Residual
    ↓
Transformer Block 2 (同上)
    ↓
Output Projection [vocab_size 维]
    ↓
Softmax → Cross-Entropy Loss
    ↓
Backward Pass (梯度反向传播)
    ↓
Adam Optimizer (参数更新)
```

#### 推理流程
```
Input Prompt → Tokenization → Embeddings
    ↓
Transformer Forward Pass (with KV-Cache)
    ↓
Output Projection → Sampling Strategy
    ├─ Greedy (argmax)
    ├─ Top-K Sampling
    ├─ Top-P (Nucleus) Sampling
    └─ Beam Search
    ↓
Generated Token → 追加到输入 → 循环直到 </s> 或 max_length
```

### 4. 配置规范 (lib.rs)

```rust
// 全局模型配置（教育友好的参数规模）
pub const MAX_SEQ_LEN: usize = 128;      // 序列最大长度
pub const EMBEDDING_DIM: usize = 256;    // 嵌入维度（v0.3.1 降低以适配小数据集）
pub const HIDDEN_DIM: usize = 512;       // 前馈隐藏层维度
pub const NUM_HEADS: usize = 8;          // 注意力头数量
pub const NUM_LAYERS: usize = 2;         // Transformer 层数
pub const VOCAB_SIZE: usize = 30000;     // 词汇表目标大小
pub const DROPOUT_RATE: f32 = 0.1;       // Dropout 比率
```

**调参原则**:
- 小数据集（<1000 样本）→ 小模型（当前配置约 10M 参数）
- EMBEDDING_DIM 必须能被 NUM_HEADS 整除
- 训练数据增加时，可按比例增大 HIDDEN_DIM 和 EMBEDDING_DIM

---

## 📝 代码规范

### 1. 命名规范

#### 文件命名
- **全小写 + 下划线**: `self_attention.rs`, `feed_forward.rs`
- **单一职责**: 每个文件实现一个主要组件

#### 变量命名
```rust
// ✅ 推荐：描述性变量名
let token_embeddings = self.token_embed.forward(&token_ids);
let attention_output = self.attention.forward(&normalized_input);
let learning_rate = 0.001;

// ❌ 避免：单字母变量（除了数学公式中的惯例）
let x = input;  // 除非是通用的 input 变量
let w = weights; // 应该用 weight_matrix
```

#### 函数命名
```rust
// ✅ 动词开头，清晰表达意图
fn build_vocabulary_from_texts() -> Vocab { ... }
fn compute_attention_scores() -> Array2<f32> { ... }
fn apply_layer_normalization() -> Array2<f32> { ... }

// ❌ 避免：模糊的命名
fn process() { ... }
fn handle() { ... }
```

### 2. 注释规范

#### 文件头注释（必须）
```rust
//! # Self-Attention Module
//!
//! 实现多头自注意力机制（Multi-Head Self-Attention），Transformer 架构的核心组件。
//!
//! ## 算法原理
//! 1. 线性投影：Q = XW_q, K = XW_k, V = XW_v
//! 2. 缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / √d_k)V
//! 3. 多头拼接：MultiHead = Concat(head_1, ..., head_h)W_o
//!
//! ## KV-Cache 优化
//! 推理时缓存 Key/Value 矩阵，避免对历史 token 的重复计算。
```

#### 函数注释（复杂函数必须）
```rust
/// 计算缩放点积注意力分数
///
/// # 参数
/// - `query`: 查询矩阵 (seq_len, d_k)
/// - `key`: 键矩阵 (seq_len, d_k)
/// - `d_k`: 键的维度（用于缩放）
///
/// # 返回
/// 注意力权重矩阵 (seq_len, seq_len)，每行和为 1
///
/// # 算法
/// ```
/// scores = (Q @ K^T) / sqrt(d_k)
/// attention_weights = softmax(scores)
/// ```
fn compute_scaled_dot_product_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    d_k: f32,
) -> Array2<f32> {
    // 实现代码
}
```

#### 行内注释（关键步骤必须）
```rust
// 计算注意力分数：Q @ K^T
let scores = query.dot(&key.t()) / (self.d_k as f32).sqrt();

// Softmax 归一化（数值稳定性处理）
let scores_max = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
let exp_scores = (scores - scores_max).mapv(f32::exp);

// 反向传播：梯度分配到多个头
for h in 0..self.num_heads {
    let head_grad = grad_concat.slice(s![.., h * head_dim..(h + 1) * head_dim]);
    // ...
}
```

### 3. 错误处理规范

```rust
// ✅ 使用 Result 类型处理可能失败的操作
pub fn load_model(path: &str) -> Result<LLM, String> {
    let file = File::open(path)
        .map_err(|e| format!("无法打开模型文件 {}: {}", path, e))?;
    // ...
}

// ✅ Panic 前提供清晰的错误信息
assert_eq!(
    query.ncols(), self.d_model,
    "Query 维度 {} 与期望的 d_model {} 不匹配",
    query.ncols(), self.d_model
);

// ❌ 避免：静默失败或不明确的 unwrap()
let result = some_operation().unwrap(); // 不好
```

### 4. 测试规范

#### 测试文件组织
```
tests/
├── llm_test.rs              # LLM 集成测试
├── transformer_test.rs      # Transformer Block 测试
├── self_attention_test.rs   # 自注意力单元测试
├── feed_forward_test.rs     # 前馈网络测试
├── embeddings_test.rs       # 嵌入层测试
├── vocab_test.rs            # 词汇表测试
└── chinese_tests.rs         # 中文处理测试
```

#### 测试命名和结构
```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 测试：前向传播输出形状正确
    #[test]
    fn test_forward_output_shape() {
        let layer = FeedForward::new(256, 512);
        let input = Array2::zeros((2, 256)); // (batch_size=2, dim=256)
        let output = layer.forward(&input);
        
        assert_eq!(output.shape(), &[2, 256], "输出形状应为 (2, 256)");
    }

    /// 测试：反向传播后参数发生变化
    #[test]
    fn test_backward_updates_parameters() {
        let mut layer = FeedForward::new(256, 512);
        let input = Array2::ones((2, 256));
        
        // 前向传播
        let output = layer.forward(&input);
        
        // 反向传播
        let grad_output = Array2::ones(output.dim());
        let lr = 0.01;
        layer.backward(&grad_output, lr);
        
        // 验证参数已更新（具体验证逻辑）
        // ...
    }

    /// 测试：中文文本分词正确性
    #[test]
    fn test_chinese_tokenization() {
        let vocab = Vocab::new();
        let text = "深度学习很有趣";
        let tokens = vocab.tokenize(text);
        
        assert!(!tokens.is_empty(), "分词结果不应为空");
        assert!(tokens.len() >= 3, "应至少分出 3 个词");
    }
}
```

#### 测试覆盖目标
- ✅ **单元测试**: 每个 Layer 的 forward/backward
- ✅ **集成测试**: 完整的训练流程（小数据集）
- ✅ **边界测试**: 空输入、单 token、最大长度
- ✅ **中文特化测试**: 成语检测、标点处理、混合文本

---

## 🔄 开发工作流程

### 1. 分支策略

```
main (主分支)
    ├─ v0.4.0 (当前稳定版本)
    │
    ├─ feature/new-layer (功能分支)
    │   └─ 添加新的神经网络层
    │
    ├─ fix/attention-bug (修复分支)
    │   └─ 修复自注意力计算错误
    │
    └─ perf/blas-optimization (性能分支)
        └─ BLAS 加速优化
```

**分支命名规范**:
- `feature/描述` - 新功能开发
- `fix/描述` - Bug 修复
- `perf/描述` - 性能优化
- `docs/描述` - 文档更新

### 2. 提交信息规范

```
类型(范围): 简短描述（50字符以内）

详细说明（可选，72字符换行）

- 关键变更点 1
- 关键变更点 2

Refs: #issue号（如果相关）
```

**类型标签**:
- `feat`: 新功能
- `fix`: Bug 修复
- `perf`: 性能优化
- `refactor`: 代码重构（不改变功能）
- `docs`: 文档更新
- `test`: 测试相关
- `chore`: 构建/工具链更新

**示例**:
```
feat(attention): 添加 KV-Cache 支持推理加速

实现 Key/Value 缓存机制，避免自回归生成时的重复计算。

- 新增 enable_kv_cache() 和 clear_kv_cache() 方法
- 推理速度提升约 3-5 倍（长序列）
- 保持训练模式下的原有行为

Refs: #42
```

### 3. 开发流程

#### 添加新功能（以新增 Layer 为例）

```bash
# 1. 创建功能分支
git checkout -b feature/position-bias-layer

# 2. 实现新层（src/position_bias.rs）
# 3. 实现 Layer trait
# 4. 添加单元测试（tests/position_bias_test.rs）
# 5. 更新文档（CLAUDE.md, README.md）

# 6. 运行测试
cargo test --test position_bias_test
cargo test  # 确保不破坏现有功能

# 7. 代码格式化和静态检查
cargo fmt
cargo clippy

# 8. 提交代码
git add src/position_bias.rs tests/position_bias_test.rs
git commit -m "feat(layer): 添加位置偏置层

实现 T5 风格的相对位置偏置机制。

- 支持可学习的位置偏置参数
- 集成到 Self-Attention 计算中
- 通过 128 个 bucket 的相对位置编码"

# 9. 合并到主分支（通过 PR）
git push origin feature/position-bias-layer
# 在 GitHub 上创建 Pull Request
```

#### 修复 Bug

```bash
# 1. 创建修复分支
git checkout -b fix/softmax-numerical-stability

# 2. 编写失败的测试用例（重现 Bug）
# 3. 修复代码
# 4. 验证测试通过

# 5. 提交
git commit -m "fix(attention): 修复 Softmax 数值稳定性问题

在 Softmax 计算中减去最大值以防止溢出。

- 修复大数值输入导致的 NaN 问题
- 添加边界测试用例"

# 6. 合并到主分支
```

### 4. 代码审查 Checklist

#### 功能正确性
- [ ] 实现符合设计文档
- [ ] 所有测试通过 (`cargo test`)
- [ ] 边界情况已测试（空输入、极大值、极小值）

#### 代码质量
- [ ] 遵循命名规范
- [ ] 关键函数有注释
- [ ] 无 Clippy 警告 (`cargo clippy`)
- [ ] 代码已格式化 (`cargo fmt`)

#### 性能和资源
- [ ] 无不必要的内存分配
- [ ] 循环内避免重复计算
- [ ] 大矩阵操作使用 ndarray 高效方法

#### 文档和测试
- [ ] 更新了 CLAUDE.md（如果架构有变化）
- [ ] 更新了 README.md（如果用户接口有变化）
- [ ] 添加了相应的单元测试

---

## 📚 依赖管理原则

### 当前依赖清单（v0.4.0）

```toml
[dependencies]
# 核心数值计算（教育项目的唯一必需依赖）
ndarray = "0.16.1"              # 多维数组和张量操作

# 可选：BLAS 加速（不影响教育目标）
blas-src = { version = "0.10", features = ["openblas"], optional = true }
openblas-src = { version = "0.10", features = ["cblas", "system"], optional = true }

# 中文处理（教育重点）
jieba-rs = "0.7"                # 中文分词

# 性能优化（可教学）
lru = "0.12"                    # LRU 缓存（tokenizer 优化）

# 基础工具（最小必需）
rand = "0.9.2"                  # 随机数生成
regex = "1.10.0"                # 正则表达式（成语检测）
serde = { version = "1.0", features = ["derive"] }  # 序列化
serde_json = "1.0"              # JSON 数据加载
bincode = "2.0.1"               # 二进制序列化
chrono = "0.4"                  # 时间戳
log = "0.4"                     # 日志
simple_logger = "4.3"           # 简单日志实现
```

### 添加新依赖的判断标准

**必须满足以下所有条件才能添加新依赖**:

1. ✅ **教育价值**: 依赖本身或其替代实现是否有教学意义？
   - 例如: `ndarray` ✅（张量操作是核心）
   - 例如: `pytorch-rs` ❌（隐藏实现细节）

2. ✅ **无法简单实现**: 自己实现是否会偏离教学重点？
   - 例如: `jieba-rs` ✅（中文分词是专门领域）
   - 例如: `regex` ✅（正则引擎实现复杂）
   - 例如: `csv` ❌（可以自己实现简单的 CSV 解析）

3. ✅ **维护活跃**: 依赖是否长期维护且稳定？

4. ✅ **纯 Rust**: 避免 C/C++ 绑定（除了 BLAS 优化）

**添加流程**:
```bash
# 1. 在 Cargo.toml 中添加依赖（注明原因）
[dependencies]
# 新依赖：用于 XXX 功能，因为 YYY 原因
new-crate = "1.0"

# 2. 在 CLAUDE.md 中更新依赖说明
# 3. 在代码中添加使用示例和注释
# 4. 更新 README.md 的依赖列表
```

---

## 🧪 测试策略

### 测试金字塔

```
       /\
      /  \  End-to-End Tests (少量)
     /────\  - 完整训练流程
    /      \ - 推理生成测试
   /────────\
  / Integration Tests (适中)
 /────────────\  - 多层组合测试
/  Unit Tests  \  - 每个 Layer 的详细测试
────────────────
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试文件
cargo test --test llm_test
cargo test --test self_attention_test

# 运行特定测试函数
cargo test test_forward_output_shape

# 显示测试输出（包括 println!）
cargo test -- --nocapture

# 并行测试（默认）
cargo test -- --test-threads=4

# 顺序测试（调试时有用）
cargo test -- --test-threads=1
```

### 性能基准测试

```bash
# 运行所有基准测试
cargo bench

# 运行特定基准测试
cargo bench --bench performance_benchmark
cargo bench --bench memory_optimization_bench
```

### 测试数据

- **小规模测试数据**: 硬编码在测试文件中（10-50 个样本）
- **集成测试数据**: 使用 `data/` 目录的子集（100 个样本）
- **基准测试数据**: 生成随机数据（可控维度和大小）

---

## 📖 文档维护规范

### 文档层次

1. **CLAUDE.md** (开发指南)
   - 面向 AI 辅助开发
   - 架构设计、数据流、开发模式
   - **更新时机**: 架构变更、新增模块

2. **README.md / README_zh.md** (用户指南)
   - 面向终端用户
   - 快速开始、功能介绍、示例
   - **更新时机**: 命令行接口变更、新功能

3. **IMPLEMENTATION_v0.X.md** (版本实现笔记)
   - 记录特定版本的设计决策
   - 性能优化细节
   - **更新时机**: 重大版本发布

4. **PERFORMANCE_OPTIMIZATIONS.md** (性能优化)
   - BLAS 集成、缓存策略、算子融合
   - **更新时机**: 性能改进

5. **BATCH_TRAINING.md** (训练指南)
   - 数据准备、训练技巧
   - **更新时机**: 训练流程变更

### 代码内文档

```rust
// 使用 Rust 文档注释（cargo doc 可生成 HTML）

//! 模块级文档（文件开头）
//! 
//! # 模块名称
//! 简短描述
//! 
//! ## 主要组件
//! - Component 1
//! - Component 2

/// 函数级文档
///
/// # 参数
/// - `param1`: 描述
///
/// # 返回
/// 描述返回值
///
/// # 示例
/// ```
/// let result = function(arg);
/// ```
pub fn function(param1: Type) -> ReturnType { ... }
```

### 生成文档

```bash
# 生成 HTML 文档
cargo doc --open

# 包含私有项
cargo doc --document-private-items

# 生成并检查所有链接
cargo doc --no-deps
```

---

## 🚀 发布流程

### 版本号规范 (Semantic Versioning)

```
v主版本.次版本.修订号

例如: v0.4.0
- 主版本: 不兼容的架构变更（目前为 0，教育项目）
- 次版本: 新功能、性能优化（保持兼容）
- 修订号: Bug 修复、文档更新
```

### 发布 Checklist

```bash
# 1. 确保所有测试通过
cargo test
cargo clippy
cargo fmt --check

# 2. 更新版本号
# - Cargo.toml: version = "0.5.0"
# - 更新 CLAUDE.md 中的版本引用

# 3. 更新 CHANGELOG.md（如果有）
# 4. 创建实现笔记（重大版本）
# cp IMPLEMENTATION_v0.4.0.md IMPLEMENTATION_v0.5.0.md

# 5. 提交版本标签
git add -A
git commit -m "chore: bump version to v0.5.0"
git tag -a v0.5.0 -m "Release v0.5.0: 新功能描述"
git push origin main --tags

# 6. 生成 Release Notes（GitHub Releases）
# 7. 编译发布版本
cargo build --release

# 8. （可选）发布到 crates.io
# cargo publish  # 仅当项目成熟且有外部用户需求时
```

---

## 🎯 贡献指南

### 如何贡献

1. **报告 Bug**
   - 在 GitHub Issues 中描述问题
   - 提供复现步骤和错误信息
   - 标注受影响的模块

2. **提出新功能**
   - 先在 Issues 中讨论设计
   - 确保符合教育目标和最小依赖原则
   - 获得维护者认可后开始实现

3. **提交代码**
   - Fork 仓库并创建功能分支
   - 遵循代码规范和测试要求
   - 提交 Pull Request 并关联 Issue

### 社区行为准则

- ✅ 尊重教育优先的项目定位
- ✅ 提交前充分测试
- ✅ 注释使用中文（面向中文学习者）
- ✅ 欢迎提问和讨论架构设计

---

## 🔧 工具链配置

### 推荐 IDE 配置

#### VS Code
```json
// .vscode/settings.json
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": ["blas"],  // 可选
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

#### IntelliJ IDEA (Rust Plugin)
- 启用 "Run Clippy on save"
- 设置 "Max line length" = 100

### rustfmt 配置

```toml
# rustfmt.toml
edition = "2024"
max_width = 100
tab_spaces = 4
use_small_heuristics = "Default"
```

### Clippy 配置

```toml
# .cargo/config.toml (可选)
[target.'cfg(all())']
rustflags = [
    "-W", "clippy::all",
    "-W", "clippy::pedantic",
    "-A", "clippy::module_name_repetitions",  # 教育代码允许重复
    "-A", "clippy::too_many_arguments",       # 某些层需要多参数
]
```

---

## 📊 性能优化指南

### 优化优先级（教育项目）

1. **算法复杂度** (最高优先级)
   - 避免不必要的 O(n²) 操作
   - 使用高效的 ndarray 方法（`.dot()` 而非手写循环）

2. **内存效率**
   - 缓存可重用的中间结果（如位置编码）
   - 避免不必要的 `.clone()`

3. **并行计算** (可选，不影响教学)
   - ndarray 的 rayon 支持（默认启用）
   - BLAS 加速（可选特性）

4. **避免过度优化**
   - ❌ 不使用 unsafe 代码（教育项目）
   - ❌ 不引入复杂的底层优化（除非有教学价值）

### Profiling 工具

```bash
# CPU 性能分析（Linux）
cargo install flamegraph
cargo flamegraph --bin llm

# 内存分析
valgrind --tool=massif target/release/llm

# 基准测试对比
cargo bench --bench performance_benchmark > baseline.txt
# (做出修改)
cargo bench --bench performance_benchmark > optimized.txt
diff baseline.txt optimized.txt
```

---

## ✅ 质量保证

### CI/CD 流程 (GitHub Actions)

```yaml
# .github/workflows/check.yml
name: Code Quality Check
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: clippy, rustfmt
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings
      - run: cargo test
```

### Pre-commit 钩子（可选）

```bash
# .git/hooks/pre-commit (chmod +x)
#!/bin/bash
cargo fmt --check || exit 1
cargo clippy -- -D warnings || exit 1
cargo test --quiet || exit 1
```

---

## 📞 联系和支持

### 问题反馈
- **Bug 报告**: GitHub Issues
- **功能讨论**: GitHub Discussions
- **紧急问题**: 项目维护者邮件（如果提供）

### 参考资源
- [Attention Is All You Need (论文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Rust ndarray 文档](https://docs.rs/ndarray/)
- [Jieba 中文分词](https://github.com/messense/jieba-rs)

---

## 📅 项目路线图

### v0.4.0 (当前版本)
- ✅ BLAS 加速支持
- ✅ Tokenizer LRU 缓存
- ✅ 融合算子优化

### v0.5.0 (计划中)
- 🔲 批量训练支持
- 🔲 更多采样策略（温度退火、Top-K 改进）
- 🔲 可视化训练曲线

### v1.0.0 (长期目标)
- 🔲 完整的教学文档和教程
- 🔲 交互式 Jupyter Notebook 示例
- 🔲 更大规模的预训练数据集

---

## 📄 许可证

本项目采用 MIT 许可证（或您选择的许可证），详见 [LICENSE.txt](../LICENSE.txt)。

**教育性使用鼓励**:
- ✅ 课程教学和作业
- ✅ 学术研究和论文引用
- ✅ 个人学习和实验

---

*本规范文档最后更新时间: 2024-10-25*  
*对应项目版本: v0.4.0*
