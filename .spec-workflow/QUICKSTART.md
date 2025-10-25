# 快速开始指南 - RustGPT-Chinese Spec Workflow

## 🚀 5 分钟快速了解

### 这是什么项目？
RustGPT-Chinese 是一个**教育性质**的从零实现的中文 Transformer 语言模型，使用纯 Rust 编写。

**核心理念**: 代码清晰度 > 性能优化，让学习者理解 LLM 的内部工作原理。

---

## 📚 文档导航

### 1️⃣ 我想快速上手 → [README.md](./README.md)
- 文档总览和快速导航
- 新开发者上手步骤（5 步走）

### 2️⃣ 我想开始贡献代码 → [SPEC_WORKFLOW.md](./SPEC_WORKFLOW.md)
- **最重要的文档**！包含：
  - 架构规范（模块层次、Layer Trait）
  - 代码规范（命名、注释、测试）
  - 开发工作流程（分支、提交、审查）
  - 依赖管理原则

### 3️⃣ 我想理解技术选型 → [TECH_STACK.md](./TECH_STACK.md)
- 为什么选择 Rust？
- 为什么用 ndarray 而非 PyTorch？
- 架构模式和数据流
- 性能基准和优化策略

### 4️⃣ 我想找到某个功能的代码 → [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)
- 完整的目录结构和文件说明
- 每个模块的详细职责
- 快速导航表格
- 代码度量和复杂度分析

### 5️⃣ 我想写出教育友好的代码 → [EDUCATIONAL_GUIDELINES.md](./EDUCATIONAL_GUIDELINES.md)
- 注释原则和最佳实践
- 数学公式实现规范
- 测试作为文档
- 代码审查 Checklist
- 完整的 Dropout 层实现示例

---

## ⚡ 最常用命令

```bash
# 运行训练和推理
cargo run --release

# 运行所有测试
cargo test

# 运行特定测试
cargo test --test llm_test

# 代码格式化
cargo fmt

# 代码检查
cargo clippy

# 生成文档
cargo doc --open

# BLAS 加速（可选，需安装 OpenBLAS）
cargo build --features blas --release
```

---

## 🎯 关键概念速查

### Layer Trait（所有神经网络层的统一接口）
```rust
pub trait Layer: Send + Sync {
    fn layer_type(&self) -> &str;
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
    fn parameters(&self) -> usize;
    fn set_training_mode(&mut self, training: bool);
}
```

### 模型配置（lib.rs）
```rust
MAX_SEQ_LEN: 128       // 序列最大长度
EMBEDDING_DIM: 256     // 嵌入维度（小模型）
HIDDEN_DIM: 512        // 前馈隐藏层维度
NUM_HEADS: 8           // 注意力头数
NUM_LAYERS: 2          // Transformer 层数
DROPOUT_RATE: 0.1      // Dropout 比率
```

### 数据流（一图胜千言）
```
用户输入
   ↓
Jieba 分词（LRU 缓存）
   ↓
Token IDs
   ↓
Embeddings (256d + 位置编码)
   ↓
TransformerBlock 1
  ├─ LayerNorm → Attention → Dropout → Residual
  └─ LayerNorm → FeedForward → Dropout → Residual
   ↓
TransformerBlock 2 (同上)
   ↓
OutputProjection (vocab_size)
   ↓
Sampling (Greedy/Top-K/Top-P/Beam)
   ↓
生成的中文文本
```

---

## 🔍 快速定位功能

| 我想... | 查看文件 |
|---------|---------|
| 理解整体架构 | `lib.rs`, `llm.rs` |
| 看注意力机制实现 | `self_attention.rs` |
| 看反向传播如何工作 | `llm.rs` (backward 方法) |
| 理解中文分词 | `vocab.rs` |
| 看训练流程 | `main.rs` |
| 添加新的神经网络层 | 参考 `feed_forward.rs` 并实现 Layer trait |
| 修改模型超参数 | `lib.rs` |
| 理解 GELU 激活函数 | `feed_forward.rs` |
| 看梯度裁剪实现 | `llm.rs` (train_monitored 方法) |
| 理解 KV-Cache 优化 | `self_attention.rs` |

---

## 📝 代码规范速查

### ✅ 好的例子
```rust
/// 计算缩放点积注意力
///
/// # 算法
/// ```
/// Attention(Q, K, V) = softmax(Q K^T / √d_k) V
/// ```
fn scaled_dot_product_attention(
    query: &Array2<f32>,  // Q
    key: &Array2<f32>,    // K
    value: &Array2<f32>,  // V
) -> Array2<f32> {
    // 1. 计算注意力分数
    let d_k = (key.ncols() as f32).sqrt();
    let scores = query.dot(&key.t()) / d_k;
    
    // 2. Softmax 归一化（数值稳定性处理）
    let max_score = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores = (scores - max_score).mapv(f32::exp);
    
    // 3. 加权求和
    let attention_weights = softmax(&exp_scores);
    attention_weights.dot(value)
}
```

### ❌ 不好的例子
```rust
// 没有注释，单字母变量，不清晰
fn attn(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let s = q.dot(&k.t()) / (k.ncols() as f32).sqrt();
    let a = softmax(&s);
    a.dot(v)
}
```

---

## 🧪 测试速查

```bash
# 运行所有测试
cargo test

# 运行特定组件测试
cargo test --test self_attention_test
cargo test --test feed_forward_test
cargo test --test vocab_test

# 显示测试输出（包括 println!）
cargo test -- --nocapture

# 运行性能基准测试
cargo bench
```

---

## 📊 项目统计（v0.4.0）

- **代码行数**: ~9,300 行
- **测试文件**: 11 个
- **核心依赖**: 6 个（ndarray, jieba-rs, rand, serde, regex, lru）
- **模型参数**: ~10M（教育友好的小规模）
- **训练时间**: 15-20 分钟（500 样本，10 epochs，CPU）
- **推理速度**: 50-80ms/token（with KV-Cache）

---

## 🎓 学习路径建议

### 新手入门（按顺序阅读）
1. `lib.rs` - 理解全局配置和 Layer trait
2. `main.rs` - 理解整体训练流程
3. `vocab.rs` - 理解中文分词
4. `embeddings.rs` - 第一个简单的层
5. `feed_forward.rs` - 理解前馈网络
6. `self_attention.rs` - 核心：注意力机制
7. `transformer.rs` - 理解层的组合
8. `llm.rs` - 理解前向/反向传播

### 进阶实验
1. 修改 `lib.rs` 中的超参数，观察训练效果
2. 将 GELU 替换为 ReLU，对比收敛速度
3. 实现 Batch Normalization 层
4. 可视化注意力权重矩阵

---

## 🤔 常见问题

### Q: 为什么不用 PyTorch？
**A**: 教育目标是展示底层实现。PyTorch 隐藏了反向传播的细节，而我们希望学习者理解梯度是如何计算的。

### Q: 模型能达到 GPT-3 的水平吗？
**A**: 不能，也不是目标。这是一个教育项目（10M 参数 vs GPT-3 的 175B 参数）。重点是理解原理，而非生产性能。

### Q: 可以用英文训练吗？
**A**: 可以，但分词效果不佳（jieba-rs 专门为中文设计）。未来版本可能添加多语言支持。

### Q: 如何添加新依赖？
**A**: 参考 [SPEC_WORKFLOW.md](./SPEC_WORKFLOW.md) 的"依赖管理原则"。只添加无法简单实现的依赖，并更新相关文档。

### Q: 代码中没有自动微分，怎么训练？
**A**: 手动推导梯度并实现反向传播。这正是教育价值所在——理解链式法则如何应用于神经网络。

---

## 📞 获取帮助

- **文档问题**: 查看 [README.md](./README.md) 的文档导航
- **代码问题**: 查看 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) 找到相关模块
- **规范问题**: 查看 [SPEC_WORKFLOW.md](./SPEC_WORKFLOW.md) 或 [EDUCATIONAL_GUIDELINES.md](./EDUCATIONAL_GUIDELINES.md)
- **Bug 报告**: 提交 GitHub Issue

---

## 🎉 开始贡献

1. **Fork 仓库**
2. **阅读** [SPEC_WORKFLOW.md](./SPEC_WORKFLOW.md)
3. **创建功能分支**: `git checkout -b feature/your-feature`
4. **遵循代码规范**（参考 [EDUCATIONAL_GUIDELINES.md](./EDUCATIONAL_GUIDELINES.md)）
5. **提交 Pull Request**

---

*祝学习愉快！如果您觉得项目有用，请给我们一个 ⭐*

---

**文档版本**: v0.4.0  
**最后更新**: 2024-10-25  
**维护者**: RustGPT-Chinese 项目组
