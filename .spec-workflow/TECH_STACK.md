# 技术栈文档 - RustGPT-Chinese

## 项目类型

**教育性深度学习框架** - 从零实现的中文 Transformer 语言模型

- **目标**: 展示 LLM 内部工作原理，用于教学和学习
- **类型**: 命令行工具 (CLI) + 库 (Library)
- **部署**: 本地运行，单机训练和推理
- **用户**: 深度学习学习者、研究人员、中文 NLP 开发者

---

## 核心技术

### 主要编程语言

- **语言**: Rust (Edition 2024)
  - **版本要求**: rustc 1.80.0+ (支持 2024 edition)
  - **选择理由**:
    - ✅ 内存安全无 GC（适合数值计算）
    - ✅ 零成本抽象（性能接近 C++）
    - ✅ 所有权系统防止数据竞争
    - ✅ 教育友好的类型系统和错误提示
  - **编译器**: `rustc`
  - **包管理器**: `cargo`

### 核心依赖库

#### 1. 数值计算和张量操作

```toml
ndarray = "0.16.1"
```
- **用途**: 多维数组（张量）计算，项目的核心依赖
- **功能**:
  - 矩阵乘法（`.dot()`）
  - 元素级运算（`.mapv()`, `+`, `*` 等）
  - 广播和形状变换
  - 并行计算（通过 rayon）
- **为什么不用 PyTorch/TensorFlow**: 教育目标是展示底层实现，而非隐藏细节

#### 2. BLAS 加速（可选）

```toml
# 可选特性，需系统安装 OpenBLAS
blas-src = { version = "0.10", features = ["openblas"], optional = true }
openblas-src = { version = "0.10", features = ["cblas", "system"], optional = true }
```
- **用途**: 加速 ndarray 的矩阵运算（30-50% 提升）
- **启用方式**: `cargo build --features blas`
- **系统要求**:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libopenblas-dev
  
  # macOS
  brew install openblas
  ```
- **教育考虑**: 可选特性，不影响代码可读性

#### 3. 中文自然语言处理

```toml
jieba-rs = "0.7"
regex = "1.10.0"
```
- **jieba-rs**: 中文分词（基于 HMM 和词典）
  - 用途: 将中文句子切分为词语
  - 示例: "深度学习" → ["深度", "学习"]
- **regex**: 正则表达式
  - 用途: 中文成语检测（四字模式）
  - 示例: 匹配"一帆风顺"等四字成语

#### 4. 序列化和数据加载

```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "2.0.1"
```
- **serde**: Rust 序列化框架（核心）
- **serde_json**: 加载训练数据（`data/pretraining_data.json`）
- **bincode**: 二进制模型序列化（`.bin` 文件）
  - 优点: 文件小、加载快
  - 用于: 模型检查点保存和恢复

#### 5. 性能优化

```toml
lru = "0.12"
```
- **用途**: LRU 缓存（Least Recently Used）
- **应用场景**: 缓存 jieba 分词结果
- **容量**: 10,000 条目
- **效果**: 重复文本加速 5-10 倍

#### 6. 基础工具

```toml
rand = "0.9.2"          # 随机数生成（权重初始化、Dropout）
chrono = "0.4"          # 时间戳（训练日志）
log = "0.4"             # 日志接口
simple_logger = "4.3"   # 简单日志实现
```

---

## 应用架构

### 架构模式

**分层神经网络架构** + **模块化组件设计**

```
┌──────────────────────────────────────────────────────────┐
│                     主程序 (main.rs)                      │
│  - 命令行交互                                             │
│  - 训练流程编排（预训练 → 指令微调 → 交互推理）            │
└────────────────────────┬─────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼──────────┐         ┌─────────▼──────────┐
│  LLM 核心类       │         │  支持模块           │
│  (llm.rs)         │         │  - Vocab           │
│  - 前向传播       │◄────────┤  - DatasetLoader   │
│  - 反向传播       │         │  - CheckpointMgr   │
│  - 生成采样       │         │  - PerformanceMon  │
└────────┬──────────┘         └────────────────────┘
         │
         │ 调用 Layer trait
         │
┌────────▼────────────────────────────────────────┐
│              神经网络层 (Layer trait)            │
├─────────────────────────────────────────────────┤
│  Embeddings → TransformerBlock (x2) → Output    │
│     ↓              ↓                      ↓     │
│  Token + Pos   Attention + FFN       Projection │
└─────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────┐
│            基础算子 (ndarray + 优化)             │
│  - 矩阵乘法 (BLAS)                              │
│  - Softmax / LayerNorm                          │
│  - Dropout / GELU                               │
│  - 融合算子 (FusedLayerNormLinear)              │
└─────────────────────────────────────────────────┘
```

### 数据流向

#### 训练流程
```
JSON 训练数据
    ↓
Vocab 构建（Jieba 分词 + 特殊 token）
    ↓
Token ID 序列
    ↓
DataLoader（批量加载 + 缓存）
    ↓
LLM Forward Pass（Teacher Forcing）
    ↓
Cross-Entropy Loss
    ↓
LLM Backward Pass（链式求导）
    ↓
Adam Optimizer（参数更新）
    ↓
CheckpointManager（定期保存）
```

#### 推理流程
```
用户输入文本
    ↓
Jieba 分词（带 LRU 缓存）
    ↓
Token ID 序列
    ↓
LLM Forward Pass（with KV-Cache）
    ↓
Sampling（Greedy / Top-K / Top-P / Beam）
    ↓
生成 Token → 追加到输入 → 循环
    ↓
后处理（去除中文空格）
    ↓
输出中文句子
```

---

## 数据存储

### 主要存储方式

**本地文件系统** - 所有数据存储在项目目录

#### 1. 训练数据
- **格式**: JSON 数组
- **位置**: `data/pretraining_data.json`, `data/chat_training_data.json`
- **结构**:
  ```json
  [
      "中文句子 1",
      "中文句子 2",
      ...
  ]
  ```
- **特点**: 纯文本，易于编辑和扩展

#### 2. 词汇表数据
- **格式**: 内存中的 HashMap (token → ID, ID → token)
- **来源**: 动态构建自训练数据
- **大小**: 目标 30,000 个 token（实际根据数据量）
- **特殊 token**:
  ```rust
  <|pad|> = 0    // 填充
  <|unk|> = 1    // 未知词
  <|bos|> = 2    // 句子开始
  </s>    = 3    // 句子结束
  <|sep|> = 4    // 分隔符
  <|cls|> = 5    // 分类
  <|mask|> = 6   // 掩码
  ```

#### 3. 模型检查点
- **格式**: 二进制 (bincode) 或 JSON (serde_json)
- **位置**: `model_checkpoint.bin` / `model_checkpoint.json`
- **内容**:
  - 所有 Layer 的参数（权重矩阵）
  - Adam 优化器状态（momentum, velocity）
  - Vocab 映射表
  - 训练元数据（epoch, 最佳 loss）
- **大小**: 约 40-100 MB（取决于 vocab 大小）

#### 4. 成语词典
- **格式**: JSON 数组
- **位置**: `data/chinese_idioms.json`
- **用途**: 成语检测和特殊 token 标记
- **示例**:
  ```json
  ["一帆风顺", "马到成功", "心想事成"]
  ```

### 缓存机制

#### Tokenizer LRU 缓存
- **位置**: 内存（全局静态变量）
- **容量**: 10,000 条目
- **键**: 原始文本字符串
- **值**: 分词结果 (`Vec<String>`)
- **淘汰策略**: LRU (最近最少使用)

#### KV-Cache（推理时）
- **位置**: Self-Attention 层内存
- **内容**: Key 和 Value 矩阵（历史 token）
- **生命周期**: 单次生成会话
- **清除时机**: 遇到 `</s>` 或手动清除

---

## 外部集成

### 无外部 API 依赖

本项目**不依赖任何外部服务**，完全离线运行：
- ❌ 无云端 API（OpenAI, Hugging Face）
- ❌ 无数据库连接
- ❌ 无网络请求
- ✅ 完全本地计算和存储

### 协议和接口

#### 命令行接口 (CLI)
- **协议**: 标准输入/输出（stdin/stdout）
- **交互模式**: REPL（读取-求值-打印循环）
- **示例**:
  ```bash
  $ cargo run
  训练完成后...
  
  请输入问题（输入 'quit' 退出）: 深度学习是什么？
  模型回答: 深度学习是机器学习的一个分支...
  
  请输入问题（输入 'quit' 退出）: quit
  ```

#### 库接口 (Library API)
- **暴露**: `pub` 模块和类型
- **用途**: 允许其他 Rust 项目集成
- **示例**:
  ```rust
  use llm::{LLM, Vocab};
  
  let vocab = Vocab::build_from_texts(&texts);
  let mut model = LLM::new(vocab);
  model.train(&dataset, 10, 0.001);
  ```

---

## 开发环境

### 构建和开发工具

#### Cargo (Rust 包管理器)
```bash
# 构建开发版本（快速编译）
cargo build

# 构建发布版本（最大优化）
cargo build --release

# 运行程序
cargo run

# 运行测试
cargo test

# 性能基准测试
cargo bench

# 生成文档
cargo doc --open
```

#### 编译优化配置
```toml
[profile.release]
opt-level = 3           # 最高优化级别
lto = true              # 链接时优化（Link-Time Optimization）
codegen-units = 1       # 单一代码生成单元（更好的内联）
strip = true            # 移除调试符号（减小二进制大小）
```

### 代码质量工具

#### Rustfmt (代码格式化)
```bash
# 格式化所有代码
cargo fmt

# 检查格式（不修改）
cargo fmt --check

# 配置文件: rustfmt.toml
edition = "2024"
max_width = 100
tab_spaces = 4
```

#### Clippy (静态分析)
```bash
# 运行 Clippy
cargo clippy

# 作为错误对待所有警告
cargo clippy -- -D warnings

# 检查项:
# - 常见错误模式
# - 性能问题（不必要的 clone）
# - 惯用法建议（.iter() vs .into_iter()）
```

#### 测试框架
```bash
# 内置测试框架 (cargo test)
# - 单元测试: src/*.rs 中的 #[test] 函数
# - 集成测试: tests/*.rs 文件

# 运行所有测试
cargo test

# 运行特定测试
cargo test --test llm_test

# 显示 println! 输出
cargo test -- --nocapture
```

#### 文档生成 (rustdoc)
```bash
# 从代码注释生成 HTML 文档
cargo doc --open

# 包含私有项
cargo doc --document-private-items
```

### 版本控制

#### Git
- **分支策略**: Feature 分支工作流
- **提交规范**: Conventional Commits
  - `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `refactor:`
- **代码审查**: GitHub Pull Requests

#### GitHub Actions (CI/CD)
```yaml
# .github/workflows/check.yml
- 自动运行: cargo fmt --check
- 自动运行: cargo clippy
- 自动运行: cargo test
```

---

## 部署和分发

### 目标平台

**跨平台支持** - 任何支持 Rust 的系统：
- ✅ Linux (x86_64, aarch64)
- ✅ macOS (Intel, Apple Silicon)
- ✅ Windows (x86_64)

### 分发方式

#### 1. 源码编译（主要方式）
```bash
# 克隆仓库
git clone https://github.com/your-username/RustGPT-Chinese.git
cd RustGPT-Chinese

# 安装依赖（仅需 Rust 工具链）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 编译并运行
cargo run --release
```

#### 2. 预编译二进制（未来）
- GitHub Releases 提供各平台二进制文件
- 用户下载即可运行（约 5-10 MB）

#### 3. Crates.io（长期目标）
```bash
# 未来可能发布到 Rust 官方包索引
cargo install llm
```

### 安装要求

#### 最小系统要求
- **操作系统**: Linux / macOS / Windows
- **CPU**: 支持 x86_64 或 ARM64 的现代处理器
- **内存**: 4 GB RAM（推荐 8 GB）
- **磁盘**: 500 MB（代码 + 依赖 + 数据）
- **Rust**: 1.80.0+

#### 可选依赖
- **OpenBLAS**: 用于 BLAS 加速（`--features blas`）
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libopenblas-dev
  
  # macOS
  brew install openblas
  ```

### 更新机制

**手动更新** - 教育项目，无自动更新：
```bash
# 拉取最新代码
git pull origin main

# 重新编译
cargo build --release
```

---

## 技术要求和约束

### 性能要求

#### 训练性能
- **目标**: 小型数据集（500 样本）训练完成时间 < 30 分钟（CPU）
- **实际**: v0.4.0 约 15-20 分钟（Intel i7, 无 BLAS）
- **优化手段**:
  - 余弦退火学习率（加速收敛）
  - 早停机制（避免过度训练）
  - 梯度累积（稳定训练）

#### 推理性能
- **目标**: 单 token 生成延迟 < 100ms（CPU）
- **实际**: v0.4.0 约 50-80ms（with KV-Cache）
- **优化手段**:
  - KV-Cache（避免重复计算）
  - Tokenizer LRU 缓存
  - BLAS 加速（可选，提升 30-50%）

#### 内存占用
- **训练时**: 约 500 MB - 1 GB
- **推理时**: 约 200 MB - 500 MB
- **模型大小**: 40-100 MB（取决于 vocab）

### 兼容性要求

#### 平台支持
- **必须支持**: Linux x86_64, macOS x86_64/ARM64
- **应该支持**: Windows x86_64
- **可能支持**: 其他 Rust 支持的平台

#### 依赖版本
- **Rust Edition**: 2024（需要 rustc 1.80.0+）
- **ndarray**: 0.16.x（稳定版本）
- **jieba-rs**: 0.7.x（中文分词）
- **serde**: 1.0.x（序列化框架）

#### 标准合规
- **编码**: UTF-8（中文文本）
- **浮点数**: IEEE 754（f32）
- **并发**: Rust 所有权模型（无数据竞争）

### 安全和合规

#### 安全考虑
- ✅ **内存安全**: Rust 所有权系统保证无内存泄漏、无悬空指针
- ✅ **无 unsafe 代码**: 项目不使用 unsafe（除依赖库内部）
- ✅ **输入验证**: 用户输入长度限制（MAX_SEQ_LEN = 128）
- ⚠️ **无加密**: 模型文件未加密（教育项目）
- ⚠️ **无认证**: 无多用户支持（单机运行）

#### 数据隐私
- ✅ **本地运行**: 所有数据处理在本地
- ✅ **无遥测**: 不收集用户数据
- ✅ **无网络**: 不发送任何网络请求

#### 合规标准
- **许可证**: MIT License（开源）
- **数据使用**: 训练数据来自公开资源（需标注来源）
- **不适用**: GDPR, HIPAA（非生产系统）

### 可扩展性和可靠性

#### 预期负载
- **用户**: 单用户（本地运行）
- **数据量**: 500-5000 训练样本
- **并发**: 无并发需求（顺序处理）

#### 可用性
- **运行模式**: 交互式 CLI（前台运行）
- **容错**: 训练中断可通过检查点恢复
- **日志**: 简单日志输出到 stdout

#### 增长预测
- **v0.5.0**: 支持批量训练（处理更大数据集）
- **v1.0.0**: 支持分布式训练（多机并行，未来目标）

---

## 技术决策和理由

### 决策日志

#### 1. 选择 Rust 而非 Python
**理由**:
- ✅ 教育目标：展示底层实现（Python 易隐藏细节）
- ✅ 性能：无 GIL、零成本抽象、编译时优化
- ✅ 安全：类型系统和所有权防止错误
- ❌ 权衡：学习曲线较陡（但有教学价值）

#### 2. 使用 ndarray 而非 PyTorch/TensorFlow 绑定
**理由**:
- ✅ 从零实现：手写反向传播展示梯度计算
- ✅ 最小依赖：ndarray 只是数组库，无 ML 抽象
- ✅ 可读性：显式的矩阵操作易于理解
- ❌ 权衡：无自动微分（需手动求导）

#### 3. Pre-LN Transformer 架构（v0.2.0）
**理由**:
- ✅ 训练稳定性：LayerNorm 在 Attention 前（避免梯度爆炸）
- ✅ 现代实践：GPT-2/3、BERT 后期模型采用
- ✅ 收敛速度：比 Post-LN 快约 20%
- ❌ 权衡：稍复杂的架构（但教学价值高）

#### 4. 小模型规模（v0.3.1: 256d, 512d, 2层）
**理由**:
- ✅ 适配小数据集：500 样本无法训练大模型
- ✅ 训练时间：CPU 训练时间 < 30 分钟
- ✅ 教学友好：参数量适中（10M），易于理解
- ❌ 权衡：泛化能力有限（但符合教育定位）

#### 5. LRU 缓存 Tokenizer 结果（v0.4.0）
**理由**:
- ✅ 训练加速：重复文本分词加速 5-10x
- ✅ 教学价值：展示缓存优化策略
- ✅ 低开销：lru crate 仅 10 KB
- ❌ 权衡：内存占用增加（约 20-50 MB）

#### 6. 可选 BLAS 支持（v0.4.0）
**理由**:
- ✅ 性能提升：矩阵乘法加速 30-50%
- ✅ 可选特性：不强制依赖（保持纯 Rust 选项）
- ✅ 真实场景：展示生产级优化
- ❌ 权衡：系统依赖（需安装 OpenBLAS）

#### 7. 二进制 + JSON 双格式模型保存
**理由**:
- ✅ 二进制：快速加载和节省空间
- ✅ JSON：可读性和调试（可手动查看权重）
- ✅ 灵活性：用户可选择格式
- ❌ 权衡：需维护两套序列化代码

---

## 已知限制

### 当前限制

#### 1. 无批量处理（batch_size = 1）
- **影响**: 训练和推理速度较慢
- **原因**: 教育简化，避免复杂的批量张量操作
- **计划**: v0.5.0 添加批量支持

#### 2. 固定序列长度（MAX_SEQ_LEN = 128）
- **影响**: 无法处理长文本（超过 128 token）
- **原因**: 内存和计算限制（教育项目规模）
- **解决**: 未来支持滑动窗口或分块处理

#### 3. 无分布式训练
- **影响**: 只能在单机上训练
- **原因**: 增加复杂度，不适合教学
- **计划**: v1.0.0 后考虑（长期目标）

#### 4. 有限的采样策略
- **当前**: Greedy, Top-K, Top-P, Beam Search
- **缺失**: Temperature Annealing, Constrained Decoding
- **计划**: v0.5.0 添加更多策略

#### 5. 无自动微分
- **影响**: 添加新层需手动推导梯度
- **原因**: 从零实现，展示反向传播原理
- **解决**: 不计划添加（违背教育目标）

#### 6. 中文特化（英文支持有限）
- **影响**: 英文文本处理效果一般
- **原因**: 使用 jieba-rs（专门为中文设计）
- **解决**: 考虑 v0.6.0 添加多语言分词器

### 技术债务

#### 1. 重复的矩阵操作代码
- **位置**: self_attention.rs, feed_forward.rs
- **影响**: 可维护性
- **计划**: 提取通用函数到 utils.rs

#### 2. 硬编码的超参数
- **位置**: lib.rs 中的 const
- **影响**: 需重新编译才能修改
- **计划**: v0.5.0 添加配置文件（TOML）

#### 3. 有限的错误处理
- **位置**: 多处 panic! 和 unwrap()
- **影响**: 错误信息不够友好
- **计划**: 逐步替换为 Result<T, Error>

---

## 性能基准

### v0.4.0 性能数据（参考）

#### 训练性能（Intel i7-10700, 16GB RAM）
| 配置 | 500 样本 (10 epochs) | 1000 样本 (10 epochs) |
|------|---------------------|----------------------|
| 纯 Rust | 18 分钟 | 37 分钟 |
| BLAS 加速 | 13 分钟 | 26 分钟 |

#### 推理性能（同上配置）
| 操作 | 延迟 (ms) | 优化手段 |
|------|----------|---------|
| Tokenization | 5-10 | LRU 缓存 |
| Forward Pass (首次) | 80-100 | - |
| Forward Pass (KV-Cache) | 50-70 | KV-Cache |
| Beam Search (width=3) | 150-200 | - |

#### 内存占用
| 阶段 | 内存使用 |
|------|---------|
| 模型加载 | 150 MB |
| 训练中 | 800 MB |
| 推理中 | 300 MB |

---

## 参考资源

### 学术论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [BERT](https://arxiv.org/abs/1810.04805) - Pre-LN 架构参考
- [GPT-3](https://arxiv.org/abs/2005.14165) - 大规模语言模型

### 技术文档
- [Rust 官方文档](https://doc.rust-lang.org/)
- [ndarray 文档](https://docs.rs/ndarray/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### 社区资源
- [Rust 中文社区](https://rustcc.cn/)
- [Jieba 中文分词](https://github.com/messense/jieba-rs)
- [OpenBLAS](https://www.openblas.net/)

---

*最后更新: 2024-10-25 | 版本: v0.4.0*
