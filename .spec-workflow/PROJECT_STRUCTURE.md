# 项目结构文档 - RustGPT-Chinese

## 📁 目录结构总览

```
RustGPT-Chinese/
├── .github/                    # GitHub 配置
│   └── workflows/              # CI/CD 工作流
│       ├── check.yml           # 代码质量检查（fmt + clippy）
│       └── test.yml            # 自动化测试
│
├── .spec-workflow/             # 项目规范文档（本目录）
│   ├── SPEC_WORKFLOW.md        # 完整规范和工作流程
│   ├── TECH_STACK.md           # 技术栈详细说明
│   ├── PROJECT_STRUCTURE.md    # 本文档
│   ├── templates/              # Spec-workflow 模板
│   └── config.example.toml     # 配置示例
│
├── .claude/                    # Claude AI 辅助开发配置
│   └── output-styles/          # 输出风格定义
│
├── benches/                    # 性能基准测试
│   ├── memory_optimization_bench.rs  # 内存优化基准
│   └── performance_benchmark.rs      # 性能基准测试
│
├── data/                       # 训练数据和资源
│   ├── pretraining_data.json   # 预训练数据（中文知识）
│   ├── chat_training_data.json # 对话训练数据
│   └── chinese_idioms.json     # 中文成语词典
│
├── examples/                   # 使用示例
│   └── (未来添加教学示例)
│
├── src/                        # 源代码（核心实现）
│   ├── lib.rs                  # 库入口 + 全局配置
│   ├── main.rs                 # 命令行主程序
│   │
│   ├── llm.rs                  # LLM 核心类（前向/反向传播）
│   ├── vocab.rs                # 词汇表和分词
│   │
│   ├── embeddings.rs           # Token 嵌入层
│   ├── self_attention.rs       # 多头自注意力
│   ├── feed_forward.rs         # 前馈神经网络
│   ├── layer_norm.rs           # 层归一化
│   ├── dropout.rs              # Dropout 正则化
│   ├── output_projection.rs    # 输出投影层
│   ├── transformer.rs          # Transformer Block
│   │
│   ├── adam.rs                 # Adam 优化器
│   ├── training_optimizations.rs  # 训练优化（学习率、早停）
│   ├── batch_loader.rs         # 批量数据加载
│   ├── checkpoint_manager.rs   # 模型检查点管理
│   │
│   ├── fused_ops.rs            # 融合算子优化
│   ├── position_encoding.rs    # 位置编码
│   ├── performance_monitor.rs  # 性能监控
│   │
│   ├── dataset_loader.rs       # 数据加载工具
│   ├── model_serialization.rs  # 模型序列化
│   └── utils.rs                # 通用工具函数
│
├── tests/                      # 集成和单元测试
│   ├── llm_test.rs             # LLM 集成测试
│   ├── transformer_test.rs     # Transformer Block 测试
│   ├── self_attention_test.rs  # 自注意力测试
│   ├── feed_forward_test.rs    # 前馈网络测试
│   ├── embeddings_test.rs      # 嵌入层测试
│   ├── output_projection_test.rs  # 输出层测试
│   ├── adam_test.rs            # 优化器测试
│   ├── dataset_loader_test.rs  # 数据加载测试
│   ├── position_encoding_test.rs  # 位置编码测试
│   ├── vocab_test.rs           # 词汇表测试
│   └── chinese_tests.rs        # 中文处理测试
│
├── Cargo.toml                  # Rust 项目配置
├── Cargo.lock                  # 依赖锁定文件
├── rustfmt.toml                # 代码格式配置
├── .gitignore                  # Git 忽略规则
│
├── LICENSE.txt                 # MIT 开源许可证
├── README.md                   # 项目说明（英文）
├── README_zh.md                # 项目说明（中文）
├── CLAUDE.md                   # AI 开发指南
├── BATCH_TRAINING.md           # 批量训练文档
├── PERFORMANCE_OPTIMIZATIONS.md  # 性能优化说明
└── IMPLEMENTATION_v0.4.0.md    # 版本实现笔记
```

---

## 📄 核心文件详解

### 1. Cargo.toml - 项目清单

```toml
[package]
name = "llm"              # 包名
version = "0.4.0"         # 当前版本
edition = "2024"          # Rust 2024 版本

[dependencies]
# 核心依赖（详见 TECH_STACK.md）
ndarray = "0.16.1"        # 张量计算
jieba-rs = "0.7"          # 中文分词
lru = "0.12"              # LRU 缓存
# ... (其他依赖)

[features]
default = []              # 默认无特殊特性
blas = ["dep:blas-src", "dep:openblas-src", "ndarray/blas"]

[profile.release]
opt-level = 3             # 最高优化
lto = true                # 链接时优化
codegen-units = 1         # 单代码生成单元
strip = true              # 移除调试符号
```

**关键配置项**:
- `edition = "2024"` - 使用最新 Rust 特性
- `features` - BLAS 加速为可选特性
- `[lib]` 和 `[[bin]]` - 支持库和可执行文件双模式

### 2. src/lib.rs - 全局配置和公共接口

**职责**:
- 定义全局常量（模型超参数）
- 声明公共模块和导出
- 定义 `Layer` trait（所有神经网络层的统一接口）

**关键代码**:
```rust
// 全局配置（教育友好的小模型参数）
pub const MAX_SEQ_LEN: usize = 128;      // 序列最大长度
pub const EMBEDDING_DIM: usize = 256;    // 嵌入维度
pub const HIDDEN_DIM: usize = 512;       // 前馈隐藏层维度
pub const NUM_HEADS: usize = 8;          // 注意力头数
pub const NUM_LAYERS: usize = 2;         // Transformer 层数
pub const VOCAB_SIZE: usize = 30000;     // 词汇表目标大小
pub const DROPOUT_RATE: f32 = 0.1;       // Dropout 比率

// Layer trait（统一接口）
pub trait Layer: Send + Sync {
    fn layer_type(&self) -> &str;
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
    fn parameters(&self) -> usize;
    fn set_training_mode(&mut self, training: bool);
}

// 模块导出
pub mod llm;
pub mod vocab;
pub mod embeddings;
pub mod self_attention;
// ... (其他模块)
```

### 3. src/main.rs - 主程序入口

**职责**:
- 命令行交互界面
- 训练流程编排（预训练 + 指令微调）
- 交互式推理模式
- 日志和性能监控

**核心流程**:
```rust
fn main() {
    simple_logger::init().unwrap();
    
    // 1. 加载训练数据
    let pretraining_data = load_json_data("data/pretraining_data.json");
    let chat_data = load_json_data("data/chat_training_data.json");
    
    // 2. 构建词汇表（合并两个数据集）
    let vocab = Vocab::build_from_texts(&all_texts);
    
    // 3. 预训练阶段（学习世界知识）
    let mut model = LLM::new(vocab.clone());
    model.train_monitored(&pretraining_data, 500, 0.001, "预训练");
    
    // 4. 指令微调（学习对话模式）
    model.train_monitored(&chat_data, 500, 0.0005, "指令微调");
    
    // 5. 保存模型
    model.save_to_file("model_checkpoint.bin");
    
    // 6. 交互式推理
    interactive_mode(&mut model);
}
```

**交互模式示例**:
```
请输入问题（输入 'quit' 退出）: 什么是深度学习？
[Beam Search, width=3, max_len=20]
模型回答: 深度学习是机器学习的一个分支，使用多层神经网络...
```

---

## 🧠 核心模块详解

### Neural Network Layers（神经网络层）

#### src/embeddings.rs - 嵌入层

**功能**:
- Token 嵌入：将 token ID 映射到 256 维向量
- 位置编码：添加位置信息（使用 `position_encoding.rs`）

**关键方法**:
```rust
pub struct Embeddings {
    token_embed: Array2<f32>,     // (vocab_size, embedding_dim)
    position_encoder: PositionEncoding,
}

impl Layer for Embeddings {
    fn forward(&mut self, token_ids: &Array2<f32>) -> Array2<f32> {
        // 1. Token 嵌入查找
        // 2. 添加位置编码
        // 3. 返回 (batch_size, seq_len, embedding_dim)
    }
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 更新 token_embed 权重
    }
}
```

**输入/输出**:
- 输入: Token IDs `(seq_len, 1)` - 整数数组
- 输出: 嵌入向量 `(seq_len, 256)` - 浮点数张量

#### src/self_attention.rs - 多头自注意力

**功能**:
- 实现 Scaled Dot-Product Attention
- 多头机制（8 个注意力头）
- KV-Cache 支持（推理优化）

**算法**:
```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
MultiHead = Concat(head_1, ..., head_8) @ W_o
```

**关键方法**:
```rust
pub struct SelfAttention {
    num_heads: usize,              // 8
    d_model: usize,                // 256
    d_k: usize,                    // 32 (256/8)
    
    w_q: Array2<f32>,              // Query 权重
    w_k: Array2<f32>,              // Key 权重
    w_v: Array2<f32>,              // Value 权重
    w_o: Array2<f32>,              // 输出权重
    
    kv_cache_enabled: bool,        // KV-Cache 开关
    cached_keys: Vec<Array2<f32>>,
    cached_values: Vec<Array2<f32>>,
}

impl Layer for SelfAttention {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 1. 线性投影: Q = X @ W_q, K = X @ W_k, V = X @ W_v
        // 2. 分割多头
        // 3. 计算注意力分数: scores = Q @ K^T / √d_k
        // 4. Softmax 归一化
        // 5. 加权求和: output = softmax(scores) @ V
        // 6. 拼接多头 + 输出投影
    }
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 反向传播更新 W_q, W_k, W_v, W_o
    }
}
```

**特殊方法**:
```rust
pub fn enable_kv_cache(&mut self);    // 启用 KV-Cache（推理加速）
pub fn clear_kv_cache(&mut self);     // 清除缓存（新会话）
```

#### src/feed_forward.rs - 前馈神经网络

**功能**:
- 两层全连接网络
- GELU 激活函数（平滑版 ReLU）
- 残差连接和 Dropout

**架构**:
```
Input (256d) → Linear (256 → 512) → GELU → Linear (512 → 256) → Output (256d)
```

**关键代码**:
```rust
pub struct FeedForward {
    w1: Array2<f32>,              // (256, 512)
    b1: Array1<f32>,              // (512,)
    w2: Array2<f32>,              // (512, 256)
    b2: Array1<f32>,              // (256,)
}

fn gelu(x: f32) -> f32 {
    // GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数
    // 近似: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
}
```

#### src/layer_norm.rs - 层归一化

**功能**:
- 归一化特征维度（防止梯度消失/爆炸）
- 可学习的 scale 和 shift 参数

**算法**:
```
μ = mean(x)
σ² = variance(x)
normalized = (x - μ) / √(σ² + ε)
output = γ * normalized + β
```

**关键参数**:
```rust
pub struct LayerNorm {
    gamma: Array1<f32>,    // Scale 参数（可学习）
    beta: Array1<f32>,     // Shift 参数（可学习）
    epsilon: f32,          // 数值稳定性（1e-5）
}
```

#### src/dropout.rs - Dropout 正则化

**功能**:
- 训练时随机丢弃 10% 的神经元
- 推理时不丢弃（乘以保留概率）

**实现**:
```rust
pub struct Dropout {
    dropout_rate: f32,           // 0.1
    training: bool,              // 训练/推理模式切换
    mask: Option<Array2<f32>>,   // 缓存的 Dropout 掩码
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if self.training {
            // 生成随机掩码: mask[i] = 1 if rand() > 0.1 else 0
            // output = input * mask / (1 - dropout_rate)
        } else {
            // 推理时直接返回
            input.clone()
        }
    }
}
```

#### src/output_projection.rs - 输出投影层

**功能**:
- 将 Transformer 输出映射到词汇表大小
- 用于下一个 token 的概率分布

**架构**:
```
Input (256d) → Linear (256 → vocab_size) → Softmax → Probabilities
```

#### src/transformer.rs - Transformer Block

**功能**:
- 组合所有层（Self-Attention + FFN + LayerNorm + Dropout）
- Pre-LN 架构（LayerNorm 在 Attention 之前）

**结构**:
```rust
pub struct TransformerBlock {
    norm1: LayerNorm,              // 第一个 LayerNorm
    attention: SelfAttention,      // 多头自注意力
    dropout1: Dropout,             // 第一个 Dropout
    
    norm2: LayerNorm,              // 第二个 LayerNorm
    feed_forward: FeedForward,     // 前馈网络
    dropout2: Dropout,             // 第二个 Dropout
}

impl Layer for TransformerBlock {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // 子层 1: LayerNorm → Self-Attention → Dropout → Residual
        let normed1 = self.norm1.forward(input);
        let attn_out = self.attention.forward(&normed1);
        let dropped1 = self.dropout1.forward(&attn_out);
        let residual1 = input + &dropped1;
        
        // 子层 2: LayerNorm → FeedForward → Dropout → Residual
        let normed2 = self.norm2.forward(&residual1);
        let ffn_out = self.feed_forward.forward(&normed2);
        let dropped2 = self.dropout2.forward(&ffn_out);
        let residual2 = &residual1 + &dropped2;
        
        residual2
    }
}
```

---

### Model Orchestration（模型编排）

#### src/llm.rs - LLM 核心类（~600 行）

**职责**: 整个语言模型的核心，负责:
1. **前向传播**: Embeddings → Transformer Blocks → Output Projection
2. **反向传播**: 链式求导更新所有层参数
3. **训练循环**: Teacher Forcing + Cross-Entropy Loss
4. **推理生成**: 多种采样策略（Greedy, Top-K, Top-P, Beam Search）
5. **上下文管理**: 对话历史维护

**关键结构**:
```rust
pub struct LLM {
    vocab: Vocab,                           // 词汇表
    layers: Vec<Box<dyn Layer>>,            // 所有神经网络层
    
    context: Vec<usize>,                    // 上下文 token IDs
    optimizer: Adam,                        // Adam 优化器
    
    max_context_length: usize,              // 上下文窗口大小（128）
}
```

**核心方法**:
```rust
// 训练方法（带监控）
pub fn train_monitored(
    &mut self, 
    dataset: &[String], 
    epochs: usize, 
    base_lr: f32,
    phase_name: &str
) {
    // 1. 数据预处理和缓存
    // 2. 余弦退火学习率调度
    // 3. 早停机制
    // 4. 梯度累积（4 步）
    // 5. 性能监控（Loss, PPL, LR, Grad, Speed, ETA）
}

// 前向传播
fn forward(&mut self, tokens: &[usize], training: bool) -> Array2<f32> {
    // 依次调用所有层的 forward 方法
}

// 反向传播
fn backward(&mut self, loss_grad: &Array2<f32>, lr: f32) {
    // 反向遍历所有层，调用 backward 方法
}

// 采样方法
pub fn sample_greedy(&self, logits: &Array1<f32>) -> usize;
pub fn sample_top_k(&self, logits: &Array1<f32>, k: usize) -> usize;
pub fn sample_top_p(&self, logits: &Array1<f32>, p: f32) -> usize;

// Beam Search 生成
pub fn generate_beam_search(
    &mut self, 
    prompt: &str, 
    beam_width: usize, 
    max_length: usize
) -> String;
```

**训练数据流**:
```
Text → Tokenize → Token IDs
  → Embeddings (256d)
  → TransformerBlock 1 (Attention + FFN)
  → TransformerBlock 2 (Attention + FFN)
  → OutputProjection (vocab_size)
  → Softmax → Probabilities
  → Cross-Entropy Loss
  → Backward Pass → Parameter Update
```

---

### Training Infrastructure（训练基础设施）

#### src/adam.rs - Adam 优化器

**功能**:
- 自适应学习率优化算法
- 维护一阶矩（动量）和二阶矩（RMSProp）

**算法**:
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

**关键参数**:
```rust
pub struct Adam {
    beta1: f32,        // 0.9（一阶矩衰减）
    beta2: f32,        // 0.999（二阶矩衰减）
    epsilon: f32,      // 1e-8（数值稳定性）
    
    m: HashMap<usize, Array2<f32>>,  // 一阶矩（momentum）
    v: HashMap<usize, Array2<f32>>,  // 二阶矩（RMSProp）
    t: usize,          // 时间步（用于偏差修正）
}
```

#### src/training_optimizations.rs - 训练优化

**功能**:
1. **余弦退火学习率调度**（Cosine Annealing with Warm Restarts）
2. **早停机制**（Early Stopping）
3. **梯度累积**（Gradient Accumulation）
4. **完整训练监控**（Loss, PPL, LR, Grad Norm, Speed, ETA）

**关键函数**:
```rust
// 余弦退火学习率（带重启）
pub fn cosine_annealing_with_restarts(
    base_lr: f32,
    epoch: usize,
    max_epochs: usize,
    num_restarts: usize
) -> f32 {
    // LR = base_lr * 0.5 * (1 + cos(π * cycle_progress))
}

// 早停检查
pub struct EarlyStopping {
    patience: usize,              // 30 epochs
    best_loss: f32,
    counter: usize,
    should_stop: bool,
}

pub fn check_early_stopping(&mut self, current_loss: f32) -> bool;
```

#### src/batch_loader.rs - 批量数据加载

**功能**:
- 一次性预处理所有训练数据（tokenization）
- 缓存 token IDs，避免重复计算
- 减少 20-30% 训练时间

**关键方法**:
```rust
pub struct BatchLoader {
    cached_token_ids: Vec<Vec<usize>>,  // 缓存的 token IDs
    vocab: Vocab,
}

pub fn preprocess_all_texts(texts: &[String], vocab: &Vocab) -> BatchLoader {
    // 一次性分词并缓存
}
```

#### src/checkpoint_manager.rs - 检查点管理

**功能**:
- 定期保存模型检查点（每 50 epochs）
- 保存最佳模型（基于 loss）
- 支持训练中断恢复

**关键方法**:
```rust
pub struct CheckpointManager {
    checkpoint_dir: String,
    best_loss: f32,
}

pub fn save_checkpoint(
    &mut self,
    model: &LLM,
    epoch: usize,
    loss: f32
) -> Result<(), String>;

pub fn load_checkpoint(path: &str) -> Result<LLM, String>;
```

---

### Data Processing（数据处理）

#### src/vocab.rs - 词汇表和分词（~1000 行）

**功能**:
1. **中文分词**: 使用 jieba-rs
2. **特殊 token 管理**: `<|pad|>`, `<|unk|>`, `<|bos|>`, `</s>` 等
3. **成语检测**: 四字成语识别（正则 + 词典）
4. **LRU 缓存**: 缓存分词结果（10,000 条目）

**核心结构**:
```rust
pub struct Vocab {
    token_to_id: HashMap<String, usize>,    // Token → ID
    id_to_token: HashMap<usize, String>,    // ID → Token
    jieba: Jieba,                            // Jieba 分词器
    idioms: HashSet<String>,                 // 成语词典
}

// 全局 LRU 缓存（线程安全）
lazy_static! {
    static ref TOKENIZER_CACHE: Mutex<LruCache<String, Vec<String>>> 
        = Mutex::new(LruCache::new(10000));
}
```

**关键方法**:
```rust
// 从文本构建词汇表
pub fn build_from_texts(texts: &[String]) -> Self {
    // 1. 分词所有文本
    // 2. 提取唯一 token
    // 3. 检测成语
    // 4. 构建双向映射
}

// 编码文本为 token IDs
pub fn encode_sequence(&self, text: &str) -> Vec<usize> {
    // 1. 检查缓存
    // 2. Jieba 分词
    // 3. Token → ID 映射
    // 4. 未知词 → <|unk|>
}

// 解码 token IDs 为文本
pub fn decode_sequence(&self, ids: &[usize]) -> String {
    // 1. ID → Token 映射
    // 2. 拼接成句子
    // 3. 去除中文之间的空格
}
```

**缓存性能**:
```rust
// 获取缓存统计
pub fn get_cache_hit_rate() -> (usize, usize, f32) {
    // 返回 (命中次数, 未命中次数, 命中率)
}
```

#### src/dataset_loader.rs - 数据加载

**功能**:
- 从 JSON 文件加载训练数据
- 简单的数组格式 `["句子1", "句子2", ...]`

**实现**:
```rust
pub fn load_json_data(path: &str) -> Result<Vec<String>, String> {
    let content = fs::read_to_string(path)?;
    let data: Vec<String> = serde_json::from_str(&content)?;
    Ok(data)
}
```

---

### Performance Optimizations（性能优化）

#### src/fused_ops.rs - 融合算子

**功能**:
- 合并多个操作减少内存分配
- 提升 15-25% 性能

**实现**:
```rust
// LayerNorm + Linear 融合
pub struct FusedLayerNormLinear {
    layer_norm: LayerNorm,
    linear_weight: Array2<f32>,
    linear_bias: Array1<f32>,
}

pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
    // 1. LayerNorm
    let normed = self.layer_norm.forward(input);
    // 2. Linear（直接在归一化结果上操作，无中间张量）
    normed.dot(&self.linear_weight) + &self.linear_bias
}

// GELU + Linear 融合
pub struct FusedGELULinear { ... }
```

#### src/position_encoding.rs - 位置编码

**功能**:
- 正弦/余弦位置编码（Attention Is All You Need）
- 预计算并缓存（避免重复计算）

**算法**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**实现**:
```rust
pub struct PositionEncoding {
    encoding: Array2<f32>,  // 预计算的 (max_seq_len, embedding_dim)
}

pub fn new(max_seq_len: usize, embedding_dim: usize) -> Self {
    // 构造时一次性计算所有位置编码
}
```

#### src/performance_monitor.rs - 性能监控

**功能**:
- 实时监控训练指标（Loss, PPL, LR, Grad Norm, Speed）
- 估算剩余时间（ETA）

**输出示例**:
```
[预训练] Epoch 10/500 | Loss: 2.345 | PPL: 10.43 | LR: 0.0009 | 
Grad: 1.234 | Speed: 15.2 samples/s | ETA: 5m 23s
```

---

### Serialization（序列化）

#### src/model_serialization.rs - 模型保存/加载

**功能**:
- 二进制序列化（bincode）- 快速、紧凑
- JSON 序列化（serde_json）- 可读、可调试
- 保存完整状态（参数 + 优化器 + Vocab）

**保存内容**:
```rust
#[derive(Serialize, Deserialize)]
pub struct ModelCheckpoint {
    vocab: Vocab,                      // 词汇表
    layer_params: Vec<LayerParams>,    // 所有层参数
    optimizer_state: OptimizerState,   // Adam 状态
    metadata: Metadata,                // 训练元数据
}
```

**关键方法**:
```rust
// 保存为二进制
pub fn save_to_file(&self, path: &str) -> Result<(), String> {
    let checkpoint = self.serialize();
    let encoded = bincode::encode_to_vec(&checkpoint, config::standard())?;
    fs::write(path, encoded)?;
}

// 从二进制加载
pub fn load_from_file(path: &str) -> Result<LLM, String> {
    let data = fs::read(path)?;
    let (checkpoint, _) = bincode::decode_from_slice(&data, config::standard())?;
    LLM::deserialize(checkpoint)
}
```

---

## 🧪 测试文件详解

### tests/ 目录结构

#### 单元测试（组件级）
- **llm_test.rs**: LLM 完整训练和推理流程
- **transformer_test.rs**: Transformer Block 前向/反向传播
- **self_attention_test.rs**: 注意力机制输出形状和梯度
- **feed_forward_test.rs**: 前馈网络功能
- **embeddings_test.rs**: 嵌入层查找和位置编码
- **output_projection_test.rs**: 输出层形状
- **adam_test.rs**: 优化器参数更新
- **dataset_loader_test.rs**: 数据加载正确性
- **position_encoding_test.rs**: 位置编码生成
- **vocab_test.rs**: 词汇表构建和编码
- **chinese_tests.rs**: 中文分词和成语检测

### 测试命名规范

```rust
#[test]
fn test_<功能>_<预期结果>() {
    // 例如:
    // test_forward_output_shape()
    // test_backward_updates_parameters()
    // test_chinese_tokenization_correctness()
}
```

### 运行测试

```bash
# 所有测试
cargo test

# 特定测试文件
cargo test --test llm_test

# 特定测试函数
cargo test test_forward_output_shape

# 显示输出
cargo test -- --nocapture
```

---

## 📚 文档文件详解

### 用户文档
- **README.md** / **README_zh.md**: 快速开始、功能介绍、安装指南
- **BATCH_TRAINING.md**: 批量训练使用教程
- **PERFORMANCE_OPTIMIZATIONS.md**: 性能优化特性说明

### 开发文档
- **CLAUDE.md**: AI 辅助开发指南（架构、数据流、开发模式）
- **IMPLEMENTATION_v0.4.0.md**: 当前版本实现笔记

### 规范文档（.spec-workflow/）
- **SPEC_WORKFLOW.md**: 完整规范和工作流程（本项目创建）
- **TECH_STACK.md**: 技术栈详解（本项目创建）
- **PROJECT_STRUCTURE.md**: 本文档

---

## 🔧 配置文件详解

### rustfmt.toml - 代码格式

```toml
edition = "2024"
max_width = 100        # 每行最大 100 字符
tab_spaces = 4         # 使用 4 空格缩进
use_small_heuristics = "Default"
```

### .gitignore - Git 忽略规则

```
/target/              # Cargo 编译输出
Cargo.lock            # 依赖锁定（库项目通常提交）
*.bin                 # 模型检查点文件
*.json.bak            # 备份文件
```

---

## 📊 数据文件详解

### data/pretraining_data.json

**格式**:
```json
[
    "地球是太阳系的第三颗行星",
    "人工智能是计算机科学的一个分支",
    "深度学习使用多层神经网络"
]
```

**特点**:
- 纯中文知识性陈述
- 约 250 条样本
- 用于预训练阶段

### data/chat_training_data.json

**格式**:
```json
[
    "你好！我是AI助手。",
    "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。"
]
```

**特点**:
- 对话风格文本
- 约 250 条样本
- 用于指令微调阶段

### data/chinese_idioms.json

**格式**:
```json
[
    "一帆风顺",
    "马到成功",
    "心想事成",
    "万事如意"
]
```

**用途**:
- 成语检测词典
- Vocab 构建时特殊标记
- 约 100+ 常见成语

---

## 🚀 构建产物

### target/ 目录（编译输出）

```
target/
├── debug/                    # 开发构建（cargo build）
│   └── llm                   # 可执行文件（未优化）
│
├── release/                  # 发布构建（cargo build --release）
│   └── llm                   # 可执行文件（完全优化）
│
└── doc/                      # 生成的文档（cargo doc）
    └── llm/
        └── index.html        # HTML 文档入口
```

### 模型检查点文件

```
model_checkpoint.bin          # 二进制格式（40-100 MB）
model_checkpoint.json         # JSON 格式（可读但更大）
```

---

## 📐 代码度量

### 代码规模（v0.4.0）

| 类别 | 文件数 | 代码行数（约） |
|------|-------|--------------|
| 核心神经网络层 | 7 | 2,500 |
| 模型编排 | 2 | 1,200 |
| 训练基础设施 | 4 | 1,500 |
| 数据处理 | 3 | 1,300 |
| 性能优化 | 3 | 800 |
| 测试 | 11 | 2,000 |
| **总计** | **30** | **~9,300** |

### 复杂度最高的文件

1. **src/llm.rs** (~600 行) - 核心模型逻辑
2. **src/vocab.rs** (~1000 行) - 词汇表和中文处理
3. **src/self_attention.rs** (~650 行) - 注意力机制
4. **src/main.rs** (~400 行) - 训练流程编排

---

## 🔍 文件快速导航

### 想要理解...

| 目标 | 主要文件 | 相关文件 |
|------|---------|---------|
| **整体架构** | lib.rs, llm.rs | CLAUDE.md |
| **训练流程** | main.rs | training_optimizations.rs |
| **前向传播** | llm.rs (forward 方法) | 各层的 forward 方法 |
| **反向传播** | llm.rs (backward 方法) | 各层的 backward 方法 |
| **注意力机制** | self_attention.rs | transformer.rs |
| **中文处理** | vocab.rs | chinese_tests.rs |
| **性能优化** | fused_ops.rs, position_encoding.rs | PERFORMANCE_OPTIMIZATIONS.md |
| **模型保存** | model_serialization.rs | checkpoint_manager.rs |

### 想要修改...

| 修改目标 | 主要文件 | 注意事项 |
|---------|---------|---------|
| **模型超参数** | lib.rs | 需保证 EMBEDDING_DIM % NUM_HEADS == 0 |
| **训练数据** | data/*.json | 纯中文 JSON 数组 |
| **学习率策略** | training_optimizations.rs | 影响收敛速度 |
| **采样策略** | llm.rs (generate_* 方法) | 影响生成质量 |
| **添加新层** | src/<新层名>.rs + lib.rs | 实现 Layer trait |
| **优化器** | adam.rs | 修改 beta1, beta2, epsilon |

---

## 📋 添加新功能的步骤

### 示例：添加新的神经网络层

1. **创建文件**: `src/new_layer.rs`
2. **实现结构体**:
   ```rust
   pub struct NewLayer {
       // 参数定义
   }
   ```
3. **实现 Layer trait**:
   ```rust
   impl Layer for NewLayer {
       fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> { ... }
       fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> { ... }
       // ...
   }
   ```
4. **在 lib.rs 中声明**:
   ```rust
   pub mod new_layer;
   ```
5. **在 llm.rs 中集成**:
   ```rust
   self.layers.push(Box::new(NewLayer::new(...)));
   ```
6. **添加测试**: `tests/new_layer_test.rs`
7. **更新文档**: CLAUDE.md, README.md

---

*最后更新: 2024-10-25 | 版本: v0.4.0*
