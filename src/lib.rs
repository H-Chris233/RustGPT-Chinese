//! # RustGPT-Chinese - 从零实现的中文语言模型
//!
//! 这是一个纯 Rust 实现的 Transformer 语言模型教育项目。
//! 本项目的目标是帮助学习者深入理解 Transformer 架构和语言模型的工作原理。
//!
//! ## 核心设计理念
//!
//! 1. **纯 Rust 实现**：不依赖 PyTorch/TensorFlow，只使用 `ndarray` 进行矩阵运算
//! 2. **从零构建**：手动实现所有神经网络组件（注意力机制、前馈网络等）
//! 3. **教育优先**：代码注重可读性和教学性，而非生产性能
//! 4. **中文特化**：针对中文语言特性进行优化（jieba 分词、成语识别等）
//!
//! ## 模块组织
//!
//! ### 核心模型组件
//! - `llm`: 语言模型主类，包含训练和推理逻辑
//! - `transformer`: Transformer 块实现（注意力 + 前馈网络）
//! - `self_attention`: 多头自注意力机制的核心实现
//!
//! ### 神经网络层
//! - `embeddings`: 词嵌入层（token embedding + 位置编码）
//! - `feed_forward`: 前馈神经网络（FFN）
//! - `layer_norm`: 层归一化
//! - `dropout`: Dropout 正则化层
//! - `output_projection`: 输出投影层（映射到词汇表）
//!
//! ### 工具模块
//! - `vocab`: 词汇表管理和中文分词（jieba-rs）
//! - `adam`: Adam 优化器实现
//! - `dataset_loader`: 训练数据加载器
//! - `position_encoding`: 正弦位置编码
//! - `model_serialization`: 模型序列化和反序列化
//! - `performance_monitor`: 性能监控工具
//! - `utils`: 通用工具函数

// ============================================================================
// 模块声明
// ============================================================================

pub mod adam; // Adam 优化器：带动量的自适应学习率优化算法
pub mod batch_loader; // 批量数据加载器：数据分桶、PAD填充、注意力掩码
pub mod checkpoint_manager; // 检查点管理器：保存和加载训练状态
pub mod dataset_loader; // 数据加载器：处理预训练和对话数据
pub mod dropout; // Dropout层：随机丢弃神经元，防止过拟合
pub mod embeddings; // 嵌入层：将token ID转换为稠密向量表示
pub mod feed_forward; // 前馈网络：Transformer中的全连接层部分
pub mod fused_ops; // 算子融合：LayerNorm+Linear、GELU+Linear等融合操作
pub mod layer_norm; // 层归一化：稳定训练的归一化技术
pub mod llm; // 语言模型主类：整合所有组件的核心模型
pub mod model_serialization; // 模型序列化：保存和加载模型权重
pub mod output_projection; // 输出投影层：将隐藏状态映射到词汇表概率
pub mod performance_monitor;
pub mod position_encoding; // 位置编码：为序列注入位置信息
pub mod self_attention; // 自注意力机制：Transformer的核心组件
pub mod training_optimizations; // 训练优化：缓存、学习率调度、早停等
pub mod transformer; // Transformer块：注意力+前馈的完整模块
pub mod utils; // 工具函数：通用辅助函数
pub mod vocab; // 词汇表：管理token和ID的映射关系 // 性能监控：记录和分析训练/推理性能

// ============================================================================
// 重导出核心类型（简化外部使用）
// ============================================================================

pub use batch_loader::{Batch, BatchLoader, PAD_TOKEN_ID};
pub use checkpoint_manager::{CheckpointManager, CheckpointMetadata, CheckpointStrategy};
pub use dataset_loader::Dataset;
pub use embeddings::Embeddings;
pub use llm::{LLM, Layer};
pub use model_serialization::{
    load_model_auto, load_model_binary, load_model_json, save_model_binary, save_model_json,
};
pub use output_projection::OutputProjection;
pub use performance_monitor::PerformanceMonitor;
pub use transformer::TransformerBlock;
pub use utils::sample_normal;
pub use vocab::Vocab;

// ============================================================================
// 模型超参数（Model Hyperparameters）
// ============================================================================

/// **最大序列长度**
///
/// 定义模型能处理的最长token序列。对于中文，一个字通常是一个token。
///
/// **为什么设置为256？**
/// - 中文句子通常比英文短（一个汉字携带更多信息）
/// - 256个token约等于256个汉字，足够覆盖大部分对话和短文本
/// - 更长的序列会增加计算复杂度（O(n²) 的注意力计算）
///
/// **实际应用**：
/// - 超过此长度的输入会被截断
/// - 在对话系统中，会保留最近的256个token作为上下文
pub const MAX_SEQ_LEN: usize = 128;

/// **嵌入维度 (Embedding Dimension)**
///
/// 每个token被表示为一个512维的向量。这是模型的"内部表示空间"的维度。
///
/// **为什么选择512？**
/// - 比传统的256维更大，能更好地表示中文字符的丰富语义
/// - 中文字符数量多（常用字3500+），需要更大的表示空间来区分
/// - 512维在教育项目中是性能和可训练性的平衡点
///
/// **影响**：
/// - 维度越大，模型表达能力越强，但计算成本也越高
/// - 所有层的输入输出都必须匹配这个维度
pub const EMBEDDING_DIM: usize = 256;

/// **隐藏层维度 (Hidden Dimension)**
///
/// 前馈网络（FFN）中间层的维度。在Transformer中，FFN通常是嵌入维度的2-4倍。
///
/// **为什么是1024？**
/// - 遵循 Transformer 论文的设计：hidden_dim = 2 × embedding_dim
/// - 更大的隐藏层能学习更复杂的特征变换
/// - 1024维足够处理中文语言的复杂模式（成语、多义词等）
///
/// **作用**：
/// - 在 FFN 中：512 → 1024 → 512（先扩展再压缩）
/// - 这种"瓶颈结构"帮助模型提取抽象特征
pub const HIDDEN_DIM: usize = 512;

/// **目标词汇表大小**
///
/// 理论上的词汇表上限。实际词汇表大小由训练数据动态构建。
///
/// **为什么是30000？**
/// - 常用中文字约3500个，加上词组、标点、英文，通常在5000-15000之间
/// - 30000是一个保守的上限，确保能容纳所有可能的词元
/// - 实际使用中，词汇表会根据训练数据动态构建，通常小于这个值
///
/// **注意**：
/// - 词汇表越大，输出层（output_projection）的参数就越多
/// - 本项目使用 jieba 分词，词汇量相对可控
pub const VOCAB_SIZE: usize = 30000;

// ============================================================================
// 数值稳定性常量 (Numerical Stability Constants)
// ============================================================================

/// **通用数值稳定性常量**
///
/// 用于避免除零错误和数值下溢。在层归一化、梯度计算等场景中使用。
///
/// **典型应用**：
/// ```rust
/// use llm::EPSILON;
/// let x = 1.0f32;
/// let variance = 0.5f32;
/// let normalized = x / (variance + EPSILON).sqrt(); // 避免除以0
/// ```
pub const EPSILON: f32 = 1e-8;

/// **对数运算专用常量**
///
/// 在计算交叉熵损失时，需要对概率取对数。由于 log(0) = -∞，
/// 我们需要确保输入始终大于零。
///
/// **应用场景**：
/// ```rust
/// use llm::LOG_EPSILON;
/// let prob = 0.5f32;
/// let loss = -prob.max(LOG_EPSILON).ln(); // 避免 log(0)
/// ```
///
/// **为什么是1e-10？**
/// - 比 EPSILON 更小，因为对数函数在接近0时变化剧烈
/// - 足够小以不影响正常概率值（通常在 0.001-0.999 范围内）
pub const LOG_EPSILON: f32 = 1e-10;

/// **Softmax 归一化专用常量**
///
/// Softmax 函数将任意实数向量转换为概率分布。
/// 这个常量用于确保 softmax 输出始终大于0。
///
/// **应用示例**：
/// ```rust
/// use llm::SOFTMAX_EPSILON;
/// let prob = 0.001f32;
/// let stable_prob = prob.max(SOFTMAX_EPSILON); // 确保概率>0
/// ```
///
/// **为什么是1e-12？**
/// - 需要比 LOG_EPSILON 更小，因为 softmax 输出可能非常接近0
/// - 对于低概率token（如罕见词），这个保护很重要
pub const SOFTMAX_EPSILON: f32 = 1e-12;
