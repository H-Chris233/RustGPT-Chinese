# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustGPT-Chinese is a Chinese-specialized transformer-based language model built from scratch in pure Rust using only `ndarray` for matrix operations. It's an educational project demonstrating how LLMs work at the implementation level, not a production-grade system.

## Development Commands

### Building and Running
```bash
# Run the main training pipeline
cargo run

# Build optimized release version
cargo build --release

# Run with verbose output
cargo test -- --nocapture
```

### Testing
```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test
cargo test --test chinese_tests
cargo test --test vocab_test

# Run individual component tests
cargo test --test feed_forward_test
cargo test --test embeddings_test
cargo test --test output_projection_test
cargo test --test adam_test
cargo test --test dataset_loader_test
cargo test --test position_encoding_test
```

### Code Quality
```bash
# Format code
cargo fmt

# Run linter
cargo clippy
```

## Architecture Overview

### Core Design Philosophy

This is a **pure Rust, from-scratch implementation** with no PyTorch, TensorFlow, or Candle dependencies. All neural network operations are implemented manually using `ndarray` for matrix operations and `jieba-rs` for Chinese tokenization.

### Data Flow

```
Input Text → Jieba Tokenization → Token IDs → Embeddings (256d)
    → 2x Transformer Blocks (attention + FFN + dropout + layer norm)
    → Output Projection → Softmax → Token Predictions
```

### Key Architecture Components

**Network Stack (in order):**
1. **Embeddings Layer** (`embeddings.rs`) - Token embedding optimized for Chinese
2. **2x Transformer Blocks** (`transformer.rs`) - Each contains:
   - Multi-head self-attention (8 heads, `self_attention.rs`)
   - Feed-forward network (`feed_forward.rs`)
   - 2x Dropout layers (10% rate, `dropout.rs`)
   - 2x Layer normalization (`layer_norm.rs`)
3. **Output Projection** (`output_projection.rs`) - Maps to vocabulary size

**Supporting Systems:**
- **Vocabulary Management** (`vocab.rs`) - Handles Chinese tokenization with jieba-rs, idiom detection, special tokens
- **Position Encoding** (`position_encoding.rs`) - Positional information for sequence understanding
- **Semantic Enhancer** (`semantic_enhancer.rs`) - Chinese-specific relationship enhancement
- **Adam Optimizer** (`adam.rs`) - Gradient-based optimization with momentum
- **Dataset Loader** (`dataset_loader.rs`) - Loads pre-training and chat training data from JSON

### Model Configuration (lib.rs) - v0.4.0

```rust
MAX_SEQ_LEN: 128        // Optimized for small datasets
EMBEDDING_DIM: 256      // Reduced for better convergence on limited data
HIDDEN_DIM: 512         // Reduced for better convergence on limited data
VOCAB_SIZE: 30000       // Target vocab size (dynamically built from data)
```

**v0.4.0 性能优化特性**:
- **🚀 BLAS 加速**: ndarray 启用 OpenBLAS 后端，张量计算性能提升 30-50%
- **💾 KV-Cache**: 推理时缓存 Key/Value 矩阵，避免重复计算（已在 v0.3.2 实现）
- **🔄 Tokenizer 缓存**: LRU 缓存（10,000 条目）缓存 jieba 分词结果，重复文本加速 5-10x
- **⚡ 算子融合**: FusedLayerNormLinear、FusedGELULinear 减少中间张量分配
- **📊 性能监控**: 缓存命中率统计，便于性能分析

**v0.5.0 混合精度训练**:
- **🎯 FP16/BF16 支持**: 前向和反向传播使用低精度计算
- **⚖️ 双权重系统**: Master 权重（FP32）+ Working 副本（FP16/BF16）
- **📈 动态损失缩放**: 自动调整缩放因子，防止梯度下溢/溢出
- **🛡️ 自动回退**: 检测持续不稳定时自动切换回 FP32
- **📊 完整监控**: 精度类型、缩放因子、溢出率实时显示
- **验证测试**: `cargo run --example mixed_precision_test`

**v0.3.1 训练优化**:
- Smaller model = fewer parameters = better fit for 200-500 training samples
- Reduces risk of severe underfitting when training data is limited
- Parameter count reduced from ~70M to ~10M (86% reduction)
- **🚀 v0.3.1 新增训练优化**: 数据缓存、余弦退火学习率、早停、梯度累积

### Training Pipeline (main.rs) - v0.3.1 with Performance Optimizations

The training process has two phases with v0.3.1 performance optimizations:

1. **Vocabulary Building**: Processes both pre-training and chat training data using jieba-rs to extract all unique tokens (Chinese words, idioms, punctuation, special tokens)

2. **Pre-training** (max 500 epochs, LR=0.001 with cosine annealing):
   - Loads data from `data/pretraining_data.json`
   - Learns Chinese world knowledge and factual statements
   - **🚀 v0.3.1优化**: 余弦退火学习率调度（2次重启），早停机制（patience=30）
   - Uses `train_monitored()` method with all optimizations

3. **Instruction Tuning** (max 500 epochs, LR=0.0005 with cosine annealing):
   - Loads data from `data/chat_training_data.json`
   - Learns conversational Chinese patterns
   - **🚀 v0.3.1优化**: 梯度累积（4步），完整训练监控，早停机制
   - Uses `train_monitored()` method with all optimizations

4. **Interactive Mode**:
   - Beam search decoding (width=3, max_length=20)
   - Context window management for multi-turn conversations
   - Chinese text post-processing to remove extra spaces

**Training Data Changes (v0.3.0):**
- Removed `</s>` tokens from all training data to prevent output contamination
- Cleaner training signal = better model quality

**🚀 v0.3.1 训练优化特性:**
- **数据预处理缓存**: 一次性tokenize，避免重复计算（减少20-30%时间）
- **余弦退火学习率**: 带重启的调度策略（提升15-25%收敛速度）
- **早停机制**: 自动检测训练收敛（节省10-40%时间）
- **梯度累积**: 4步累积，等价batch_size=4（提升40%稳定性）
- **完整监控**: Loss, PPL, LR, Grad, Speed, ETA实时显示
- **训练方法**: 使用新的`train_monitored()`方法替代原有`train()`

### Chinese Language Handling

**Tokenization Strategy** (vocab.rs, llm.rs):
- Detects Chinese characters (Unicode range 0x4E00-0x9FFF)
- Uses jieba-rs for Chinese word segmentation
- Falls back to whitespace tokenization for non-Chinese text
- Handles Chinese punctuation as separate tokens
- Extracts 4-character Chinese idioms (成语) with pattern matching

**Special Processing:**
- Idiom detection via regex patterns and dictionary lookup (`data/chinese_idioms.json`)
- Phrase extraction for multi-character meaningful phrases
- Post-processing removes extra spaces between Chinese characters

### Training Mechanics (llm.rs)

**Forward Pass:**
- Teacher forcing: input is `tokens[:-1]`, target is `tokens[1:]`
- Each token position predicts the next token
- Supports multiple sampling strategies:
  - Greedy decoding (takes highest probability)
  - Top-k sampling (samples from k most probable tokens)
  - Top-p (nucleus) sampling (samples from smallest set with cumulative prob > p)
  - Beam search (maintains multiple candidate sequences)
  - Temperature scaling for output diversity

**Backward Pass:**
- Cross-entropy loss with numerical stability (`max(1e-15)` clipping)
- Softmax + cross-entropy gradient: `softmax_probs - one_hot(target)`
- Gradient clipping (L2 norm max 5.0) for stability
- Gradients propagated backwards through all layers
- Each layer updates its own parameters with Adam optimizer

**Context Management:**
- Context window maintains conversation history (up to MAX_SEQ_LEN tokens, 128 in v0.3.1)
- Oldest tokens removed when exceeding max length
- Context cleared on `</s>` token detection (though `</s>` removed from training data in v0.3.0)

### Layer Interface (llm.rs)

All layers implement the `Layer` trait:
```rust
trait Layer {
    fn layer_type(&self) -> &str;                          // Returns layer name
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;  // Forward pass
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;  // Backward pass
    fn parameters(&self) -> usize;                         // Parameter count
    fn set_training_mode(&mut self, training: bool);       // Toggle dropout
}
```

### Transformer Block Details (transformer.rs)

Each block applies this sequence (Pre-LN architecture):
1. LayerNorm → Self-attention → Dropout (10%) → Residual connection
2. LayerNorm → Feed-forward → Dropout (10%) → Residual connection

Note: v0.2.0 upgraded to Pre-LN architecture with explicit residual connections for better training stability.

### Special Tokens

Defined in `vocab.rs`:
- `<|pad|>` (ID: 0) - Padding
- `<|unk|>` (ID: 1) - Unknown tokens
- `<|bos|>` (ID: 2) - Beginning of sequence
- `</s>` (ID: 3) - End of sequence (note: removed from training data in v0.3.0 but still defined in vocab)
- `<|sep|>` (ID: 4) - Separator
- `<|cls|>` (ID: 5) - Classification
- `<|mask|>` (ID: 6) - Masked token
- `<|bos|>` (ID: 2) - Beginning of sequence
- `</s>` (ID: 3) - End of sequence (used to trigger context clearing)
- `<|sep|>` (ID: 4) - Separator
- `<|cls|>` (ID: 5) - Classification
- `<|mask|>` (ID: 6) - Masked token

## Common Development Patterns

### Adding a New Layer

1. Create the layer struct with parameters and cache for forward pass values
2. Implement the `Layer` trait with forward/backward/parameters/set_training_mode
3. In forward: cache inputs needed for backward pass
4. In backward: compute gradients w.r.t. inputs and update parameters
5. Add the layer to the network stack in `main.rs` or `llm.rs`
6. Create a test file in `tests/` following existing patterns

### Modifying Training Data

Training data is loaded from JSON files in `data/`:
- `data/pretraining_data.json` - Array of Chinese factual statements (no `</s>` tokens in v0.3.0)
- `data/chat_training_data.json` - Array of conversational exchanges (no `</s>` tokens in v0.3.0)
- `data/chinese_idioms.json` - Array of 4-character Chinese idioms

Format: Simple JSON arrays of strings.

**v0.3.0 Note**: `</s>` tokens were removed from training data to prevent output contamination and improve model quality.

### Working with Chinese Text

When processing Chinese text:
- Always check for Chinese characters using: `(char as u32) >= 0x4E00 && (char as u32) <= 0x9FFF`
- Use jieba-rs tokenizer for segmentation: `jieba.cut(text, false)`
- Be aware that tokenization happens in both `vocab.rs` and `llm.rs`
- Post-processing removes spaces between Chinese characters for fluency

### Debugging Model Output

Key files to examine:
- `src/llm.rs:251` - Training loop with loss printing
- `src/llm.rs:95` - Sampling methods (temperature, top-k, top-p)
- `src/llm.rs:122` - Beam search implementation
- `src/llm.rs:742` - Chinese text post-processing
- `src/main.rs:124` - Interactive mode with beam search

## Mixed Precision Training (v0.5.0)

### Overview

RustGPT-Chinese supports FP16/BF16 mixed precision training to accelerate training and reduce memory usage while maintaining numerical stability through dynamic loss scaling.

### Quick Start

```rust
use llm::{LLM, MixedPrecisionConfig, MixedPrecisionTrainer};

// Create FP16 configuration
let config = MixedPrecisionConfig::fp16();
let mut trainer = MixedPrecisionTrainer::new(config);

// Train with mixed precision
let mut llm = LLM::default();
trainer.train_monitored(&mut llm, tokenized_data, 100, 0.001, 30);
```

### Key Features

1. **Precision Types**:
   - `FP16`: Half precision (16-bit), 50% memory, good for most tasks
   - `BF16`: BFloat16 (16-bit), same dynamic range as FP32, better stability
   - `FP32`: Full precision (32-bit), baseline/fallback

2. **Dynamic Loss Scaling**:
   - Automatically adjusts scale factor to prevent gradient underflow
   - Detects overflow (NaN/Inf) and skips corrupted updates
   - Grows scale when training is stable
   - Backs off scale when overflow occurs

3. **Dual-Weight System**:
   - Master weights: Always FP32 (high precision)
   - Working copy: FP16/BF16 for computation (low precision)
   - Gradients accumulated in FP32 after unscaling

4. **Auto-Fallback**:
   - Detects persistent instability (e.g., 5 consecutive overflows)
   - Automatically switches back to FP32
   - Logs fallback events for debugging

### Configuration Examples

```rust
// BF16 with custom parameters
let config = MixedPrecisionConfig::bf16()
    .with_loss_scale(32768.0)           // Initial scale
    .with_growth_params(2.0, 2000)       // Growth factor and interval
    .with_backoff_factor(0.5)            // Backoff on overflow
    .with_auto_fallback(true, 5);        // Enable fallback after 5 overflows

// Disable mixed precision (pure FP32)
let config = MixedPrecisionConfig::disabled();
```

### Monitoring

Training logs show:
- Current precision type (FP16/BF16/FP32)
- Loss scale factor
- Overflow count and rate
- Fallback events

```
[MIXED PRECISION] Training with precision: FP16, loss scale: 65536
Epoch 10: Loss = 3.89, LR = 0.000995, Precision = FP16, Scale = 65536.0, Overflows = 2/110 (1.82%)
[OVERFLOW] Step 125: Loss scale reduced from 65536.0 to 32768.0 (overflow #3/125)
[SCALE] Loss scale increased from 32768.0 to 65536.0 after 2000 stable steps
```

### Validation

Run the validation test:
```bash
cargo run --example mixed_precision_test
```

This compares FP32, FP16, and BF16 training on a small model and verifies:
- Loss convergence (< 5% difference from FP32)
- Perplexity stability (< 3% difference)
- Overflow rate (< 10%)

### Best Practices

1. **Start with BF16**: Better numerical stability than FP16
2. **Monitor overflow rate**: Should stay below 10%
3. **Compare with FP32**: Validate loss curves match
4. **Use auto-fallback**: Prevents training failure
5. **Check logs**: Watch for [OVERFLOW] and [FALLBACK] events

### Implementation Files

- `src/mixed_precision.rs` - Configuration and precision types
- `src/loss_scaler.rs` - Dynamic loss scaling implementation
- `src/mixed_precision_trainer.rs` - Training loop integration
- `src/precision_convert.rs` - FP32 ↔ FP16/BF16 conversion utilities
- `examples/mixed_precision_test.rs` - Validation script
- `docs/MIXED_PRECISION.md` - Detailed documentation

### Troubleshooting

- **High overflow rate (>10%)**: Use BF16 or lower learning rate
- **Loss not converging**: Switch to FP32 or increase loss scale
- **Frequent fallback**: Increase fallback threshold or use FP32
- **Scale keeps decreasing**: Check for gradient explosion, reduce LR

For more details, see `docs/MIXED_PRECISION.md`.

## Performance Optimization Features (v0.4.0)

### 1. BLAS-Accelerated Tensor Operations (Optional)

BLAS support is available as an optional feature:
```bash
# Enable BLAS acceleration (requires system OpenBLAS)
cargo build --features blas
cargo run --features blas --release

# Without BLAS (default, pure Rust)
cargo build
cargo run --release
```

**Benefits** (when enabled):
- 30-50% faster matrix multiplication
- Optimized GEMM (General Matrix Multiply) operations
- Better cache utilization

**Requirements**:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# macOS (Homebrew)
brew install openblas
```

### 2. Tokenizer LRU Cache

Caches jieba tokenization results to avoid redundant computation:
- **Capacity**: 10,000 entries
- **Strategy**: LRU (Least Recently Used)
- **Hit rate monitoring**: `vocab::get_cache_hit_rate()`
- **Location**: `src/vocab.rs` - global `TOKENIZER_CACHE`

**Usage**:
```rust
use llm::vocab::{Vocab, get_cache_hit_rate, reset_cache_stats};

let vocab = Vocab::build_from_texts(&texts);
let tokens = vocab.encode_sequence("深度学习很有趣"); // Uses cache automatically
let (hits, misses, rate) = get_cache_hit_rate();
```

### 3. KV-Cache for Inference

Already implemented in `self_attention.rs`:
- Caches Key and Value matrices during autoregressive generation
- Avoids recomputing attention for previous tokens
- Enable with `attention.enable_kv_cache()`
- Clear with `attention.clear_kv_cache()`

### 4. Operator Fusion

New fused operations in `src/fused_ops.rs`:
- **FusedLayerNormLinear**: Combines LayerNorm + Linear in one pass
- **FusedGELULinear**: Combines GELU activation + Linear

**Benefits**:
- Reduces intermediate tensor allocations
- Better memory locality
- 15-25% faster than separate operations

**Usage**:
```rust
use llm::fused_ops::FusedLayerNormLinear;

let fused_op = FusedLayerNormLinear::new(512, 1024);
let output = fused_op.forward(&input);
```

### 5. Performance Benchmarks

Run benchmarks to measure improvements:
```bash
# Run all benchmarks
cargo bench --bench performance_benchmark

# Run specific benchmark
cargo bench --bench memory_optimization_bench
```

Benchmarks test:
- Tensor operation speed (BLAS vs pure Rust)
- Tokenizer cache hit rates
- KV-cache speedup in inference
- Fused operations vs separate operations

## Testing Strategy

Tests are organized by component in the `tests/` directory. Each test file corresponds to a source module. Key test patterns:

- **Forward pass tests**: Verify output shapes and basic functionality
- **Backward pass tests**: Check gradient computation (often by verifying parameters change)
- **Chinese-specific tests**: Validate tokenization, idiom detection, and text processing
- **Integration tests**: Test full training pipeline components together
- **Performance tests**: Benchmark critical paths (in `benches/` directory)

## Known Limitations

- Model persistence is supported via binary (.bin) and JSON formats (see `model_serialization.rs`)
- Small model size (v0.3.0: 10M parameters) limits generalization beyond training data
- Training data limited to ~250 samples - more data needed for production use
- No attention masking for autoregressive generation (relies on teacher forcing)
- No batching support (processes one sequence at a time)
- Training data is loaded from JSON files (expandable)
- No learning rate warmup (uses exponential decay only)
- No gradient accumulation
- No residual connections in transformer blocks (reduces training stability)
- Limited vocabulary size (dynamically built from training data only)
- No attention masking for autoregressive generation
- No batching support (processes one sequence at a time)
- Training data is hardcoded in JSON files
- No learning rate warmup
- No gradient accumulation

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `check.yml` - Runs `cargo clippy` and `cargo fmt --check`
- `test.yml` - Runs full test suite with `cargo test`

## Project Philosophy

This project prioritizes **educational clarity over production performance**. Code is intentionally verbose and explicit to demonstrate how transformers work at the implementation level. Avoid adding heavy ML framework dependencies - keep the "from scratch" philosophy intact.
