# 🦀 RustGPT-Chinese - Chinese-Supported LLM

[![Check](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/check.yml/badge.svg)](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/check.yml) [![Test](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/test.yml/badge.svg)](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/test.yml)

**[中文！](README.md)**

A complete **Chinese-Supported Large Language Model implementation in pure Rust** with no external ML frameworks. Built from the ground up using only `ndarray` for matrix operations, featuring a modern **Pre-LN Transformer architecture** (GPT-2 standard).

## 🚀 What This Is

This project demonstrates how to build a transformer-based language model from scratch in Rust that supports Chinese language processing, including:

- **Modern Pre-LN Transformer Architecture** - GPT-2/3 standard with explicit residual connections
- **Pre-training** on Chinese factual text completion
- **Instruction tuning** for Chinese conversational AI
- **Interactive chat mode** for Chinese language testing
- **Full backpropagation** with gradient clipping and Adam optimizer
- **Modular architecture** with clean separation of concerns
- **Chinese-optimized tokenization** using jieba-rs with global singleton optimization (50-70% faster)
- **Multi-head self-attention mechanism** (8 heads) for better Chinese grammar understanding
- **Context window management** for maintaining conversation history
- **Advanced decoding methods** (top-k/top-p sampling, beam search, temperature scaling)
- **Regularization techniques** (Dropout, Layer Normalization) for improved stability
- **Performance monitoring** with detailed timing and profiling

## ❌ What This Isn't

This is not a production grade Chinese LLM. It is so far away from the larger Chinese models.

This is just a toy project that demonstrates how Chinese LLMs work under the hood.

## 🆕 Recent Updates

### Training API Cleanup & Teaching-Focused Simplification (2026-03-06)
- ✅ **Reduced public training surface**: the recommended public entrypoints are now `train(...)`, `train_monitored(...)`, `train_bucketed_sequential(...)`, and `train_with_checkpointing(...)`
- ✅ **Shared single-step training path**: the teaching-oriented training flow now centers on `PreparedTrainingStep`, `prepare_training_step(...)`, and `backward_with_ctx(...)`
- ✅ **Unified loss accounting**: public training APIs now report **token-weighted mean loss**, so `Loss` / `PPL` are semantically consistent across entrypoints
- ✅ **Honest batch naming**: `train_bucketed_sequential(...)` means “batch-organized data with per-sample sequential updates”, not strict batch-gradient averaging

### Training Correctness: PAD/mask/illegal targets (2026-03-05)
- ✅ **Batch/PAD mask wiring**: `SelfAttention` / `TransformerBlock` consume `attention_mask` in `forward_batch(...)` (key padding mask)
- ✅ **Remove `PAD_ID==0` assumptions**: padding/loss/grads use `vocab.pad_token_id()` as the single source of truth; `BatchLoader` supports injecting `pad_token_id`
- ✅ **Training signal guardrails**: shape mismatch / out-of-range targets / all-PAD now skip optimizer steps to avoid Adam momentum drift and to improve debuggability
- ✅ **New/updated tests**: `self_attention_forward_batch_mask_test`, updated `compute_gradients_step_test`

### v0.4.0 - Checkpoint Management & Training Resume (2025-10-28)
- 🚀 **Checkpoint Manager** - Supports Best/Last/Periodic saving strategies with automatic cleanup
- ✅ **Complete State Saving** - Model parameters + Adam optimizer state (m, v, timestep) for true resume
- ✅ **Early Stopping Integration** - Auto-saves best checkpoint, auto-rollback to best state on early stop
- ✅ **Resume Training** - Resume from checkpoints with full training continuity
- ✅ **CLI Parameter Support** - `--resume`, `--resume-from`, `--checkpoint-dir` and more
- ✅ **Integration Tests** - Verifies loss continuity after save/restore (< 0.1% difference)

### v0.3.1 - Training Performance Optimization (2025-10-16)
- 🚀 **Phase 1 Training Optimizations** - Introduced monitoring, scheduling, and caching support for training
- ✅ **Data Preprocessing Cache** - Avoid repeated tokenization
- ✅ **Cosine LR Scheduling** - The current main training path has since converged to warmup + cosine decay (no restarts by default)
- ✅ **Early Stopping** - Auto-detect convergence
- ✅ **Enhanced Training Monitor** - Loss, PPL, LR, Grad, Speed, ETA monitoring
- ✅ **Gradient Accumulation** - Supported through the monitored training path when enabled

### v0.3.0 - Model Optimization for Small Datasets (2025-10-15)
- ✅ **Reduced Model Size** - Optimized for limited training data: 2 layers (was 4), 256 embedding dim (was 512)
- ✅ **Training Enhancement** - Increased epochs to 500 (was 100), higher learning rates (0.001/0.0005)
- ✅ **Cleaner Output** - Removed `</s>` tokens from training data to prevent output contamination
- ✅ **Parameter Reduction** - ~86% fewer parameters (10M vs 70M) for better convergence on small datasets
- 🎯 **Target Use Case** - Optimized for 200-500 training samples, expected loss < 0.1

### v0.2.0 - Architecture Refactoring (2025-10-12)
- ✅ **Pre-LN Transformer Architecture** - Upgraded from Post-LN to Pre-LN (GPT-2 standard) for better training stability
- ✅ **Explicit Residual Connections** - Moved residual connections from sub-layers to TransformerBlock for clarity
- ✅ **Removed Semantic Enhancer** - Simplified model by removing unverified experimental feature
- ✅ **Performance Optimization** - Jieba singleton optimization (50-70% faster), attention reshape optimization (20-30% faster)
- ✅ **Compiler Optimizations** - LTO, opt-level 3, codegen-units 1 for release builds
- ✅ **Performance Monitoring** - Added comprehensive performance tracking and profiling

## 🔍 Key Files to Explore

Start with these core files to understand the implementation:

- **[`src/main.rs`](src/main.rs)** - Training pipeline, data preparation, and interactive mode
- **[`src/llm.rs`](src/llm.rs)** - Core LLM implementation and training logic
- **[`src/transformer.rs`](src/transformer.rs)** - Pre-LN Transformer block with explicit residual connections

## 🏗️ Architecture

The model uses a **Pre-LN Transformer architecture** (GPT-2 standard) with the following components:

```
Input Text → Tokenization (supports Chinese with jieba-rs) → Token Embeddings + Positional Encoding
    ↓
[2x Transformer Blocks] ← Optimized for small datasets
    Each block:
    • LayerNorm → Multi-Head Attention (8 heads) → Dropout → Residual Connection
    • LayerNorm → Feed-Forward Network → Dropout → Residual Connection
    ↓
Output Projection → Softmax → Token Predictions
```

### Why Pre-LN Transformer?

Pre-LN (Layer Normalization before sub-layers) is the modern standard used in GPT-2, GPT-3, and beyond:
- ✅ **More stable training** - Better gradient flow
- ✅ **Faster convergence** - Reduced gradient vanishing/explosion
- ✅ **More robust** - Less sensitive to learning rate

**Architecture Comparison:**

```
Post-LN (Old):                      Pre-LN (Current - GPT-2 Standard):
Input                               Input
  ↓                                   ↓
Attention                           LayerNorm
  ↓                                   ↓
LayerNorm                           Attention
  ↓                                   ↓
Dropout                             Dropout
  ↓                                   ↓
(+Input)                            (+Input) ← Explicit residual
  ↓                                   ↓
FFN                                 LayerNorm
  ↓                                   ↓
LayerNorm                           FFN
  ↓                                   ↓
Dropout                             Dropout
  ↓                                   ↓
Output                              (+X) ← Explicit residual
                                      ↓
                                    Output
```

### Project Structure

```
src/
├── main.rs              # 🎯 Training pipeline and interactive mode
├── llm.rs               # 🧠 Core LLM implementation and training logic
├── lib.rs               # 📚 Library exports and constants
├── transformer.rs       # 🔄 Pre-LN Transformer block with explicit residual connections
├── self_attention.rs    # 👀 Multi-head self-attention mechanism (8 heads)
├── feed_forward.rs      # ⚡ Position-wise feed-forward networks
├── embeddings.rs        # 📊 Token embedding layer with positional encoding
├── output_projection.rs # 🎰 Final linear layer for vocabulary predictions
├── vocab.rs            # 📝 Vocabulary management with optimized jieba-rs tokenization
├── layer_norm.rs       # 🧮 Layer normalization (learnable γ and β)
├── dropout.rs          # 🚫 Dropout regularization (10% rate, inverted dropout)
├── position_encoding.rs # 📍 Sinusoidal position encoding
├── adam.rs             # 🎓 Adam optimizer (β₁=0.9, β₂=0.999)
├── performance_monitor.rs # ⏱️ Performance profiling and timing
└── dataset_loader.rs   # 📁 Training data loading
```

## 🧪 What The Model Learns

The implementation includes training phases that support Chinese:

1. **Pre-training**: Can learn world knowledge from Chinese factual statements
   - "太阳从东方升起，在西方落下"
   - "水由于重力而从高处流向低处"
   - "山脉是高大而多岩石的地形"
   - Enhanced with Chinese cultural knowledge, idioms, and historical facts

2. **Instruction Tuning**: Can learn Chinese conversational patterns
   - "用户：山脉是如何形成的？助手：山脉通过构造力或火山活动在长时间的地质时期内形成..."
   - Handles Chinese greetings, explanations, and follow-up questions
   - Incorporates Chinese cultural references and idioms

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/H-Chris233/RustGPT-Chinese.git
cd RustGPT-Chinese
cargo run

# The model will:
# 1. Build a vocabulary from Chinese training data (with jieba-rs tokenization support)
# 2. Pre-train on Chinese factual statements
# 3. Instruction-tune on Chinese conversational data
# 4. Auto-save checkpoints (best + last)
# 5. Enter interactive mode for testing
```

### Performance Tips

For maximum performance, use release mode:
```bash
cargo build --release
./target/release/llm
```

Release mode enables:
- **Link-time optimization (LTO)** - Cross-crate inlining
- **Maximum optimization level** (opt-level 3)
- **Single codegen unit** - Better optimization opportunities
- **Expected speedup**: 10-20% over debug mode

## 🎮 Interactive Mode

After training, test the model interactively with Chinese:

```
Enter prompt: 山脉是如何形成的?
Model output: 山脉通过构造力或火山活动在长时间的地质时期内形成

Enter prompt: 降雨的原因是什么?
Model output: 降雨是由云中的水蒸气凝结成水滴，当水滴变得太重而无法悬浮在空气中时形成的
```

## 🧭 Recommended Training Entrypoints

| API | Input | Best For | Notes |
|---|---|---|---|
| `train(...)` | Raw text | Minimal teaching path | Shows the smallest complete training loop |
| `train_monitored(...)` | Raw text | Recommended main training path | Includes scheduling, early stopping, monitoring, and gradient accumulation |
| `train_bucketed_sequential(...)` | Raw text | Advanced training path | Uses bucketing and padding masks, but still updates parameters sequentially per sample inside each batch |
| `train_with_checkpointing(...)` | Pre-tokenized data | Resume / long-running training | Adds checkpoint save/load, rollback, and continuity guarantees |

> Note: the public training APIs now use a shared **token-weighted mean loss** convention, so reported `Loss` / `PPL` values are comparable across entrypoints.

## 🧮 Technical Implementation

### Model Configuration
- **Vocabulary Size**: Dynamic (built from training data with jieba-rs integration for Chinese support)
- **Embedding Dimension**: 256 (optimized for small datasets)
- **Hidden Dimension**: 512 (optimized for small datasets)
- **Max Sequence Length**: 128 tokens (optimized for small datasets)
- **Architecture**: 2 Pre-LN Transformer blocks + embeddings + output projection
- **Total Parameters**: ~10M (optimized for limited training data)
- **Training Strategy**: multi-stage training with monitored training, checkpointing, and resume support

### Training Details
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=1e-8) with gradient clipping
- **LR Schedule**: Cosine decay + warmup (no restarts by default)
- **Default LR (main training entry)**: 0.0001
- **Loss Function**: `log_softmax` + NLL (cross entropy from log-probs); PAD (`vocab.pad_token_id()`) is excluded from loss/grads
- **Gradient Clipping**: L2 norm capped at 1.0 on the main training path
- **Regularization**: Dropout layers with 10% rate (inverted dropout)
- **Current training semantics**:
  - Monitored training uses warmup + cosine decay (no restarts by default)
  - Public training APIs report token-weighted mean loss
  - Gradient accumulation is optional and controlled by `train_monitored(...)`
  - Checkpointed training adds save/load, rollback, and resume support

### Key Features
- **Modern Pre-LN Transformer** - GPT-2/3 standard architecture for stable training
- **Explicit Residual Connections** - Clear and maintainable architecture
- **Optimized Chinese tokenization** - jieba-rs with global singleton (50-70% faster)
- **Multi-head self-attention** - 8 heads with optimized reshape operations (20-30% faster)
- **Advanced decoding methods**:
  - Greedy decoding (argmax)
  - Top-k sampling (nucleus sampling)
  - Top-p sampling (cumulative probability)
  - Beam search with log probabilities
  - Temperature scaling for output diversity
- **Gradient clipping** - L2 norm for training stability
- **Modular layer system** - Clean interfaces with Layer trait
- **Comprehensive test coverage** - Unit tests for all components
- **Context window management** - Sliding window for conversation history
- **Performance monitoring** - Detailed timing and profiling tools
- **Compiler optimizations** - LTO, opt-level 3, single codegen unit

### Performance Optimizations

| Optimization | Speedup | Status |
|--------------|---------|--------|
| Jieba singleton (OnceLock) | 50-70% | ✅ Implemented |
| Data preprocessing cache | 20-30% | ✅ v0.3.1 |
| Cosine annealing LR | 15-25%* | ✅ v0.3.1 |
| Early stopping | 10-40%* | ✅ v0.3.1 |
| Gradient accumulation | 40% stability* | ✅ v0.3.1 |
| Attention reshape (slice ops) | 20-30% | ✅ Implemented |
| Compiler optimizations (LTO) | 10-20% | ✅ Implemented |
| Optional BLAS (`--features blas`) | env-dependent | ✅ Implemented |
| **Total expected improvement** | **80-100%** | ✅ Implemented |

*训练质量和稳定性提升，不仅仅是速度优化

## 🔧 Development

```bash
# Run all tests
cargo test

# Test specific components
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test
cargo test --test chinese_tests
cargo test --test vocab_test

# Build optimized version
cargo build --release

# Run with verbose output
cargo test -- --nocapture

# Format code
cargo fmt

# Run linter
cargo clippy
```

## 🧠 Learning Resources

This implementation demonstrates key ML concepts for multilingual language models with Chinese support:
- **Pre-LN Transformer architecture** - Modern standard for stable training
- **Explicit residual connections** - Clear gradient flow management
- **Multi-head attention** - Parallel attention mechanisms
- **Feed-forward networks** - Position-wise transformations
- **Layer normalization** - Per-layer feature normalization
- **Backpropagation** - Automatic differentiation through custom layers
- **Language model training** - Pre-training + fine-tuning
- **Chinese tokenization** - jieba-rs integration and optimization
- **Gradient-based optimization** - Adam optimizer with momentum
- **Context management** - Conversation history tracking
- **Regularization techniques** - Dropout for generalization

Perfect for understanding how LLMs with Chinese support work under the hood!

## 📊 Dependencies

- `ndarray` - N-dimensional arrays for matrix operations (optional BLAS via `--features blas`)
- `jieba-rs` - Chinese text segmentation and tokenization
- `lru` - LRU cache for tokenizer optimization
- `rand` - Random number generation for initialization
- `regex` - Pattern matching for Chinese idioms recognition
- `bincode` - Serialization and binary encoding
- `serde` + `serde_json` - Data serialization
- `log` + `simple_logger` - Logging

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Development guidelines for Claude Code assistant
- **[批量训练与动态掩码](docs/批量训练与动态掩码.md)** - Batch training & dynamic masking notes (CN)
- **[检查点管理与早停机制实现](docs/检查点管理与早停机制实现.md)** - Checkpointing & early stopping (CN)

## 🤝 Contributing

Contributions are welcome! This project is perfect for learning and experimentation.

### High Priority Features Needed
- **🧮 True vectorized batch training** - Batch GEMMs + “accumulate/average grads then one optimizer step” (current `train_bucketed_sequential` is still per-sample SGD inside a batch)
- **📊 Evaluation metrics** - Perplexity, benchmarks, training visualizations
- **🎯 Attention visualization** - Visualize attention patterns for Chinese text
- **📈 Training curves** - Loss/accuracy plotting

### Areas for Improvement
- **Advanced architectures** (Rotary Position Embedding (RoPE), Flash Attention)
- **Training improvements** (strict batch update semantics, mixed precision, better diagnostics)
- **Chinese data handling** (Larger Chinese datasets, streaming data loading)
- **Model analysis** (Attention visualization, gradient analysis, interpretability)

### Current Architecture Status
- ✅ **Pre-LN Transformer** - Modern GPT-2 standard architecture
- ✅ **Explicit residual connections** - Clear and maintainable
- ✅ **Performance optimized** - 60-80% faster than initial version
- ✅ **Padding mask support** - `forward_with_padding_mask(...)` / `forward_batch(..., attention_mask)` (single-sample training does not pad by default)
- ✅ **Gradient accumulation** - Configurable via accumulation steps (default disabled for stability)
- ✅ **Warmup + cosine schedule** - Default training path uses warmup (no restarts)

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/vectorized-batch`
3. Make your changes and add tests
4. Run the test suite: `cargo test`
5. Format and lint: `cargo fmt && cargo clippy`
6. Submit a pull request with a clear description

### Code Style
- Follow standard Rust conventions (`cargo fmt`)
- Add comprehensive tests for new features
- Update documentation and README as needed
- Keep the "from scratch" philosophy - avoid heavy ML dependencies
- Focus on Chinese language processing improvements
- Add comments explaining complex algorithms

### Ideas for Contributions
- 🚀 **Beginner**: More Chinese training data, config files, tests/docs fixes
- 🔥 **Intermediate**: True vectorized batch training, evaluation metrics, training visualization
- ⚡ **Advanced**: Flash Attention, RoPE, mixed precision training

Questions? Open an issue or start a discussion!

## 📜 License

This project is open source and available for educational purposes.

---

**Built with 🦀 Rust and ❤️ for understanding Chinese LLMs**

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!
