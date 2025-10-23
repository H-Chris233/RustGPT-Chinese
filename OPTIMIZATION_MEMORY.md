# 内存与缓存复用优化

本文档记录了针对 RustGPT-Chinese 项目的内存分配和缓存复用优化。

## 优化概览

### 1. Dataset::new 去除双重 clone
**文件**: `src/dataset_loader.rs`

**问题**: 旧的多格式实现读取完数据后仍然在构造 `Dataset` 时执行 `clone()`，导致 `Vec<String>` 被复制两次。

**优化**: 简化为纯 JSON 数据路径后，`Dataset::new` 直接调用 `get_data_from_json` 并返回所有权：
```rust
pub fn new(pretraining_data_path: String, chat_training_data_path: String) -> Self {
    let pretraining_data = get_data_from_json(&pretraining_data_path);
    let chat_training_data = get_data_from_json(&chat_training_data_path);

    Dataset {
        pretraining_data,
        chat_training_data,
    }
}
```

**效果**:
- 移除 2 次完整的 `Vec<String>` 深拷贝
- 数据加载逻辑更简单，避免了额外的多格式处理分支

---

### 2. Embeddings 位置编码缓存复用
**文件**: `src/embeddings.rs`

**问题**:
原代码在每次 `forward` 调用时都重新分配 `Array2` 来存储位置编码：
```rust
// 优化前
pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
    let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
    
    // 每次都分配新的 Array2 (seq_len × EMBEDDING_DIM)
    let mut position_embeds = Array2::<f32>::zeros((token_ids.len(), EMBEDDING_DIM));
    Zip::indexed(&mut position_embeds).par_for_each(|(i, j), value| {
        *value = self.position_encoder.get_encoding(i, j);
    });
    
    token_embeds + position_embeds
}
```

**优化**:
1. 在 `Embeddings` 结构体中添加预分配的缓存字段：
```rust
pub struct Embeddings {
    // ... 其他字段
    pub position_cache: Array2<f32>,  // 新增：预分配缓冲区
}
```

2. 直接从预生成的位置编码矩阵中 slice：
```rust
// 优化后
pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
    let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
    
    // 直接 slice，避免分配
    let seq_len = token_ids.len();
    let position_embeds = self.position_encoder.encoding.slice(ndarray::s![0..seq_len, ..]);
    
    token_embeds + position_embeds
}
```

**效果**:
- 每次 `forward` 调用减少 1 次 `Array2<f32>` 分配
- 对于典型的训练序列（seq_len=32, EMBEDDING_DIM=256），每次节省 32KB 内存分配
- 在训练 500 epochs × 200 samples = 100,000 次调用中，累计减少约 3.2 GB 分配

---

### 3. 采样方法缓冲区复用
**文件**: `src/llm.rs`

**问题**:
`top_k_sampling` 和 `top_p_sampling` 方法在每次调用时分配多个临时 Vec：
```rust
// 优化前
fn top_k_sampling(&self, probs: &Array2<f32>, k: usize) -> Vec<usize> {
    for row in probs.rows() {
        let mut prob_idx_pairs: Vec<(f32, usize)> = row  // 每次分配
            .iter().enumerate().map(|(idx, &prob)| (prob, idx)).collect();
        prob_idx_pairs.sort_by(...);
        
        let mut top_k_probs = vec![0.0; self.vocab.words.len()];  // 每次分配
        // ...
    }
}
```

**优化**:
1. 在 `LLM` 结构体中添加可重用缓冲区：
```rust
pub struct LLM {
    // ... 其他字段
    pub sampling_prob_buffer: Vec<f32>,
    pub sampling_idx_buffer: Vec<(f32, usize)>,
    pub beam_candidates_buffer: Vec<(Vec<usize>, f32)>,
}
```

2. 复用缓冲区：
```rust
// 优化后
fn top_k_sampling(&mut self, probs: &Array2<f32>, k: usize) -> Vec<usize> {
    for row in probs.rows() {
        self.sampling_idx_buffer.clear();  // 清空而非重新分配
        self.sampling_idx_buffer.extend(...);
        self.sampling_idx_buffer.sort_by(...);
        
        self.sampling_prob_buffer.clear();  // 复用
        self.sampling_prob_buffer.resize(self.vocab.words.len(), 0.0);
        // ...
    }
}
```

**效果**:
- 每次采样调用减少 2 次 `Vec` 分配（`prob_idx_pairs` + `top_k_probs`）
- 对于 vocab_size=30000，每次节省约 240 KB 分配
- 推理时每生成一个 token 调用一次，减少频繁的内存分配/释放

---

### 4. Beam Search 候选缓冲区复用
**文件**: `src/llm.rs`

**问题**:
Beam search 在每次迭代中分配新的候选列表：
```rust
// 优化前
fn beam_search(&mut self, ...) -> String {
    for _ in initial_tokens.len()..max_length {
        let mut candidates = Vec::new();  // 每次迭代都分配
        for (seq, log_prob) in &current_beams {
            // ... 生成候选
            candidates.push((new_seq, new_log_prob));
        }
        candidates.sort_by(...);
        current_beams = candidates.into_iter().take(beam_width).collect();
    }
}
```

**优化**:
复用 `beam_candidates_buffer`：
```rust
// 优化后
fn beam_search(&mut self, ...) -> String {
    for _ in initial_tokens.len()..max_length {
        self.beam_candidates_buffer.clear();  // 复用缓冲区
        for (seq, log_prob) in &current_beams {
            // ... 生成候选
            self.beam_candidates_buffer.push((new_seq, new_log_prob));
        }
        self.beam_candidates_buffer.sort_by(...);
        current_beams = self.beam_candidates_buffer.iter().take(beam_width).cloned().collect();
    }
}
```

**效果**:
- 每次 beam search 迭代减少 1 次 `Vec<(Vec<usize>, f32)>` 分配
- Beam width=3, max_length=20 的搜索中，减少约 20 次分配

---

## 性能基准测试

运行基准测试：
```bash
cargo run --bin memory_optimization_bench --release
```

基准测试会验证：
1. Dataset 加载性能（去除 clone）
2. Embeddings forward 性能（位置编码缓存复用）
3. 推理方法性能（采样缓冲区复用）
4. Beam search 性能（候选缓冲区复用）

预期改进：
- **内存分配次数**: 在典型训练场景（500 epochs × 200 samples）中减少约 100,000 次分配
- **峰值内存**: 减少约 5-10% 
- **训练速度**: 提升约 2-5%（通过减少分配器压力和提升缓存局部性）

---

## 深度分析工具

### 1. 使用 Valgrind/Heaptrack 分析内存
```bash
# 安装 heaptrack
sudo apt install heaptrack

# 运行分析
heaptrack cargo run --release

# 查看报告
heaptrack_gui heaptrack.cargo.*.zst
```

### 2. 使用 cargo-flamegraph 分析性能
```bash
# 安装 flamegraph
cargo install flamegraph

# 生成火焰图
sudo cargo flamegraph --bin llm

# 查看 flamegraph.svg
```

### 3. 使用 dhat 分析堆分配
在 `Cargo.toml` 中添加：
```toml
[dependencies]
dhat = "0.3"
```

在 `main.rs` 开头添加：
```rust
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
```

---

## 未来优化方向

1. **对象池模式**: 为频繁分配的 `Array2` 引入对象池
2. **arena 分配器**: 使用 bump allocator 减少训练时的内存碎片
3. **零拷贝优化**: 在可能的地方使用 `Cow` 或引用避免拷贝
4. **SIMD 优化**: 在位置编码和 softmax 计算中使用 SIMD 指令
5. **批处理**: 支持 batch forward，提升 GPU 利用率（未来扩展）

---

## 注意事项

- 这些优化专注于**减少内存分配次数**和**提升缓存局部性**
- 对于小规模训练（< 100 epochs），改进可能不明显
- 在大规模训练（数千 epochs，大数据集）时效果显著
- 优化不影响模型精度，只改进运行时性能

---

## 相关文档

- [CHANGELOG.md](./CHANGELOG.md) - 版本更新记录
- [CLAUDE.md](./CLAUDE.md) - 项目架构文档
- [README.md](./README.md) - 项目总览

---

最后更新: 2024-10 (RustGPT-Chinese v0.3.1 内存优化专项)
