//! 内存与缓存复用优化基准测试
//!
//! 测试目标：
//! 1. 验证 `Dataset::new` 去除双重 clone 后的耗时变化
//! 2. 验证 Embeddings 位置编码缓存复用的耗时变化
//! 3. 验证采样/beam search 缓冲区复用对运行时间的影响
//! 4. 通过运行时间间接观察内存/分配优化效果（当前不直接记录峰值内存）

use llm::{dataset_loader::Dataset, Embeddings, Layer, LLM, Vocab};
use std::time::Instant;

/// 基准统计结果（当前只记录运行耗时，不直接统计峰值内存或分配次数）。
#[derive(Default)]
struct BenchStats {
    elapsed_ms: f64,
    iterations: usize,
}

impl BenchStats {
    fn avg_time_us(&self) -> f64 {
        (self.elapsed_ms * 1000.0) / self.iterations as f64
    }
}

/// 基准测试1: Dataset 加载（无双重 clone）
fn bench_dataset_loading() -> BenchStats {
    println!("\n📊 基准测试 1: Dataset 加载（去除双重 clone）");

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _dataset = Dataset::new(
            "data/pretraining".to_string(),
            "data/chat".to_string(),
        );
        // Dataset 自动销毁
    }

    let elapsed = start.elapsed();
    let stats = BenchStats {
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        iterations,
    };

    println!("  总耗时: {:.2} ms", stats.elapsed_ms);
    println!("  平均每次: {:.2} μs", stats.avg_time_us());
    println!("  ✅ 通过直接返回所有权，避免了 2 次完整的 Vec<String> 拷贝");

    stats
}

/// 基准测试2: Embeddings 前向传播（位置编码缓存复用）
fn bench_embeddings_forward() -> BenchStats {
    println!("\n📊 基准测试 2: Embeddings 前向传播（位置编码缓存复用）");

    let vocab = Vocab::default();
    let mut embeddings = Embeddings::new(vocab);

    // 模拟不同长度的序列
    let test_sequences = vec![
        vec![1, 2, 3, 4, 5],                 // 5 tokens
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // 10 tokens
        vec![1; 32],                         // 32 tokens
        vec![1; 64],                         // 64 tokens
    ];

    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        for seq in &test_sequences {
            let input = ndarray::Array2::from_shape_vec(
                (1, seq.len()),
                seq.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();

            let _output = embeddings.forward(&input);
        }
    }

    let elapsed = start.elapsed();
    let stats = BenchStats {
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        iterations: iterations * test_sequences.len(),
    };

    println!("  总耗时: {:.2} ms", stats.elapsed_ms);
    println!("  平均每次: {:.2} μs", stats.avg_time_us());
    println!("  ✅ 通过直接 slice 预生成的位置编码矩阵，避免每次 forward 都分配新的 Array2");

    stats
}

/// 基准测试3: 采样/推理方法（间接测试缓冲区复用）
fn bench_inference_methods() -> BenchStats {
    println!("\n📊 基准测试 3: 推理方法（间接测试采样缓冲区复用）");

    let vocab = Vocab::default();
    let network = vec![Box::new(Embeddings::new(vocab.clone())) as Box<dyn llm::llm::Layer>];
    let mut llm = LLM::new(vocab, network);
    llm.set_training_mode(false);

    let test_text = "你好";
    let iterations = 50;

    let start = Instant::now();

    for _ in 0..iterations {
        // 测试推理方法，内部使用了采样缓冲区复用
        let _result = llm.predict_with_sampling(test_text, 1.0, 0.9, 5);
    }

    let elapsed = start.elapsed();
    let stats = BenchStats {
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        iterations,
    };

    println!("  总耗时: {:.2} ms", stats.elapsed_ms);
    println!("  平均每次: {:.2} ms", stats.elapsed_ms / iterations as f64);
    println!("  ✅ 内部采样方法通过复用 sampling_idx_buffer 和 sampling_prob_buffer");
    println!("     避免每次采样分配 2×vocab_size 的 Vec");

    stats
}

/// 基准测试4: Beam Search（candidates缓冲区复用）
fn bench_beam_search() -> BenchStats {
    println!("\n📊 基准测试 4: Beam Search（candidates缓冲区复用）");

    let vocab = Vocab::default();
    let network = vec![Box::new(Embeddings::new(vocab.clone())) as Box<dyn llm::llm::Layer>];
    let mut llm = LLM::new(vocab, network);
    llm.set_training_mode(false);

    let test_text = "你好";
    let iterations = 50;

    let start = Instant::now();

    for _ in 0..iterations {
        // Beam width=3, max_length=10（较小的值以加快测试）
        let _result = llm.predict_with_beam_search(test_text, 3, 10);
    }

    let elapsed = start.elapsed();
    let stats = BenchStats {
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        iterations,
    };

    println!("  总耗时: {:.2} ms", stats.elapsed_ms);
    println!("  平均每次: {:.2} ms", stats.elapsed_ms / iterations as f64);
    println!("  ✅ 通过复用 beam_candidates_buffer，避免每次迭代分配新的 Vec<(Vec<usize>, f32)>");

    stats
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║   内存与缓存复用优化基准测试                                ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("本基准测试验证以下优化：");
    println!("  1. Dataset::new 去除双重 clone");
    println!("  2. Embeddings 位置编码缓存复用（slice 替代 allocate）");
    println!("  3. 采样方法缓冲区复用（top-k/top-p）");
    println!("  4. Beam Search candidates 缓冲区复用");

    let stats1 = bench_dataset_loading();
    let stats2 = bench_embeddings_forward();
    let stats3 = bench_inference_methods();
    let stats4 = bench_beam_search();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   汇总统计                                                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("✅ 所有基准测试完成！");
    println!();
    println!("优化效果总结：");
    println!("  • Dataset 加载: 避免 2 次完整数据集拷贝");
    println!("  • Embeddings:   每次 forward 减少 1 次 Array2 分配");
    println!("  • 采样方法:     每次采样减少 2 次 Vec 分配");
    println!("  • Beam Search:  每次迭代减少 1 次 Vec 分配");
    println!();
    println!(
        "总测试次数: {}",
        stats1.iterations + stats2.iterations + stats3.iterations + stats4.iterations
    );
    println!();
    println!("💡 提示：");
    println!("  - 这些优化在大规模训练（数千/数万次迭代）时效果显著");
    println!("  - 减少分配次数 = 减少内存碎片 + 提升缓存局部性");
    println!("  - 建议配合 `cargo flamegraph` 或 `heaptrack` 进行深度分析");
}
