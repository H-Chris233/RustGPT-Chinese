/// 自注意力性能基准测试
///
/// 验证掩码缓存和优化矩阵乘法的性能提升
use llm::{EMBEDDING_DIM, Layer, self_attention::SelfAttention};
use ndarray::Array2;
use std::time::Instant;

#[test]
#[ignore]
fn benchmark_mask_caching() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    let seq_len = 64;
    let input = Array2::ones((seq_len, EMBEDDING_DIM));

    // 预热
    for _ in 0..3 {
        let _ = self_attention.forward(&input);
    }

    // 基准测试：测量多次前向传播的平均时间
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = self_attention.forward(&input);
    }
    let duration = start.elapsed();
    let avg_time = duration / iterations;

    println!("序列长度 {}: 平均前向传播时间 = {:?}", seq_len, avg_time);
    println!("掩码缓存条目数: {}", self_attention.causal_mask_cache.len());

    // 验证性能合理性（应该在毫秒级别）
    assert!(
        avg_time.as_millis() < 100,
        "前向传播时间过长: {:?}",
        avg_time
    );
}

#[test]
#[ignore]
fn benchmark_different_sequence_lengths() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    let seq_lengths = [8, 16, 32, 64, 128];

    println!("\n=== 不同序列长度的性能基准 ===");

    for &seq_len in &seq_lengths {
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // 预热
        for _ in 0..3 {
            let _ = self_attention.forward(&input);
        }

        // 基准测试
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self_attention.forward(&input);
        }
        let duration = start.elapsed();
        let avg_time = duration / iterations;

        println!("序列长度 {:3}: {:8.2?} 平均", seq_len, avg_time);
    }

    println!("总缓存条目数: {}", self_attention.causal_mask_cache.len());
    assert_eq!(self_attention.causal_mask_cache.len(), seq_lengths.len());
}

#[test]
#[ignore]
fn benchmark_numerical_stability() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    let seq_len = 32;

    // 测试不同数值范围的输入
    let test_cases = vec![
        ("小数值", 0.001),
        ("中等数值", 1.0),
        ("大数值", 100.0),
        ("极大数值", 1000.0),
    ];

    println!("\n=== 数值稳定性基准 ===");

    for (name, scale) in test_cases {
        let input = Array2::ones((seq_len, EMBEDDING_DIM)) * scale;

        let iterations = 20;
        let start = Instant::now();
        let mut all_finite = true;

        for _ in 0..iterations {
            let output = self_attention.forward(&input);
            all_finite &= output.iter().all(|&v| v.is_finite());
        }

        let duration = start.elapsed();
        let avg_time = duration / iterations;

        println!(
            "{:12}: {:8.2?} 平均, 数值稳定: {}",
            name,
            avg_time,
            if all_finite { "✓" } else { "✗" }
        );

        assert!(all_finite, "{} 情况下输出包含非有限值", name);
    }
}

#[test]
#[ignore]
fn benchmark_gradient_computation() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    let seq_len = 32;
    let input = Array2::ones((seq_len, EMBEDDING_DIM));

    println!("\n=== 前向/反向传播基准 ===");

    // 预热
    for _ in 0..3 {
        let output = self_attention.forward(&input);
        let grad = Array2::ones(output.dim());
        let _ = self_attention.backward(&grad, 0.001);
    }

    // 基准测试
    let iterations = 50;

    // 前向传播
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = self_attention.forward(&input);
    }
    let forward_time = start.elapsed() / iterations;

    // 前向+反向传播
    let start = Instant::now();
    for _ in 0..iterations {
        let output = self_attention.forward(&input);
        let grad = Array2::ones(output.dim());
        let _ = self_attention.backward(&grad, 0.001);
    }
    let total_time = start.elapsed() / iterations;
    let backward_time = total_time - forward_time;

    println!("前向传播: {:8.2?}", forward_time);
    println!("反向传播: {:8.2?}", backward_time);
    println!("总计时间: {:8.2?}", total_time);
    println!(
        "反向/前向比: {:.2}x",
        backward_time.as_nanos() as f64 / forward_time.as_nanos() as f64
    );
}

#[test]
#[ignore]
fn benchmark_cache_hit_rate() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 模拟真实使用场景：重复使用相同的序列长度
    let common_lengths = [10, 20, 30];
    let iterations = 100;

    println!("\n=== 缓存命中率基准 ===");

    let start = Instant::now();
    for i in 0..iterations {
        let seq_len = common_lengths[i % common_lengths.len()];
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        let _ = self_attention.forward(&input);
    }
    let duration = start.elapsed();

    println!("总迭代次数: {}", iterations);
    println!("唯一序列长度: {}", common_lengths.len());
    println!("缓存条目数: {}", self_attention.causal_mask_cache.len());
    println!("总时间: {:?}", duration);
    println!("平均时间/迭代: {:?}", duration / (iterations as u32));

    // 验证所有长度都被缓存
    assert_eq!(self_attention.causal_mask_cache.len(), common_lengths.len());
}
