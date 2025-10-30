//! # 混合精度训练验证脚本
//!
//! 对比 FP32、FP16、BF16 三种精度下的训练效果，验证混合精度系统的正确性。
//!
//! ## 测试场景
//!
//! 1. **小模型配置**：2-layer Transformer，简化词汇表
//! 2. **三种精度对比**：FP32（baseline）、FP16、BF16
//! 3. **训练指标**：Loss 曲线、Perplexity、训练速度
//! 4. **验收标准**：
//!    - Loss 不发散（与 FP32 差异 < 5%）
//!    - Perplexity 差异 < 3%
//!    - 无严重溢出（溢出率 < 10%）
//!
//! ## 运行方法
//!
//! ```bash
//! cargo run --example mixed_precision_test
//! ```

use llm::{
    embeddings::Embeddings, output_projection::OutputProjection, transformer::TransformerBlock,
    vocab::Vocab, MixedPrecisionConfig, MixedPrecisionTrainer, LLM, EMBEDDING_DIM, HIDDEN_DIM,
};
use simple_logger::SimpleLogger;
use std::time::Instant;

/// 创建小型测试模型（使用默认配置，2层 Transformer）
fn create_small_model(vocab: &Vocab) -> LLM {
    let vocab_size = vocab.words.len();

    let embeddings = Embeddings::new(vocab.clone());
    let transformer_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    LLM::new(
        vocab.clone(),
        vec![
            Box::new(embeddings),
            Box::new(transformer_1),
            Box::new(transformer_2),
            Box::new(output_proj),
        ],
    )
}

/// 生成简单的训练数据
fn generate_test_data(vocab: &Vocab) -> Vec<Vec<usize>> {
    let sentences = vec![
        "我 喜欢 编程",
        "深度 学习 很 有趣",
        "Rust 是 一门 优秀 的 语言",
        "神经 网络 可以 学习 模式",
        "训练 模型 需要 数据",
        "机器 学习 改变 世界",
        "人工 智能 的 未来",
        "算法 优化 很 重要",
        "测试 混合 精度 训练",
        "验证 数值 稳定性",
    ];

    sentences.iter().map(|s| vocab.encode_sequence(s)).collect()
}

/// 计算 Perplexity
fn calculate_perplexity(avg_loss: f32) -> f32 {
    avg_loss.exp()
}

/// 运行单次训练实验
fn run_experiment(
    precision_name: &str,
    config: MixedPrecisionConfig,
    vocab: &Vocab,
    tokenized_data: &[Vec<usize>],
    epochs: usize,
    initial_lr: f32,
) -> (f32, f32, f64, f32) {
    println!("\n{}", "=".repeat(60));
    println!("开始实验: {}", precision_name);
    println!("{}", "=".repeat(60));

    let mut model = create_small_model(vocab);
    let mut trainer = MixedPrecisionTrainer::new(config);

    let start_time = Instant::now();

    let actual_epochs = trainer.train_monitored(
        &mut model,
        tokenized_data.to_vec(),
        epochs,
        initial_lr,
        50, // patience (不会触发早停，因为只训练100步)
    );

    let training_time = start_time.elapsed().as_secs_f64();

    // 获取最终统计
    let (total_overflows, total_steps, overflow_rate) = trainer.scaler_stats();

    // 评估最终损失
    model.set_training_mode(false);
    let mut total_loss = 0.0;
    let mut count = 0;

    for row in tokenized_data {
        if row.len() < 2 {
            continue;
        }
        let input_ids = &row[..row.len() - 1];
        let target_ids = &row[1..];

        let mut input = ndarray::Array2::zeros((1, input_ids.len()));
        input.row_mut(0).assign(
            &input_ids
                .iter()
                .map(|&x| x as f32)
                .collect::<ndarray::Array1<f32>>(),
        );

        for layer in &mut model.network {
            input = layer.forward(&input);
        }

        let probs = llm::utils::softmax(&input);
        total_loss += LLM::cross_entropy_loss_step(&probs, target_ids);
        count += 1;
    }

    let final_loss = total_loss / count as f32;
    let perplexity = calculate_perplexity(final_loss);

    println!("\n实验结果:");
    println!("  - 实际训练 epoch 数: {}", actual_epochs);
    println!("  - 最终损失: {:.4}", final_loss);
    println!("  - Perplexity: {:.4}", perplexity);
    println!("  - 训练时间: {:.2}s", training_time);
    println!(
        "  - 溢出统计: {}/{} ({:.2}%)",
        total_overflows,
        total_steps,
        overflow_rate * 100.0
    );
    println!("  - 是否回退: {}", trainer.is_fallback_triggered());
    println!("  - 最终精度: {}", trainer.current_precision());

    (final_loss, perplexity, training_time, overflow_rate)
}

fn main() {
    // 初始化日志
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .unwrap();

    println!("\n混合精度训练验证测试");
    println!("{}\n", "=".repeat(60));

    // 构建小词汇表
    let test_texts = vec![
        "我 喜欢 编程".to_string(),
        "深度 学习 很 有趣".to_string(),
        "Rust 是 一门 优秀 的 语言".to_string(),
        "神经 网络 可以 学习 模式".to_string(),
        "训练 模型 需要 数据".to_string(),
        "机器 学习 改变 世界".to_string(),
        "人工 智能 的 未来".to_string(),
        "算法 优化 很 重要".to_string(),
        "测试 混合 精度 训练".to_string(),
        "验证 数值 稳定性".to_string(),
    ];

    let vocab = Vocab::build_from_texts(&test_texts);
    println!("词汇表大小: {}", vocab.words.len());

    // 生成训练数据
    let tokenized_data = generate_test_data(&vocab);
    println!("训练样本数: {}", tokenized_data.len());

    let epochs = 100;
    let initial_lr = 0.001;

    // === 实验 1: FP32 Baseline ===
    let (fp32_loss, fp32_ppl, fp32_time, _) = run_experiment(
        "FP32 (Baseline)",
        MixedPrecisionConfig::disabled(),
        &vocab,
        &tokenized_data,
        epochs,
        initial_lr,
    );

    // === 实验 2: FP16 ===
    let (fp16_loss, fp16_ppl, fp16_time, fp16_overflow) = run_experiment(
        "FP16",
        MixedPrecisionConfig::fp16(),
        &vocab,
        &tokenized_data,
        epochs,
        initial_lr,
    );

    // === 实验 3: BF16 ===
    let (bf16_loss, bf16_ppl, bf16_time, bf16_overflow) = run_experiment(
        "BF16",
        MixedPrecisionConfig::bf16(),
        &vocab,
        &tokenized_data,
        epochs,
        initial_lr,
    );

    // === 对比分析 ===
    println!("\n{}", "=".repeat(60));
    println!("对比分析");
    println!("{}\n", "=".repeat(60));

    println!("损失对比:");
    println!("  FP32:  {:.4}", fp32_loss);
    println!(
        "  FP16:  {:.4} (差异: {:.2}%)",
        fp16_loss,
        ((fp16_loss - fp32_loss) / fp32_loss * 100.0).abs()
    );
    println!(
        "  BF16:  {:.4} (差异: {:.2}%)",
        bf16_loss,
        ((bf16_loss - fp32_loss) / fp32_loss * 100.0).abs()
    );

    println!("\nPerplexity 对比:");
    println!("  FP32:  {:.4}", fp32_ppl);
    println!(
        "  FP16:  {:.4} (差异: {:.2}%)",
        fp16_ppl,
        ((fp16_ppl - fp32_ppl) / fp32_ppl * 100.0).abs()
    );
    println!(
        "  BF16:  {:.4} (差异: {:.2}%)",
        bf16_ppl,
        ((bf16_ppl - fp32_ppl) / fp32_ppl * 100.0).abs()
    );

    println!("\n训练时间对比:");
    println!("  FP32:  {:.2}s", fp32_time);
    println!(
        "  FP16:  {:.2}s (速度: {:.2}x)",
        fp16_time,
        fp32_time / fp16_time
    );
    println!(
        "  BF16:  {:.2}s (速度: {:.2}x)",
        bf16_time,
        fp32_time / bf16_time
    );

    println!("\n溢出率统计:");
    println!("  FP16:  {:.2}%", fp16_overflow * 100.0);
    println!("  BF16:  {:.2}%", bf16_overflow * 100.0);

    // === 验收判断 ===
    println!("\n{}", "=".repeat(60));
    println!("验收判断");
    println!("{}\n", "=".repeat(60));

    let loss_threshold = 0.05; // 5%
    let ppl_threshold = 0.03; // 3%
    let overflow_threshold = 0.10; // 10%

    let fp16_loss_diff = ((fp16_loss - fp32_loss) / fp32_loss).abs();
    let bf16_loss_diff = ((bf16_loss - fp32_loss) / fp32_loss).abs();
    let fp16_ppl_diff = ((fp16_ppl - fp32_ppl) / fp32_ppl).abs();
    let bf16_ppl_diff = ((bf16_ppl - fp32_ppl) / fp32_ppl).abs();

    let fp16_loss_ok = fp16_loss_diff < loss_threshold;
    let bf16_loss_ok = bf16_loss_diff < loss_threshold;
    let fp16_ppl_ok = fp16_ppl_diff < ppl_threshold;
    let bf16_ppl_ok = bf16_ppl_diff < ppl_threshold;
    let fp16_overflow_ok = fp16_overflow < overflow_threshold;
    let bf16_overflow_ok = bf16_overflow < overflow_threshold;

    println!(
        "✓ FP16 Loss 稳定性:     {} ({:.2}% < {:.0}%)",
        if fp16_loss_ok { "通过" } else { "失败" },
        fp16_loss_diff * 100.0,
        loss_threshold * 100.0
    );
    println!(
        "✓ BF16 Loss 稳定性:     {} ({:.2}% < {:.0}%)",
        if bf16_loss_ok { "通过" } else { "失败" },
        bf16_loss_diff * 100.0,
        loss_threshold * 100.0
    );
    println!(
        "✓ FP16 Perplexity:     {} ({:.2}% < {:.0}%)",
        if fp16_ppl_ok { "通过" } else { "失败" },
        fp16_ppl_diff * 100.0,
        ppl_threshold * 100.0
    );
    println!(
        "✓ BF16 Perplexity:     {} ({:.2}% < {:.0}%)",
        if bf16_ppl_ok { "通过" } else { "失败" },
        bf16_ppl_diff * 100.0,
        ppl_threshold * 100.0
    );
    println!(
        "✓ FP16 溢出率:         {} ({:.2}% < {:.0}%)",
        if fp16_overflow_ok { "通过" } else { "失败" },
        fp16_overflow * 100.0,
        overflow_threshold * 100.0
    );
    println!(
        "✓ BF16 溢出率:         {} ({:.2}% < {:.0}%)",
        if bf16_overflow_ok { "通过" } else { "失败" },
        bf16_overflow * 100.0,
        overflow_threshold * 100.0
    );

    let all_passed = fp16_loss_ok
        && bf16_loss_ok
        && fp16_ppl_ok
        && bf16_ppl_ok
        && fp16_overflow_ok
        && bf16_overflow_ok;

    println!("\n{}", "=".repeat(60));
    if all_passed {
        println!("✅ 所有测试通过！混合精度训练系统验证成功。");
    } else {
        println!("❌ 部分测试失败，请检查混合精度实现。");
    }
    println!("{}\n", "=".repeat(60));

    // 输出 CSV 格式数据（可用于绘图）
    println!("\nCSV 格式输出（用于可视化）:");
    println!("Precision,Loss,Perplexity,Time(s),Overflow(%)");
    println!(
        "FP32,{:.4},{:.4},{:.2},0.00",
        fp32_loss, fp32_ppl, fp32_time
    );
    println!(
        "FP16,{:.4},{:.4},{:.2},{:.2}",
        fp16_loss,
        fp16_ppl,
        fp16_time,
        fp16_overflow * 100.0
    );
    println!(
        "BF16,{:.4},{:.4},{:.2},{:.2}",
        bf16_loss,
        bf16_ppl,
        bf16_time,
        bf16_overflow * 100.0
    );
}
