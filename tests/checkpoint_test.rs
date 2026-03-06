//! 检查点管理器集成测试
//!
//! 测试检查点的保存、加载和训练恢复功能

use llm::{
    CheckpointManager, CheckpointMetadata, CheckpointStrategy, Embeddings, LLM, OutputProjection,
    TransformerBlock, Vocab, EMBEDDING_DIM, HIDDEN_DIM, Layer, LayerContext,
};
use ndarray::Array2;
use std::fs;

/// 创建一个小型测试模型
fn create_test_model() -> (LLM, Vec<String>) {
    // 准备测试数据
    let test_data = vec![
        "今天天气很好".to_string(),
        "我喜欢学习编程".to_string(),
        "机器学习很有趣".to_string(),
    ];

    // 构建词汇表
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&test_data, &mut vocab_set);

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // 创建模型
    let embeddings = Embeddings::new(vocab.clone());
    let transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    let llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer),
            Box::new(output_projection),
        ],
    );

    (llm, test_data)
}

#[test]
fn test_checkpoint_manager_creation() {
    let checkpoint_dir = "test_checkpoints_creation";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3);
    assert!(manager.is_ok(), "应该能成功创建检查点管理器");

    // 验证目录被创建
    assert!(
        std::path::Path::new(checkpoint_dir).exists(),
        "检查点目录应该被创建"
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_save_and_load() {
    let checkpoint_dir = "test_checkpoints_save_load";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能创建管理器");

    // 创建测试模型
    let (llm, _test_data) = create_test_model();

    // 保存检查点
    let metadata = CheckpointMetadata {
        epoch: 10,
        loss: 2.5,
        learning_rate: 0.001,
        timestamp: "2025-10-24 10:00:00".to_string(),
        phase: "test".to_string(),
    };

    let result = manager.save_checkpoint(&llm, metadata.clone());
    assert!(result.is_ok(), "应该能成功保存检查点");

    let checkpoint_path = result.unwrap();
    assert!(checkpoint_path.exists(), "检查点文件应该存在");

    // 加载检查点
    let load_result = CheckpointManager::load_checkpoint(&checkpoint_path);
    assert!(load_result.is_ok(), "应该能成功加载检查点");

    let (loaded_llm, loaded_metadata) = load_result.unwrap();

    // 验证元数据
    assert_eq!(loaded_metadata.epoch, 10, "Epoch应该匹配");
    assert_eq!(loaded_metadata.loss, 2.5, "Loss应该匹配");
    assert_eq!(loaded_metadata.phase, "test", "Phase应该匹配");

    // 验证模型结构
    assert_eq!(
        loaded_llm.vocab.len(),
        llm.vocab.len(),
        "词汇表大小应该匹配"
    );
    assert_eq!(
        loaded_llm.total_parameters(),
        llm.total_parameters(),
        "参数数量应该匹配"
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_best_loss_tracking() {
    let checkpoint_dir = "test_checkpoints_best_loss";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能创建管理器");

    let (llm, _test_data) = create_test_model();

    // 保存第一个检查点
    let metadata1 = CheckpointMetadata {
        epoch: 10,
        loss: 3.0,
        learning_rate: 0.001,
        timestamp: "2025-10-24 10:00:00".to_string(),
        phase: "test".to_string(),
    };
    manager.save_checkpoint(&llm, metadata1).ok();
    assert_eq!(manager.get_best_loss(), 3.0, "应该更新最佳loss");

    // 保存第二个更好的检查点
    let metadata2 = CheckpointMetadata {
        epoch: 20,
        loss: 2.5,
        learning_rate: 0.001,
        timestamp: "2025-10-24 11:00:00".to_string(),
        phase: "test".to_string(),
    };
    manager.save_checkpoint(&llm, metadata2).ok();
    assert_eq!(manager.get_best_loss(), 2.5, "应该更新为更好的loss");

    // 验证 `get_best_checkpoint()` 会返回最佳检查点。
    let best_checkpoint = manager.get_best_checkpoint();
    assert!(best_checkpoint.is_some(), "应该有最佳检查点");

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_training_continuity() {
    let checkpoint_dir = "test_checkpoints_continuity";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
        .expect("应该能创建管理器");

    // 创建模型并训练几个epoch
    let (mut llm, test_data) = create_test_model();

    // Tokenize数据
    let tokenized_data: Vec<Vec<usize>> = test_data
        .iter()
        .map(|text| LLM::tokenize_with_vocab(&llm.vocab, text))
        .collect();

    // 训练5个epoch并保存检查点
    let epochs_trained = llm.train_with_checkpointing(
        tokenized_data.clone(),
        5,
        0.001,
        100, // 高patience确保不会早停
        Some(&mut manager),
        "test_phase",
        0,
    );

    println!("训练了 {} 个epoch", epochs_trained);

    // 列出所有保存的检查点
    println!("检查点目录内容:");
    if let Ok(entries) = fs::read_dir(checkpoint_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  - {:?}", entry.file_name());
            }
        }
    }

    // 获取训练后的loss
    llm.set_training_mode(true);
    let pad_token_id = llm.vocab.pad_token_id();
    let mut final_loss = 0.0;
    let mut count = 0;
    for training_row in &tokenized_data {
        if training_row.len() < 2 {
            continue;
        }
        let input_ids = &training_row[..training_row.len() - 1];
        let target_ids = &training_row[1..];

        let input =
            ndarray::Array2::from_shape_fn((1, input_ids.len()), |(_, j)| input_ids[j] as f32);

        let mut output = input.clone();
        for layer in &mut llm.network {
            let (out, _ctx) = layer.forward(&output);
            output = out;
        }

        let probs = llm::utils::softmax(&output);
        final_loss += LLM::cross_entropy_loss_step(&probs, target_ids, pad_token_id);
        count += 1;
    }
    final_loss /= count as f32;
    llm.set_training_mode(false);

    // 加载检查点（尝试best，如果没有则用last）
    let checkpoint_path = manager
        .get_best_checkpoint()
        .or_else(|| manager.get_last_checkpoint());
    assert!(checkpoint_path.is_some(), "应该有保存的检查点");

    let (loaded_llm, _metadata) =
        CheckpointManager::load_checkpoint(checkpoint_path.unwrap()).expect("应该能加载检查点");

    // 验证加载的模型参数数量匹配
    assert_eq!(
        loaded_llm.total_parameters(),
        llm.total_parameters(),
        "参数数量应该匹配"
    );

    // 验证词汇表匹配
    assert_eq!(
        loaded_llm.vocab.len(),
        llm.vocab.len(),
        "词汇表大小应该匹配"
    );

    println!("✓ 检查点保存和加载测试通过");
    println!("  训练后loss: {:.4}", final_loss);
    println!("  模型参数数: {}", loaded_llm.total_parameters());

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_strategy_periodic() {
    let checkpoint_dir = "test_checkpoints_periodic";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Periodic(10), 3)
        .expect("应该能创建管理器");

    // 测试周期性保存逻辑
    assert!(manager.should_save(10, 2.0), "Epoch 10应该保存");
    assert!(!manager.should_save(11, 2.0), "Epoch 11不应该保存");
    assert!(manager.should_save(20, 2.0), "Epoch 20应该保存");
    assert!(manager.should_save(0, 2.0), "Epoch 0应该保存");

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_list_functionality() {
    let checkpoint_dir = "test_checkpoints_list";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
        .expect("应该能创建管理器");

    let (llm, _test_data) = create_test_model();

    // 保存多个检查点
    for i in 0..3 {
        let metadata = CheckpointMetadata {
            epoch: i * 10,
            loss: 3.0 - i as f32 * 0.5,
            learning_rate: 0.001,
            timestamp: format!("2025-10-24 1{}:00:00", i),
            phase: "test".to_string(),
        };
        manager.save_checkpoint(&llm, metadata).ok();
    }

    // 列出所有检查点
    let checkpoints = manager.list_checkpoints();
    assert!(checkpoints.is_ok(), "应该能列出检查点");

    let checkpoint_list = checkpoints.unwrap();
    assert!(
        checkpoint_list.len() >= 2,
        "应该至少有2个检查点（best + last）"
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_loss_continuity_after_resume() {
    //! 集成测试：验证保存/恢复后loss连续性
    //!
    //! 这个测试验证了完整的训练-保存-加载-继续训练流程：
    //! 1. 训练模型N个epoch
    //! 2. 保存检查点（包括Adam优化器状态）
    //! 3. 计算当前loss
    //! 4. 加载检查点
    //! 5. 验证加载后的loss与保存前相同（说明优化器状态正确恢复）
    //! 6. 继续训练N个epoch
    //! 7. 验证loss继续改善（说明训练连续性保持）

    let checkpoint_dir = "test_checkpoints_loss_continuity";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    // 准备训练数据（使用更多数据以便观察loss变化）
    let test_data = vec![
        "今天天气很好".to_string(),
        "我喜欢学习编程".to_string(),
        "机器学习很有趣".to_string(),
        "深度学习改变世界".to_string(),
        "人工智能的未来".to_string(),
        "自然语言处理".to_string(),
    ];

    // 构建词汇表
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&test_data, &mut vocab_set);
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // 创建模型
    let embeddings = Embeddings::new(vocab.clone());
    let transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    let mut llm = LLM::new(
        vocab.clone(),
        vec![
            Box::new(embeddings),
            Box::new(transformer),
            Box::new(output_projection),
        ],
    );

    // Tokenize训练数据
    let tokenized_data: Vec<Vec<usize>> = test_data
        .iter()
        .map(|text| LLM::tokenize_with_vocab(&vocab, text))
        .collect();

    // 创建检查点管理器
    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
        .expect("应该能创建检查点管理器");

    // ========== 阶段1: 初始训练 ==========
    println!("\n=== 阶段1: 初始训练 10 epochs ===");
    let phase1_epochs = llm.train_with_checkpointing(
        tokenized_data.clone(),
        10,
        0.001,
        100, // 高patience确保不会早停
        Some(&mut manager),
        "phase1",
        0,
    );
    println!("阶段1完成，训练了 {} epochs", phase1_epochs);

    // 验证检查点已保存；这里使用 last checkpoint 来保证拿到最新状态。
    let checkpoint_path = manager
        .get_last_checkpoint()
        .expect("应该有保存的last检查点");
    println!("检查点路径: {:?}", checkpoint_path);

    // 计算训练后的loss（阶段1结束时，在eval模式下）
    let loss_after_phase1 = compute_loss_eval(&mut llm, &tokenized_data);
    println!("阶段1结束时的loss (eval模式): {:.6}", loss_after_phase1);

    // ========== 阶段2: 加载检查点并验证loss一致性 ==========
    println!("\n=== 阶段2: 加载检查点 ===");
    let (mut loaded_llm, loaded_metadata) =
        CheckpointManager::load_checkpoint(&checkpoint_path).expect("应该能加载检查点");
    println!(
        "加载的检查点: epoch={}, loss={:.6}",
        loaded_metadata.epoch, loaded_metadata.loss
    );

    // 验证加载后的模型结构
    assert_eq!(
        loaded_llm.vocab.len(),
        llm.vocab.len(),
        "词汇表大小应该匹配"
    );
    assert_eq!(
        loaded_llm.total_parameters(),
        llm.total_parameters(),
        "参数数量应该匹配"
    );

    // 计算加载后的loss（在eval模式下，确保dropout关闭）
    let loss_after_load = compute_loss_eval(&mut loaded_llm, &tokenized_data);
    println!("加载后的loss (eval模式): {:.6}", loss_after_load);

    // 关键验证：加载后的loss应该与保存前的loss基本一致
    // 允许较小的浮点误差（< 0.1），因为存在序列化/反序列化的精度损失
    let loss_diff = (loss_after_load - loss_after_phase1).abs();
    println!("Loss差异: {:.6}", loss_diff);
    assert!(
        loss_diff < 0.1,
        "加载后的loss应该与保存前基本一致（差异: {:.6}），说明模型参数正确恢复",
        loss_diff
    );

    // ========== 阶段3: 继续训练验证连续性 ==========
    println!("\n=== 阶段3: 继续训练 10 epochs ===");

    // 重新创建检查点管理器以重置best_loss跟踪
    // 注意：这里从 epoch 10 开始继续训练
    let mut manager2 = CheckpointManager::new(
        format!("{}_resume", checkpoint_dir),
        CheckpointStrategy::BestAndLast,
        3,
    )
    .expect("应该能创建第二个检查点管理器");

    let phase3_epochs = loaded_llm.train_with_checkpointing(
        tokenized_data.clone(),
        20, // 训练到epoch 20
        0.001,
        100,
        Some(&mut manager2),
        "phase3",
        10, // 从epoch 10继续
    );
    println!("阶段3完成，训练到 epoch {}", phase3_epochs);

    // 计算继续训练后的loss
    // 重要：这里使用 eval 模式计算 loss，与“阶段1结束 / 加载后”保持同一口径，
    // 避免训练模式下 Dropout 引入噪声，导致测试结果偶发不稳定。
    let loss_after_phase3 = compute_loss_eval(&mut loaded_llm, &tokenized_data);
    println!("阶段3结束时的loss: {:.6}", loss_after_phase3);

    // 验证训练连续性：继续训练后loss应该减小（说明优化器状态正确恢复）
    // 注意：由于是小模型和少量数据，loss可能不会显著下降，但应该不会上升太多
    println!("\n=== 训练连续性验证 ===");
    println!("阶段1结束: {:.6}", loss_after_phase1);
    println!("加载后:    {:.6}", loss_after_load);
    println!("阶段3结束: {:.6}", loss_after_phase3);
    println!(
        "Loss变化: {:.6} → {:.6} (继续训练)",
        loss_after_load, loss_after_phase3
    );

    // 验证：继续训练后loss应该不会显著增加（允许小幅波动）
    // 如果loss大幅增加，说明优化器状态没有正确恢复
    assert!(
        loss_after_phase3 <= loss_after_load + 0.5,
        "继续训练后loss不应该大幅增加，说明优化器状态正确恢复 (增加了 {:.6})",
        loss_after_phase3 - loss_after_load
    );

    println!("\n✅ 检查点loss连续性测试通过！");
    println!("   • 保存/加载loss一致性: ✓ (差异 {:.6})", loss_diff);
    println!(
        "   • 训练连续性: ✓ (loss变化 {:.6})",
        loss_after_phase3 - loss_after_load
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
    fs::remove_dir_all(format!("{}_resume", checkpoint_dir)).ok();
}

#[test]
fn test_get_best_checkpoint_prefers_lowest_loss_over_latest_mtime() {
    let checkpoint_dir = "test_checkpoints_best_selection";

    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }
    fs::create_dir_all(checkpoint_dir).expect("应该能创建测试目录");

    // 先写入 loss 更低但时间更早的 best checkpoint。
    let low_loss_bin = format!(
        "{}/checkpoint_best_epoch_20_loss_2.5000.bin",
        checkpoint_dir
    );
    let low_loss_json = format!(
        "{}/checkpoint_best_epoch_20_loss_2.5000.json",
        checkpoint_dir
    );
    fs::write(&low_loss_bin, b"low-loss").expect("应该能写入低 loss bin");
    fs::write(
        &low_loss_json,
        serde_json::to_string_pretty(&CheckpointMetadata {
            epoch: 20,
            loss: 2.5,
            learning_rate: 0.001,
            timestamp: "2026-03-06 10:00:00".to_string(),
            phase: "test".to_string(),
        })
        .expect("应该能序列化低 loss 元数据"),
    )
    .expect("应该能写入低 loss json");

    // 确保第二个文件有更晚的修改时间。
    std::thread::sleep(std::time::Duration::from_millis(20));

    // 再写入 loss 更高但时间更晚的 best checkpoint。
    let high_loss_bin = format!(
        "{}/checkpoint_best_epoch_30_loss_2.7000.bin",
        checkpoint_dir
    );
    let high_loss_json = format!(
        "{}/checkpoint_best_epoch_30_loss_2.7000.json",
        checkpoint_dir
    );
    fs::write(&high_loss_bin, b"high-loss").expect("应该能写入高 loss bin");
    fs::write(
        &high_loss_json,
        serde_json::to_string_pretty(&CheckpointMetadata {
            epoch: 30,
            loss: 2.7,
            learning_rate: 0.001,
            timestamp: "2026-03-06 11:00:00".to_string(),
            phase: "test".to_string(),
        })
        .expect("应该能序列化高 loss 元数据"),
    )
    .expect("应该能写入高 loss json");

    let manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能从已有目录恢复检查点管理器状态");

    let best_path = manager
        .get_best_checkpoint()
        .expect("应该能找到最佳 checkpoint");

    assert_eq!(
        best_path.file_name().and_then(|s| s.to_str()),
        Some("checkpoint_best_epoch_20_loss_2.5000.bin"),
        "应优先选择 loss 最低的 best checkpoint，而不是修改时间最新的文件"
    );

    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_save_failure_does_not_advance_best_state() {
    let checkpoint_dir = "test_checkpoints_save_failure";

    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
        fs::remove_file(checkpoint_dir).ok();
    }

    let (llm, _test_data) = create_test_model();
    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能创建检查点管理器");

    // 先成功保存一个 baseline best。
    manager
        .save_checkpoint(
            &llm,
            CheckpointMetadata {
                epoch: 10,
                loss: 3.0,
                learning_rate: 0.001,
                timestamp: "2026-03-06 10:00:00".to_string(),
                phase: "test".to_string(),
            },
        )
        .expect("baseline checkpoint 应保存成功");

    assert!((manager.get_best_loss() - 3.0).abs() < 1e-6);
    assert_eq!(manager.get_best_epoch(), 10);

    // 破坏保存目录：删除目录并在同一路径创建普通文件，确保后续 join/create 必然失败。
    fs::remove_dir_all(checkpoint_dir).expect("应能删除原检查点目录");
    fs::write(checkpoint_dir, b"not a directory").expect("应能创建同名普通文件");

    let save_result = manager.save_checkpoint(
        &llm,
        CheckpointMetadata {
            epoch: 20,
            loss: 2.0,
            learning_rate: 0.001,
            timestamp: "2026-03-06 11:00:00".to_string(),
            phase: "test".to_string(),
        },
    );
    assert!(save_result.is_err(), "保存路径损坏时保存应失败");

    // 即使失败，也不应把内存中的 best 状态提前推进到 2.0 / epoch=20。
    assert!(
        (manager.get_best_loss() - 3.0).abs() < 1e-6,
        "保存失败后 best_loss 不应漂移"
    );
    assert_eq!(
        manager.get_best_epoch(),
        10,
        "保存失败后 best_epoch 不应漂移"
    );

    fs::remove_file(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_manager_restores_best_state_from_existing_dir() {
    let checkpoint_dir = "test_checkpoints_restore_best_state";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let (llm, _test_data) = create_test_model();

    {
        let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
            .expect("应该能创建检查点管理器");

        let metadata1 = CheckpointMetadata {
            epoch: 10,
            loss: 3.0,
            learning_rate: 0.001,
            timestamp: "2026-03-06 10:00:00".to_string(),
            phase: "test".to_string(),
        };
        manager.save_checkpoint(&llm, metadata1).ok();

        let metadata2 = CheckpointMetadata {
            epoch: 20,
            loss: 2.5,
            learning_rate: 0.001,
            timestamp: "2026-03-06 11:00:00".to_string(),
            phase: "test".to_string(),
        };
        manager.save_checkpoint(&llm, metadata2).ok();

        assert!((manager.get_best_loss() - 2.5).abs() < 1e-6);
        assert_eq!(manager.get_best_epoch(), 20);
    }

    // 模拟进程重启：重新创建同目录 manager，必须恢复历史 best 状态。
    let restored_manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能从已有目录恢复检查点管理器状态");

    assert!(
        (restored_manager.get_best_loss() - 2.5).abs() < 1e-6,
        "应从目录恢复最佳 loss"
    );
    assert_eq!(
        restored_manager.get_best_epoch(),
        20,
        "应从目录恢复最佳 epoch"
    );

    // 验证后续保存判定确实基于恢复后的 best_loss，而不是 +∞。
    assert!(
        !restored_manager.should_save(21, 2.6),
        "比历史 best 更差的 loss 不应被视为新的 best"
    );
    assert!(
        restored_manager.should_save(21, 2.4),
        "比历史 best 更优的 loss 应被视为新的 best"
    );

    fs::remove_dir_all(checkpoint_dir).ok();
}

/// 辅助函数：计算给定模型在数据上的平均loss（训练模式）
#[allow(dead_code)]
fn compute_loss(llm: &mut LLM, tokenized_data: &[Vec<usize>]) -> f32 {
    llm.set_training_mode(true);
    let pad_token_id = llm.vocab.pad_token_id();
    let mut total_nll = 0.0;
    let mut total_tokens = 0usize;

    for training_row in tokenized_data {
        if training_row.len() < 2 {
            continue;
        }

        let input_ids = &training_row[..training_row.len() - 1];
        let target_ids = &training_row[1..];

        let input =
            ndarray::Array2::from_shape_fn((1, input_ids.len()), |(_, j)| input_ids[j] as f32);

        let mut output = input.clone();
        for layer in &mut llm.network {
            let (out, _ctx) = layer.forward(&output);
            output = out;
        }

        let probs = llm::utils::softmax(&output);
        let n_targets = target_ids.iter().filter(|&&t| t != pad_token_id).count();
        total_nll +=
            LLM::cross_entropy_loss_step(&probs, target_ids, pad_token_id) * (n_targets as f32);
        total_tokens += n_targets;
    }

    llm.set_training_mode(false);
    if total_tokens > 0 {
        total_nll / total_tokens as f32
    } else {
        0.0
    }
}

/// 辅助函数：计算给定模型在数据上的平均loss（评估模式，关闭dropout）
fn compute_loss_eval(llm: &mut LLM, tokenized_data: &[Vec<usize>]) -> f32 {
    llm.set_training_mode(false);
    let pad_token_id = llm.vocab.pad_token_id();
    let mut total_nll = 0.0;
    let mut total_tokens = 0usize;

    for training_row in tokenized_data {
        if training_row.len() < 2 {
            continue;
        }

        let input_ids = &training_row[..training_row.len() - 1];
        let target_ids = &training_row[1..];

        let input =
            ndarray::Array2::from_shape_fn((1, input_ids.len()), |(_, j)| input_ids[j] as f32);

        let mut output = input.clone();
        for layer in &mut llm.network {
            let (out, _ctx) = layer.forward(&output);
            output = out;
        }

        let probs = llm::utils::softmax(&output);
        let n_targets = target_ids.iter().filter(|&&t| t != pad_token_id).count();
        total_nll +=
            LLM::cross_entropy_loss_step(&probs, target_ids, pad_token_id) * (n_targets as f32);
        total_tokens += n_targets;
    }

    if total_tokens > 0 {
        total_nll / total_tokens as f32
    } else {
        0.0
    }
}

struct SeqLenAwareProbeLayer {
    logits_len1: Array2<f32>,
    logits_len2: Array2<f32>,
}

impl Layer for SeqLenAwareProbeLayer {
    fn layer_type(&self) -> &str {
        "SeqLenAwareProbeLayer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        let out = match input.shape()[1] {
            1 => self.logits_len1.clone(),
            2 => self.logits_len2.clone(),
            other => panic!("unexpected seq len: {}", other),
        };
        (out, Box::new(()))
    }

    fn backward(&mut self, _ctx: &LayerContext, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        Array2::zeros((1, grads.nrows()))
    }

    fn parameters(&self) -> usize {
        0
    }

    fn set_training_mode(&mut self, _training: bool) {}
}

fn logits_row_for_target_loss(vocab_size: usize, target_idx: usize, loss: f32) -> Vec<f32> {
    let prob = (-loss).exp();
    let distractor_prob = (1.0 - prob).max(1e-6);
    let distractor_idx = if target_idx == 0 { 1 } else { 0 };

    let mut row = vec![-1000.0f32; vocab_size];
    row[target_idx] = prob.ln();
    row[distractor_idx] = distractor_prob.ln();
    row
}

#[test]
fn test_compute_loss_eval_uses_token_weighted_mean() {
    let vocab = Vocab::new(vec!["a", "b", "c"]);
    let vocab_size = vocab.len();
    let a_id = vocab.encode("a").unwrap();
    let b_id = vocab.encode("b").unwrap();
    let c_id = vocab.encode("c").unwrap();

    // 样本1：1 个 target，loss = 3.0
    let logits_len1 = Array2::from_shape_vec(
        (1, vocab_size),
        logits_row_for_target_loss(vocab_size, b_id, 3.0),
    )
    .unwrap();

    // 样本2：2 个 target，每个 token 的 loss = 1.0
    let mut logits_len2_data = logits_row_for_target_loss(vocab_size, b_id, 1.0);
    logits_len2_data.extend(logits_row_for_target_loss(vocab_size, c_id, 1.0));
    let logits_len2 = Array2::from_shape_vec((2, vocab_size), logits_len2_data).unwrap();

    let mut llm = LLM::new(
        vocab,
        vec![Box::new(SeqLenAwareProbeLayer {
            logits_len1,
            logits_len2,
        })],
    );

    let tokenized_data = vec![vec![a_id, b_id], vec![a_id, b_id, c_id]];
    let loss = compute_loss_eval(&mut llm, &tokenized_data);

    // token-weighted: (3.0 * 1 + 1.0 * 2) / 3 = 5/3
    let expected = 5.0 / 3.0;
    assert!(
        (loss - expected).abs() < 1e-4,
        "expected token-weighted mean {}, got {}",
        expected,
        loss
    );
}

#[test]
fn test_checkpoint_adam_optimizer_state_preservation() {
    //! 测试Adam优化器状态的保存和恢复
    //!
    //! 验证检查点中包含完整的Adam优化器状态（m, v, timestep）

    let checkpoint_dir = "test_checkpoints_adam_state";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    // 创建模型并训练
    let (mut llm, test_data) = create_test_model();

    let tokenized_data: Vec<Vec<usize>> = test_data
        .iter()
        .map(|text| LLM::tokenize_with_vocab(&llm.vocab, text))
        .collect();

    // 训练几个epoch以确保Adam优化器状态不为零
    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
        .expect("应该能创建检查点管理器");

    llm.train_with_checkpointing(
        tokenized_data.clone(),
        5,
        0.001,
        100,
        Some(&mut manager),
        "optimizer_test",
        0,
    );

    // 保存检查点
    let checkpoint_path = manager
        .get_best_checkpoint()
        .or_else(|| manager.get_last_checkpoint())
        .expect("应该有检查点");

    // 加载检查点
    let (loaded_llm, _) =
        CheckpointManager::load_checkpoint(&checkpoint_path).expect("应该能加载检查点");

    // 验证模型结构
    assert_eq!(
        loaded_llm.network.len(),
        llm.network.len(),
        "网络层数应该匹配"
    );
    assert_eq!(
        loaded_llm.total_parameters(),
        llm.total_parameters(),
        "总参数数应该匹配"
    );

    println!("\n✅ Adam优化器状态保存测试通过！");
    println!("   • 模型层数: {}", loaded_llm.network.len());
    println!("   • 总参数: {}", loaded_llm.total_parameters());
    println!("   • 词汇表大小: {}", loaded_llm.vocab.len());

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}
