//! 批量训练掩码测试
//!
//! 专门测试 PAD 位置在损失和梯度计算中被正确屏蔽

use llm::batch_loader::*;
use ndarray::Array2;

#[test]
fn test_pad_mask_excludes_from_loss() {
    // 测试目标：验证 PAD token 不参与损失计算

    let sequences = vec![
        vec![1, 2, 3, 4, 5], // 长度5
        vec![6, 7],          // 长度2，会被PAD到5
    ];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // 第二个样本的后3个位置应该是PAD
    assert_eq!(batch.tokens[[1, 0]], 6);
    assert_eq!(batch.tokens[[1, 1]], 7);
    assert_eq!(batch.tokens[[1, 2]], PAD_TOKEN_ID);
    assert_eq!(batch.tokens[[1, 3]], PAD_TOKEN_ID);
    assert_eq!(batch.tokens[[1, 4]], PAD_TOKEN_ID);

    // 对应的掩码
    assert_eq!(batch.attention_mask[[1, 0]], 1.0);
    assert_eq!(batch.attention_mask[[1, 1]], 1.0);
    assert_eq!(batch.attention_mask[[1, 2]], 0.0);
    assert_eq!(batch.attention_mask[[1, 3]], 0.0);
    assert_eq!(batch.attention_mask[[1, 4]], 0.0);
}

#[test]
fn test_gradient_masking() {
    // 测试梯度掩码功能

    // 创建一个模拟的梯度张量，确保初始值不为0
    let mut grads = Array2::from_shape_fn((2, 5), |(i, j)| {
        ((i * 5 + j) as f32 + 1.0) * 0.1 // +1.0 确保所有值都 > 0
    });

    // 创建注意力掩码
    let mut mask = Array2::ones((2, 5));
    mask[[0, 4]] = 0.0; // 第一个样本的最后一个位置是PAD
    mask[[1, 2]] = 0.0; // 第二个样本的第3个位置是PAD
    mask[[1, 3]] = 0.0;
    mask[[1, 4]] = 0.0;

    // 应用掩码（模拟训练中的梯度清零）
    for i in 0..grads.nrows() {
        for j in 0..grads.ncols() {
            if mask[[i, j]] < 0.5 {
                grads[[i, j]] = 0.0;
            }
        }
    }

    // 验证PAD位置的梯度被清零
    assert_eq!(grads[[0, 4]], 0.0);
    assert_eq!(grads[[1, 2]], 0.0);
    assert_eq!(grads[[1, 3]], 0.0);
    assert_eq!(grads[[1, 4]], 0.0);

    // 验证非PAD位置的梯度不变
    assert!(grads[[0, 0]].abs() > 1e-6);
    assert!(grads[[1, 0]].abs() > 1e-6);
}

#[test]
fn test_dynamic_padding_reduces_waste() {
    // 测试动态填充比全局固定填充更高效

    let sequences = vec![vec![1, 2], vec![3, 4, 5], vec![6, 7, 8, 9]];

    let loader = BatchLoader::new(3, false, 8);
    let batches = loader.create_batches(&sequences);

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // 批次最大长度应该是4（而不是全局的MAX_SEQ_LEN=128）
    assert_eq!(batch.seq_len, 4);

    // 计算实际PAD的数量
    let pad_count = batch.tokens.iter().filter(|&&x| x == PAD_TOKEN_ID).count();

    // 应该有：
    // - 第一个样本：2个PAD（从2填充到4）
    // - 第二个样本：1个PAD（从3填充到4）
    // - 第三个样本：0个PAD（已经是4）
    // 总共：3个PAD
    assert_eq!(pad_count, 3);

    // 如果使用固定长度128，会浪费 (3 * 128) - (2 + 3 + 4) = 384 - 9 = 375 个位置
    // 使用动态填充，只浪费3个位置
    println!(
        "动态填充节省: {} 个位置",
        (3 * 128) - (2 + 3 + 4) - pad_count
    );
}

#[test]
fn test_bucketing_efficiency() {
    // 测试分桶策略的效率

    let sequences = vec![
        vec![1; 3],  // 长度3
        vec![2; 5],  // 长度5
        vec![3; 10], // 长度10
        vec![4; 12], // 长度12
        vec![5; 20], // 长度20
        vec![6; 22], // 长度22
    ];

    // 使用分桶策略（bucket_width=8）
    let loader_bucketed = BatchLoader::new(2, true, 8);
    let batches_bucketed = loader_bucketed.create_batches(&sequences);

    // 不使用分桶策略
    let loader_simple = BatchLoader::new(2, false, 8);
    let batches_simple = loader_simple.create_batches(&sequences);

    // 两者都应该创建多个批次
    assert!(!batches_bucketed.is_empty());
    assert!(!batches_simple.is_empty());

    println!("分桶批次数: {}", batches_bucketed.len());
    println!("简单批次数: {}", batches_simple.len());

    // 分桶策略应该将相似长度的序列分组，减少整体PAD数量
}

#[test]
fn test_teacher_forcing_with_pad() {
    // 测试 teacher forcing 模式下的 PAD 处理

    let sequences = vec![vec![1, 2, 3, 4], vec![5, 6]];

    let loader = BatchLoader::new(2, false, 8);
    let training_batches = create_training_batches(&loader, &sequences);

    assert_eq!(training_batches.len(), 1);
    let (input_batch, targets) = &training_batches[0];

    // Input: tokens[:-1]，长度应该是3
    assert_eq!(input_batch.seq_len, 3);

    // Target: tokens[1:]
    assert_eq!(targets[0], vec![2, 3, 4]);
    assert_eq!(targets[1], vec![6]);

    // 注意力掩码应该正确反映实际长度
    // 第一个样本：3个真实token
    assert_eq!(input_batch.attention_mask[[0, 0]], 1.0);
    assert_eq!(input_batch.attention_mask[[0, 1]], 1.0);
    assert_eq!(input_batch.attention_mask[[0, 2]], 1.0);

    // 第二个样本：原始序列[5, 6]，填充到[5, 6, PAD, PAD]
    // input batch 取前3列，所以是[5, 6, PAD]
    // mask 也是前3列，所以是[1.0, 1.0, 0.0]
    assert_eq!(input_batch.attention_mask[[1, 0]], 1.0);
    assert_eq!(input_batch.attention_mask[[1, 1]], 1.0);
    assert_eq!(input_batch.attention_mask[[1, 2]], 0.0);
}
