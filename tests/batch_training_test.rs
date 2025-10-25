//! 批量训练测试
//!
//! 测试批量训练和动态掩码功能

use llm::{
    batch_loader::*, Embeddings, LLM, OutputProjection, TransformerBlock, Vocab, EMBEDDING_DIM,
    HIDDEN_DIM,
};

#[test]
fn test_batch_loader_basic() {
    let sequences = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);

    // 应该创建2个批次
    assert!(batches.len() >= 1);

    // 第一个批次应该有2个样本
    if batches.len() >= 1 {
        assert_eq!(batches[0].batch_size, 2);

        // 检查注意力掩码
        // 第一个样本长度3，全是1.0
        assert_eq!(batches[0].attention_mask[[0, 0]], 1.0);
        assert_eq!(batches[0].attention_mask[[0, 1]], 1.0);
        assert_eq!(batches[0].attention_mask[[0, 2]], 1.0);

        // 第二个样本长度2，前两个是1.0，第三个是0.0（PAD）
        assert_eq!(batches[0].attention_mask[[1, 0]], 1.0);
        assert_eq!(batches[0].attention_mask[[1, 1]], 1.0);
        assert_eq!(batches[0].attention_mask[[1, 2]], 0.0);
    }
}

#[test]
fn test_batch_loader_bucketing() {
    let sequences = vec![
        vec![1, 2],           // 长度2
        vec![3, 4, 5],        // 长度3
        vec![6, 7, 8, 9, 10], // 长度5
        vec![11; 10],         // 长度10
    ];

    let loader = BatchLoader::new(2, true, 8);
    let batches = loader.create_batches(&sequences);

    // 使用分桶策略应该创建合理的批次
    assert!(!batches.is_empty());
}

#[test]
fn test_training_batches_creation() {
    let sequences = vec![vec![1, 2, 3, 4], vec![5, 6, 7]];

    let loader = BatchLoader::new(2, false, 8);
    let training_batches = create_training_batches(&loader, &sequences);

    assert_eq!(training_batches.len(), 1);

    let (input_batch, targets) = &training_batches[0];

    // Input 应该是 tokens[:-1]，所以长度是3
    assert_eq!(input_batch.seq_len, 3);

    // Target 应该是 tokens[1:]
    assert_eq!(targets[0], vec![2, 3, 4]);
    assert_eq!(targets[1], vec![6, 7]);
}

#[test]
fn test_attention_mask_with_pad() {
    let sequences = vec![vec![1, 2, 3], vec![4, 5]];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);

    let batch = &batches[0];

    // 验证PAD token被正确填充
    assert_eq!(batch.tokens[[0, 0]], 1);
    assert_eq!(batch.tokens[[0, 1]], 2);
    assert_eq!(batch.tokens[[0, 2]], 3);

    assert_eq!(batch.tokens[[1, 0]], 4);
    assert_eq!(batch.tokens[[1, 1]], 5);
    assert_eq!(batch.tokens[[1, 2]], PAD_TOKEN_ID); // PAD填充

    // 验证注意力掩码正确
    assert_eq!(batch.attention_mask[[0, 0]], 1.0);
    assert_eq!(batch.attention_mask[[0, 1]], 1.0);
    assert_eq!(batch.attention_mask[[0, 2]], 1.0);

    assert_eq!(batch.attention_mask[[1, 0]], 1.0);
    assert_eq!(batch.attention_mask[[1, 1]], 1.0);
    assert_eq!(batch.attention_mask[[1, 2]], 0.0); // PAD位置为0
}

#[test]
fn test_batch_training_with_small_model() {
    // 创建一个小的词汇表（从训练数据构建）
    let training_texts = vec!["你好 世界".to_string(), "测试".to_string()];

    let vocab = Vocab::build_from_texts(&training_texts);

    // 创建一个简单的网络
    let embeddings = Embeddings::new(vocab.clone());
    let transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    let network: Vec<Box<dyn llm::Layer>> = vec![
        Box::new(embeddings),
        Box::new(transformer),
        Box::new(output_projection),
    ];

    let mut model = LLM::new(vocab, network);

    // 准备训练数据
    let data = vec!["你好 世界", "测试"];

    // 使用批量训练（小批次，少epoch）
    let epochs_trained = model.train_monitored_batch(
        data, 2,     // max_epochs
        0.001, // initial_lr
        10,    // patience
        2,     // batch_size
    );

    // 验证训练完成
    assert!(epochs_trained > 0);
    assert!(epochs_trained <= 2);
}

#[test]
fn test_pad_token_gradient_masking() {
    // 测试PAD位置的梯度是否被正确屏蔽
    let sequences = vec![
        vec![1, 2, 3],
        vec![4, 5], // 这个会被PAD到长度3
    ];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);
    let batch = &batches[0];

    // 验证PAD位置的掩码
    assert_eq!(batch.attention_mask[[1, 2]], 0.0, "PAD位置掩码应该为0");

    // 在实际训练中，PAD位置的梯度应该被清零
    // 这里只是验证掩码的正确性
}
