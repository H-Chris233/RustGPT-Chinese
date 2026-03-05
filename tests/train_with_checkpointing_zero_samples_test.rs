use llm::{Embeddings, LLM, OutputProjection, Vocab, EMBEDDING_DIM};

/// `train_with_checkpointing` 在没有任何有效样本时不应除零/崩溃。
#[test]
fn train_with_checkpointing_handles_zero_samples() {
    // 最小网络：Embeddings -> OutputProjection
    let vocab = Vocab::new(vec!["甲", "乙", "丙"]);
    let vocab_size = vocab.words.len();

    let embeddings = Embeddings::new(vocab.clone());
    let output = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    let mut llm = LLM::new(vocab, vec![Box::new(embeddings), Box::new(output)]);

    // 空数据：sample_count 将保持为 0
    let tokenized_data = Vec::<Vec<usize>>::new();

    let trained_until = llm.train_with_checkpointing(
        tokenized_data,
        3,      // max_epochs
        0.001,  // lr
        1,      // patience
        None,   // checkpoint_manager
        "test", // phase
        0,      // resume_epoch
    );

    // 关键是“不要 panic/NaN”，并且能立即返回。
    assert_eq!(trained_until, 0);
}

