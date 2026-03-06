use llm::{Embeddings, LLM, OutputProjection, Vocab, EMBEDDING_DIM};

fn create_minimal_llm() -> LLM {
    // 最小网络：Embeddings -> OutputProjection
    let vocab = Vocab::new(vec!["甲", "乙", "丙"]);
    let vocab_size = vocab.words.len();

    let embeddings = Embeddings::new(vocab.clone());
    let output = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    LLM::new(vocab, vec![Box::new(embeddings), Box::new(output)])
}

/// `train_with_checkpointing` 在没有任何有效样本时不应除零/崩溃。
#[test]
fn train_with_checkpointing_handles_zero_samples() {
    let mut llm = create_minimal_llm();

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
    assert!(!llm.training, "零样本提前返回后应恢复为非训练模式");
}

/// `train_with_checkpointing` 在 resume 场景下遇到零样本时，
/// 应返回传入的 `resume_epoch`，而不是重置为 0。
#[test]
fn train_with_checkpointing_zero_samples_preserves_resume_epoch() {
    let mut llm = create_minimal_llm();
    let resume_epoch = 3usize;

    let trained_until = llm.train_with_checkpointing(
        Vec::new(),
        8,        // max_epochs
        0.001,    // lr
        2,        // patience
        None,     // checkpoint_manager
        "resume", // phase
        resume_epoch,
    );

    assert_eq!(trained_until, resume_epoch);
    assert!(!llm.training, "resume 的零样本提前返回后应恢复为非训练模式");
}
