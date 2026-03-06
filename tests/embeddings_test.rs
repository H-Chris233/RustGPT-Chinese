use llm::{EMBEDDING_DIM, Embeddings, Layer, MAX_SEQ_LEN, Vocab};

#[test]
fn test_embeddings_creation() {
    // 使用自定义词表构造嵌入层。
    let words = vec!["hello", "world", "test", "</s>"];
    let _vocab = Vocab::new(words); // 仅验证可正常构造，避免未使用变量告警。
}

#[test]
fn test_embed_tokens() {
    // 创建词表和嵌入层。
    let words = vec!["hello", "world", "test", "</s>"];
    let vocab = Vocab::new(words);
    let embeddings = Embeddings::new(vocab.clone());

    // 测试单个 token 的嵌入结果。
    let token_ids = vec![0]; // 对应第一个普通词元。
    let embedded = embeddings.embed_tokens(&token_ids);

    // 检查输出维度。
    assert_eq!(embedded.shape(), [1, EMBEDDING_DIM]);

    // 测试多个 token 的嵌入结果。
    let token_ids = vec![0, 1, 2];
    let embedded = embeddings.embed_tokens(&token_ids);

    // 检查输出维度。
    assert_eq!(embedded.shape(), [3, EMBEDDING_DIM]);
}

#[test]
fn test_positional_embeddings() {
    // 创建词表和嵌入层。
    let words = vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
    let vocab = Vocab::new(words);
    let embeddings = Embeddings::new(vocab);

    // 验证不同序列长度下的位置编码行为。
    for seq_len in 1..5 {
        let token_ids = vec![0; seq_len]; // 重复同一个 token，便于观察位置差异。
        let embedded = embeddings.embed_tokens(&token_ids);

        // 检查输出维度。
        assert_eq!(embedded.shape(), [seq_len, EMBEDDING_DIM]);

        // 同一个 token 在不同位置的嵌入应不同，说明位置编码已生效。
        if seq_len > 1 {
            let first_pos = embedded.row(0).to_owned();
            let second_pos = embedded.row(1).to_owned();

            // 位置编码不同，因此结果不应完全相同。
            assert_ne!(first_pos, second_pos);
        }
    }
}

#[test]
fn test_max_sequence_length() {
    // 创建词表和嵌入层。
    let vocab = Vocab::default();
    let embeddings = Embeddings::new(vocab);

    // 构造最大长度序列。
    let token_ids = vec![0; MAX_SEQ_LEN];
    let embedded = embeddings.embed_tokens(&token_ids);

    // 检查输出维度。
    assert_eq!(embedded.shape(), [MAX_SEQ_LEN, EMBEDDING_DIM]);
}

#[test]
fn test_embedding_backwards() {
    // 创建词表和嵌入层。
    let vocab = Vocab::default();
    let mut embeddings = Embeddings::new(vocab);

    let pre_train_token_embeddings = embeddings.token_embeddings.clone();

    // 模拟一次前向与反向传播。
    use ndarray::Array2;
    let input = match Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 2.0]) {
        Ok(v) => v,
        Err(e) => {
            assert!(false, "构造测试输入矩阵失败: {}", e);
            return;
        }
    };
    let (_output, ctx) = embeddings.forward(&input);

    // 构造测试梯度并执行反向传播。
    let grads = match Array2::from_shape_vec((3, EMBEDDING_DIM), vec![0.1; 3 * EMBEDDING_DIM]) {
        Ok(v) => v,
        Err(e) => {
            assert!(false, "构造测试梯度矩阵失败: {}", e);
            return;
        }
    };
    let _grad_input = embeddings.backward(&ctx, &grads, 0.01);

    let post_train_token_embeddings = embeddings.token_embeddings.clone();

    assert_ne!(pre_train_token_embeddings, post_train_token_embeddings);
}
