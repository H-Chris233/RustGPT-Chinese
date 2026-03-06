use llm::{EMBEDDING_DIM, MAX_SEQ_LEN, position_encoding::PositionEncoding};

#[test]
fn test_position_encoding_creation() {
    let pos_enc = PositionEncoding::new();
    assert_eq!(pos_enc.encoding.dim(), (MAX_SEQ_LEN, EMBEDDING_DIM));
}

#[test]
fn test_position_encoding_values() {
    let pos_enc = PositionEncoding::new();
    // 验证若干已知位置编码值。
    let val1 = pos_enc.get_encoding(0, 0); // 理论上应为 sin(0)=0。
    let val2 = pos_enc.get_encoding(1, 0); // 第 0 维对应 sin(1)。

    assert!((val1 - 0.0).abs() < 1e-5); // 近似等于 0。
    assert!((val2 - 0.841471).abs() < 1e-5); // 近似等于 sin(1)。
}

#[test]
fn test_apply_to_input() {
    let pos_enc = PositionEncoding::new();
    let mut input = ndarray::Array2::ones((5, EMBEDDING_DIM)); // 5 个 token 的嵌入矩阵。
    let original_sum = input.sum();

    pos_enc.apply_to_input(&mut input);

    // 应用位置编码后，矩阵总和应发生变化。
    let new_sum = input.sum();
    assert_ne!(original_sum, new_sum);
}
