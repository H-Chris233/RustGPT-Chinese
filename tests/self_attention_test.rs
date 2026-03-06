use llm::{EMBEDDING_DIM, Layer, self_attention::SelfAttention};
use ndarray::Array2;

#[test]
fn test_self_attention_forward() {
    // 创建自注意力模块。
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 构造输入张量：(seq_len=3, embedding_dim=EMBEDDING_DIM)。
    let input = Array2::ones((3, EMBEDDING_DIM));

    // 执行前向传播。
    let (output, _ctx) = self_attention.forward(&input);

    // 输出形状应与输入一致。
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_self_attention_with_different_sequence_lengths() {
    // 创建自注意力模块。
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 验证不同序列长度下都能正常前向。
    for seq_len in 1..5 {
        // 构造输入张量。
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // 执行前向传播。
        let (output, _ctx) = self_attention.forward(&input);

        // 输出形状应与当前序列长度匹配。
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}
