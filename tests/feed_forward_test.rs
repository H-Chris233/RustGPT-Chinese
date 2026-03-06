use llm::{EMBEDDING_DIM, HIDDEN_DIM, Layer, feed_forward::FeedForward};
use ndarray::Array2;

#[test]
fn test_feed_forward_forward() {
    // 创建前馈网络模块。
    let mut feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // 构造输入张量：(seq_len=3, embedding_dim=EMBEDDING_DIM)。
    let input = Array2::ones((3, EMBEDDING_DIM));

    // 执行前向传播。
    let (output, _ctx) = feed_forward.forward(&input);

    // 输出形状应与输入一致。
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_feed_forward_with_different_sequence_lengths() {
    // 创建前馈网络模块。
    let mut feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // 验证不同序列长度下都能正常前向。
    for seq_len in 1..5 {
        // 构造输入张量。
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // 执行前向传播。
        let (output, _ctx) = feed_forward.forward(&input);

        // 输出形状应与当前序列长度匹配。
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}

#[test]
fn test_feed_forward_and_backward() {
    // 创建前馈网络模块。
    let mut feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // 构造输入张量：(seq_len=3, embedding_dim=EMBEDDING_DIM)。
    let input = Array2::ones((3, EMBEDDING_DIM));

    // 执行前向传播。
    let (output, ctx) = feed_forward.forward(&input);

    let grads = Array2::ones((3, EMBEDDING_DIM));

    // 执行反向传播。
    let grad_input = feed_forward.backward(&ctx, &grads, 0.01);

    // 确认反向传播返回的梯度与前向输出不同。
    assert_ne!(output, grad_input);
}
