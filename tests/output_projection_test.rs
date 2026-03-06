use llm::{EMBEDDING_DIM, Layer, output_projection::OutputProjection};
use ndarray::Array2;

#[test]
fn test_output_projection_creation() {
    let vocab_size = 10;
    let output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    // 检查权重矩阵维度。
    assert_eq!(output_proj.w_out.shape(), [EMBEDDING_DIM, vocab_size]);

    // 检查偏置向量维度。
    assert_eq!(output_proj.b_out.shape(), [1, vocab_size]);

    // 检查优化器内部状态维度。
    assert_eq!(output_proj.optimizer.m.shape(), [EMBEDDING_DIM, vocab_size]);
    assert_eq!(output_proj.optimizer.v.shape(), [EMBEDDING_DIM, vocab_size]);
}

#[test]
fn test_output_projection_forward() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    // 构造输入张量：(seq_len=3, embedding_dim=EMBEDDING_DIM)。
    let input = Array2::ones((3, EMBEDDING_DIM));

    // 执行前向传播。
    let (output, _ctx) = output_proj.forward(&input);

    // 输出应为 `(seq_len, vocab_size)`。
    assert_eq!(output.shape(), [3, vocab_size]);
}

#[test]
fn test_output_projection_with_different_sequence_lengths() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    // 验证不同序列长度下都能正常前向。
    for seq_len in 1..5 {
        // 构造输入张量。
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // 执行前向传播。
        let (output, _ctx) = output_proj.forward(&input);

        // 输出形状应与当前序列长度匹配。
        assert_eq!(output.shape(), [seq_len, vocab_size]);
    }
}

#[test]
fn test_output_projection_backward() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    // 构造输入张量。
    let input = Array2::ones((3, EMBEDDING_DIM));

    // 先执行一次前向传播，以取得 backward 所需的 ctx。
    let (_output, ctx) = output_proj.forward(&input);

    // 构造上游梯度张量。
    let grads = Array2::ones((3, vocab_size));

    // 执行反向传播。
    let grad_input = output_proj.backward(&ctx, &grads, 0.01);

    // 检查输入梯度形状。
    assert_eq!(grad_input.shape(), [3, EMBEDDING_DIM]);

    // 记录参数，后续验证是否发生更新。
    let w_out_before = output_proj.w_out.clone();
    let b_out_before = output_proj.b_out.clone();

    // 再执行一次前向与反向传播。
    let (_output, ctx2) = output_proj.forward(&input);
    let _grad_input = output_proj.backward(&ctx2, &grads, 0.01);

    // 参数应发生变化。
    assert_ne!(output_proj.w_out, w_out_before);
    assert_ne!(output_proj.b_out, b_out_before);
}

#[test]
fn test_output_projection_training() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);

    // 构造输入张量。
    let input = Array2::ones((3, EMBEDDING_DIM));

    // 连续执行多步训练。
    for _ in 0..5 {
        // 前向传播。
        let (_output, ctx) = output_proj.forward(&input);

        // 构造模拟交叉熵梯度的上游信号。
        let mut grads = Array2::zeros((3, vocab_size));
        grads[[0, 0]] = 1.0; // 仅给第一个位置一个非零梯度。

        // 反向传播。
        let _grad_input = output_proj.backward(&ctx, &grads, 0.01);
    }

    // 记录参数，后续验证是否发生更新。
    assert_ne!(output_proj.w_out.sum(), 0.0);
    assert_ne!(output_proj.b_out.sum(), 0.0);
}
