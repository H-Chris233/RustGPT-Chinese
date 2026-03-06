use llm::{Layer, transformer::TransformerBlock};
use ndarray::Array2;

fn make_weight(rows: usize, cols: usize, scale: f32, offset: f32) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let idx = (r * cols + c) as f32;
        let v = idx * scale + offset;
        if (r + c) % 2 == 0 { v } else { -v }
    })
}

fn make_transformer_block(dim: usize, hidden_dim: usize) -> TransformerBlock {
    let mut block = TransformerBlock::new(dim, hidden_dim);

    block.attention.w_q = make_weight(dim, dim, 0.001, 0.01);
    block.attention.w_k = make_weight(dim, dim, 0.001, -0.02);
    block.attention.w_v = make_weight(dim, dim, 0.001, 0.03);
    block.attention.w_o = make_weight(dim, dim, 0.001, -0.04);

    block.feed_forward.w1 = make_weight(dim, hidden_dim, 0.001, 0.02);
    block.feed_forward.b1 = make_weight(1, hidden_dim, 0.0005, -0.01);
    block.feed_forward.w2 = make_weight(hidden_dim, dim, 0.001, -0.03);
    block.feed_forward.b2 = make_weight(1, dim, 0.0005, 0.01);

    block.norm1.gamma = Array2::from_shape_fn((1, dim), |(_, c)| 1.0 + c as f32 * 0.01);
    block.norm1.beta = Array2::from_shape_fn((1, dim), |(_, c)| -0.02 + c as f32 * 0.005);
    block.norm2.gamma = Array2::from_shape_fn((1, dim), |(_, c)| 0.9 + c as f32 * 0.01);
    block.norm2.beta = Array2::from_shape_fn((1, dim), |(_, c)| 0.01 - c as f32 * 0.004);

    // 关闭 dropout，避免 forward/backward 比较受到随机 mask 影响。
    block.set_training_mode(false);
    block
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f32, |m, (&x, &y)| m.max((x - y).abs()))
}

#[test]
fn test_transformer_block_forward_shape() {
    let dim = 8;
    let hidden_dim = 16;
    let mut transformer = TransformerBlock::new(dim, hidden_dim);

    let input = Array2::ones((1, dim));
    let (output, _ctx) = transformer.forward(&input);

    assert_eq!(output.shape(), [1, dim]);
}

#[test]
fn transformer_block_backward_matches_accumulate_then_step_for_single_micro_batch() {
    let dim = 8;
    let hidden_dim = 16;
    let lr = 1e-3_f32;

    let input = Array2::from_shape_fn((3, dim), |(r, c)| 0.03 * r as f32 - 0.02 * c as f32 + 0.1);
    let grad_out =
        Array2::from_shape_fn((3, dim), |(r, c)| 0.02 * r as f32 + 0.01 * c as f32 - 0.05);

    let mut direct = make_transformer_block(dim, hidden_dim);
    let mut accum = make_transformer_block(dim, hidden_dim);

    let (_out_direct, ctx_direct) = direct.forward(&input);
    let grad_input_direct = direct.backward(&ctx_direct, &grad_out, lr);

    let (_out_accum, ctx_accum) = accum.forward(&input);
    accum.zero_grad_accum();
    let grad_input_accum = accum.backward_accumulate_with_ctx(&ctx_accum, &grad_out);
    accum.step_accumulated(lr, 1.0);

    let tol = 1e-5_f32;
    assert!(
        max_diff(&grad_input_direct, &grad_input_accum) < tol,
        "grad_input mismatch between direct backward and accumulate+step"
    );
    assert!(max_diff(&direct.attention.w_q, &accum.attention.w_q) < tol);
    assert!(max_diff(&direct.attention.w_k, &accum.attention.w_k) < tol);
    assert!(max_diff(&direct.attention.w_v, &accum.attention.w_v) < tol);
    assert!(max_diff(&direct.attention.w_o, &accum.attention.w_o) < tol);
    assert!(max_diff(&direct.feed_forward.w1, &accum.feed_forward.w1) < tol);
    assert!(max_diff(&direct.feed_forward.b1, &accum.feed_forward.b1) < tol);
    assert!(max_diff(&direct.feed_forward.w2, &accum.feed_forward.w2) < tol);
    assert!(max_diff(&direct.feed_forward.b2, &accum.feed_forward.b2) < tol);
    assert!(max_diff(&direct.norm1.gamma, &accum.norm1.gamma) < tol);
    assert!(max_diff(&direct.norm1.beta, &accum.norm1.beta) < tol);
    assert!(max_diff(&direct.norm2.gamma, &accum.norm2.gamma) < tol);
    assert!(max_diff(&direct.norm2.beta, &accum.norm2.beta) < tol);
}
