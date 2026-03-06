use ndarray::Array2;

use llm::{Layer, output_projection::OutputProjection};

fn make_w_out(embedding_dim: usize, vocab_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((embedding_dim, vocab_size), |(r, c)| {
        let v = 0.01 * (r as f32 + 1.0) + 0.005 * (c as f32);
        if (r + c) % 2 == 0 { v } else { -v }
    })
}

fn make_b_out(vocab_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((1, vocab_size), |(_, c)| 0.02 * (c as f32) - 0.03)
}

fn make_op_with_params(
    embedding_dim: usize,
    vocab_size: usize,
    w_out: &Array2<f32>,
    b_out: &Array2<f32>,
) -> OutputProjection {
    let mut op = OutputProjection::new(embedding_dim, vocab_size);
    op.w_out = w_out.clone();
    op.b_out = b_out.clone();
    op
}

fn compute_loss(op: &mut OutputProjection, input: &Array2<f32>, grad_out: &Array2<f32>) -> f32 {
    let (out, _ctx) = op.forward(input);
    (&out * grad_out).sum()
}

#[test]
fn output_projection_parameter_gradients_match_numerical() {
    let embedding_dim = 4;
    let vocab_size = 6;
    let seq_len = 3;

    let input = Array2::from_shape_fn((seq_len, embedding_dim), |(r, c)| {
        0.1 * (r as f32) + 0.05 * (c as f32) - 0.02
    });
    let grad_out = Array2::from_shape_fn((seq_len, vocab_size), |(r, c)| {
        0.02 * (r as f32) - 0.01 * (c as f32) + 0.005
    });

    let w_out = make_w_out(embedding_dim, vocab_size);
    let b_out = make_b_out(vocab_size);

    let mut op = make_op_with_params(embedding_dim, vocab_size, &w_out, &b_out);
    let (_out, ctx) = op.forward(&input);
    op.zero_grad_accum();
    let _ = op.backward_accumulate_with_ctx(&ctx, &grad_out);

    let grad_w = op.grad_w_out_accum.clone();
    let grad_b = op.grad_b_out_accum.clone();

    let eps = 1e-3_f32;
    let tol = 1e-2_f32;

    // 抽样若干权重位置
    for &(r, c) in &[(0usize, 0usize), (1, 3), (3, 5)] {
        let mut pos = w_out.clone();
        pos[[r, c]] += eps;
        let mut neg = w_out.clone();
        neg[[r, c]] -= eps;

        let loss_pos = compute_loss(
            &mut make_op_with_params(embedding_dim, vocab_size, &pos, &b_out),
            &input,
            &grad_out,
        );
        let loss_neg = compute_loss(
            &mut make_op_with_params(embedding_dim, vocab_size, &neg, &b_out),
            &input,
            &grad_out,
        );
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_w[[r, c]];
        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < tol,
            "w_out grad mismatch at ({}, {}): num={}, ana={}, rel_err={}",
            r,
            c,
            num,
            ana,
            rel_err
        );
    }

    // bias：全量验证（vocab_size 很小）
    for c in 0..vocab_size {
        let mut pos = b_out.clone();
        pos[[0, c]] += eps;
        let mut neg = b_out.clone();
        neg[[0, c]] -= eps;

        let loss_pos = compute_loss(
            &mut make_op_with_params(embedding_dim, vocab_size, &w_out, &pos),
            &input,
            &grad_out,
        );
        let loss_neg = compute_loss(
            &mut make_op_with_params(embedding_dim, vocab_size, &w_out, &neg),
            &input,
            &grad_out,
        );
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_b[[0, c]];
        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < tol,
            "b_out grad mismatch at (0, {}): num={}, ana={}, rel_err={}",
            c,
            num,
            ana,
            rel_err
        );
    }
}
