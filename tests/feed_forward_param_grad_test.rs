use ndarray::Array2;

use llm::{Layer, feed_forward::FeedForward};

fn make_w1(embedding_dim: usize, hidden_dim: usize) -> Array2<f32> {
    Array2::from_shape_fn((embedding_dim, hidden_dim), |(r, c)| {
        let v = 0.05 * (r as f32 + 1.0) + 0.01 * (c as f32);
        if c % 2 == 0 { v } else { -v }
    })
}

fn make_w2(hidden_dim: usize, embedding_dim: usize) -> Array2<f32> {
    Array2::from_shape_fn((hidden_dim, embedding_dim), |(r, c)| {
        let v = 0.03 * (r as f32 + 1.0) - 0.02 * (c as f32);
        if r % 2 == 0 { v } else { -v }
    })
}

fn make_ffn_with_params(
    embedding_dim: usize,
    hidden_dim: usize,
    w1: &Array2<f32>,
    b1: &Array2<f32>,
    w2: &Array2<f32>,
    b2: &Array2<f32>,
) -> FeedForward {
    let mut ffn = FeedForward::new(embedding_dim, hidden_dim);
    ffn.w1 = w1.clone();
    ffn.b1 = b1.clone();
    ffn.w2 = w2.clone();
    ffn.b2 = b2.clone();
    ffn
}

fn compute_loss(ffn: &mut FeedForward, input: &Array2<f32>, grad_out: &Array2<f32>) -> f32 {
    let (out, _ctx) = ffn.forward(input);
    (&out * grad_out).sum()
}

#[test]
fn feed_forward_parameter_gradients_match_numerical() {
    let embedding_dim = 4;
    let hidden_dim = 6;
    let seq_len = 3;

    let input = Array2::from_shape_vec(
        (seq_len, embedding_dim),
        vec![
            0.5, 1.0, 0.3, 0.7, //
            0.2, 0.4, 1.2, 0.1, //
            0.9, 0.8, 0.6, 0.2, //
        ],
    )
    .unwrap();
    let grad_out = Array2::from_shape_fn((seq_len, embedding_dim), |(r, c)| {
        0.02 * (r as f32) - 0.03 * (c as f32) + 0.01
    });

    // 固定参数（保证可复现），并让 ReLU 的激活/失活有明确 margin，避免在 0 附近数值不稳定。
    let w1 = make_w1(embedding_dim, hidden_dim);
    let w2 = make_w2(hidden_dim, embedding_dim);
    let b1 = Array2::from_shape_fn((1, hidden_dim), |(_, c)| if c % 2 == 0 { 1.0 } else { -1.0 });
    let b2 = Array2::from_shape_vec((1, embedding_dim), vec![0.1, -0.2, 0.05, 0.0]).unwrap();

    let mut ffn = make_ffn_with_params(embedding_dim, hidden_dim, &w1, &b1, &w2, &b2);
    let (_out, ctx) = ffn.forward(&input);

    // 确保 ReLU 的输入远离 0（不可导点），避免有限差分在边界附近波动。
    //
    // 注意：这里不要依赖 `ffn.hidden_pre_activation` 等旧式缓存字段。
    // 因为本轮重构的目标之一，就是让梯度/累积链路可以逐步摆脱 `cached_*` 字段。
    //
    // 由于本测试已经固定了 (input, w1, b1)，我们直接按定义计算 pre-activation：
    //   h_pre = input · W1 + b1
    let h_pre = input.dot(&w1) + &b1;
    let min_abs = h_pre
        .iter()
        .fold(f32::INFINITY, |m, &v| m.min(v.abs()));
    assert!(
        min_abs > 1e-2,
        "ReLU pre-activation too close to 0; test construction unstable (min_abs={})",
        min_abs
    );
    ffn.zero_grad_accum();
    let _ = ffn.backward_accumulate_with_ctx(&ctx, &grad_out);

    let grad_w1 = ffn.grad_w1_accum.clone();
    let grad_b1 = ffn.grad_b1_accum.clone();
    let grad_w2 = ffn.grad_w2_accum.clone();
    let grad_b2 = ffn.grad_b2_accum.clone();

    let eps = 1e-3_f32;
    let tol = 1e-2_f32;

    // 抽样参数位置
    let samples_w1: &[(usize, usize)] = &[(0, 0), (1, 2), (3, 4)];
    let samples_w2: &[(usize, usize)] = &[(0, 0), (2, 1), (5, 3)];

    // w1
    for &(r, c) in samples_w1 {
        let mut pos = w1.clone();
        pos[[r, c]] += eps;
        let mut neg = w1.clone();
        neg[[r, c]] -= eps;

        let loss_pos = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &pos, &b1, &w2, &b2),
            &input,
            &grad_out,
        );
        let loss_neg = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &neg, &b1, &w2, &b2),
            &input,
            &grad_out,
        );
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_w1[[r, c]];
        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < tol,
            "w1 grad mismatch at ({}, {}): num={}, ana={}, rel_err={}",
            r,
            c,
            num,
            ana,
            rel_err
        );
    }

    // b1（抽样 3 个位置）
    for c in [0usize, 1usize, 4usize] {
        let mut pos = b1.clone();
        pos[[0, c]] += eps;
        let mut neg = b1.clone();
        neg[[0, c]] -= eps;

        let loss_pos = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &w1, &pos, &w2, &b2),
            &input,
            &grad_out,
        );
        let loss_neg = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &w1, &neg, &w2, &b2),
            &input,
            &grad_out,
        );
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_b1[[0, c]];
        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < tol,
            "b1 grad mismatch at (0, {}): num={}, ana={}, rel_err={}",
            c,
            num,
            ana,
            rel_err
        );
    }

    // w2
    for &(r, c) in samples_w2 {
        let mut pos = w2.clone();
        pos[[r, c]] += eps;
        let mut neg = w2.clone();
        neg[[r, c]] -= eps;

        let loss_pos = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &w1, &b1, &pos, &b2),
            &input,
            &grad_out,
        );
        let loss_neg = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &w1, &b1, &neg, &b2),
            &input,
            &grad_out,
        );
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_w2[[r, c]];
        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < tol,
            "w2 grad mismatch at ({}, {}): num={}, ana={}, rel_err={}",
            r,
            c,
            num,
            ana,
            rel_err
        );
    }

    // b2
    for c in 0..embedding_dim {
        let mut pos = b2.clone();
        pos[[0, c]] += eps;
        let mut neg = b2.clone();
        neg[[0, c]] -= eps;

        let loss_pos = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &w1, &b1, &w2, &pos),
            &input,
            &grad_out,
        );
        let loss_neg = compute_loss(
            &mut make_ffn_with_params(embedding_dim, hidden_dim, &w1, &b1, &w2, &neg),
            &input,
            &grad_out,
        );
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_b2[[0, c]];
        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < tol,
            "b2 grad mismatch at (0, {}): num={}, ana={}, rel_err={}",
            c,
            num,
            ana,
            rel_err
        );
    }
}
