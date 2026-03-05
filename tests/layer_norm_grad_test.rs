use ndarray::Array2;

use llm::{Layer, layer_norm::LayerNorm};

/// 用有限差分验证 LayerNorm 的梯度（输入、gamma、beta）。
///
/// 选择 loss = sum(output * grad_out)，其解析梯度应与 backward 结果一致。
#[test]
fn layer_norm_gradients_match_numerical() {
    let seq_len = 2;
    let dim = 3;

    let input = Array2::from_shape_vec(
        (seq_len, dim),
        vec![
            0.3, -1.2, 2.0, //
            1.5, 0.7, -0.4, //
        ],
    )
    .unwrap();

    let grad_out = Array2::from_shape_vec(
        (seq_len, dim),
        vec![
            0.2, -0.3, 0.1, //
            -0.4, 0.5, 0.25, //
        ],
    )
    .unwrap();

    // 构造一个非平凡参数，避免“全 1/全 0”导致某些误差被掩盖。
    let gamma = Array2::from_shape_vec((1, dim), vec![1.2, -0.7, 0.3]).unwrap();
    let beta = Array2::from_shape_vec((1, dim), vec![0.1, 0.2, -0.1]).unwrap();

    // 解析梯度（来自 backward_accumulate，不更新参数）
    let mut ln = LayerNorm::new(dim);
    ln.gamma = gamma.clone();
    ln.beta = beta.clone();

    let (_y, _ctx) = ln.forward(&input);
    ln.zero_grad_accum();
    let grad_input = ln.backward_accumulate(&grad_out);
    let grad_gamma = ln.grad_gamma_accum.clone();
    let grad_beta = ln.grad_beta_accum.clone();

    // 数值梯度
    let eps = 1e-3_f32;

    let compute_loss = |gamma: &Array2<f32>, beta: &Array2<f32>, input: &Array2<f32>| -> f32 {
        let mut ln = LayerNorm::new(dim);
        ln.gamma = gamma.clone();
        ln.beta = beta.clone();
        let (y, _ctx) = ln.forward(input);
        (&y * &grad_out).sum()
    };

    // input grads
    for r in 0..seq_len {
        for c in 0..dim {
            let mut x_pos = input.clone();
            x_pos[[r, c]] += eps;
            let mut x_neg = input.clone();
            x_neg[[r, c]] -= eps;
            let loss_pos = compute_loss(&gamma, &beta, &x_pos);
            let loss_neg = compute_loss(&gamma, &beta, &x_neg);
            let num = (loss_pos - loss_neg) / (2.0 * eps);
            let ana = grad_input[[r, c]];

            let denom = 1.0_f32.max(num.abs()).max(ana.abs());
            let rel_err = (num - ana).abs() / denom;
            assert!(
                rel_err < 1e-2,
                "grad_input mismatch at ({}, {}): num={}, ana={}, rel_err={}",
                r,
                c,
                num,
                ana,
                rel_err
            );
        }
    }

    // gamma grads
    for c in 0..dim {
        let mut g_pos = gamma.clone();
        g_pos[[0, c]] += eps;
        let mut g_neg = gamma.clone();
        g_neg[[0, c]] -= eps;
        let loss_pos = compute_loss(&g_pos, &beta, &input);
        let loss_neg = compute_loss(&g_neg, &beta, &input);
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_gamma[[0, c]];

        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < 1e-2,
            "grad_gamma mismatch at (0, {}): num={}, ana={}, rel_err={}",
            c,
            num,
            ana,
            rel_err
        );
    }

    // beta grads
    for c in 0..dim {
        let mut b_pos = beta.clone();
        b_pos[[0, c]] += eps;
        let mut b_neg = beta.clone();
        b_neg[[0, c]] -= eps;
        let loss_pos = compute_loss(&gamma, &b_pos, &input);
        let loss_neg = compute_loss(&gamma, &b_neg, &input);
        let num = (loss_pos - loss_neg) / (2.0 * eps);
        let ana = grad_beta[[0, c]];

        let denom = 1.0_f32.max(num.abs()).max(ana.abs());
        let rel_err = (num - ana).abs() / denom;
        assert!(
            rel_err < 1e-2,
            "grad_beta mismatch at (0, {}): num={}, ana={}, rel_err={}",
            c,
            num,
            ana,
            rel_err
        );
    }
}
