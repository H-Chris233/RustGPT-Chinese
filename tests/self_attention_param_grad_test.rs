use ndarray::Array2;

use llm::{Layer, self_attention::SelfAttention};

fn make_weight(dim: usize, scale: f32, offset: f32) -> Array2<f32> {
    Array2::from_shape_fn((dim, dim), |(r, c)| {
        let idx = (r * dim + c) as f32;
        let v = idx * scale + offset;
        if (r + c) % 2 == 0 { v } else { -v }
    })
}

fn make_attn_with_weights(
    dim: usize,
    w_q: &Array2<f32>,
    w_k: &Array2<f32>,
    w_v: &Array2<f32>,
    w_o: &Array2<f32>,
) -> SelfAttention {
    let mut attn = SelfAttention::new(dim);
    attn.w_q = w_q.clone();
    attn.w_k = w_k.clone();
    attn.w_v = w_v.clone();
    attn.w_o = w_o.clone();
    attn
}

fn compute_loss(attn: &mut SelfAttention, input: &Array2<f32>, grad_out: &Array2<f32>) -> f32 {
    let (out, _ctx) = attn.forward(input);
    (&out * grad_out).sum()
}

/// 对 SelfAttention 的参数梯度做有限差分校验（抽样元素）。
///
/// loss 定义为：`L = sum(output * grad_out)`，则 `grad_out = dL/doutput`。
#[test]
fn self_attention_parameter_gradients_match_numerical() {
    let dim = 16;
    let seq_len = 4;

    // 固定输入与上游梯度，保证测试可复现
    let input = Array2::from_shape_fn((seq_len, dim), |(r, c)| {
        0.03 * (r as f32) + 0.01 * (c as f32) - 0.02
    });
    let grad_out = Array2::from_shape_fn((seq_len, dim), |(r, c)| {
        0.02 * (r as f32) - 0.015 * (c as f32) + 0.001 * ((r + c) as f32)
    });

    // 固定权重，避免随机初始化导致偶发误差/不可复现
    let w_q = make_weight(dim, 0.001, 0.01);
    let w_k = make_weight(dim, 0.001, -0.02);
    let w_v = make_weight(dim, 0.001, 0.03);
    let w_o = make_weight(dim, 0.001, -0.04);

    // 解析梯度：通过 backward_accumulate_with_ctx(ctx, grads) 计算并写入 grad_w_*_accum
    let mut attn = make_attn_with_weights(dim, &w_q, &w_k, &w_v, &w_o);
    let (_out, ctx) = attn.forward(&input);
    attn.zero_grad_accum();
    let _ = attn.backward_accumulate_with_ctx(&ctx, &grad_out);
    let grad_w_q = attn.grad_w_q_accum.clone();
    let grad_w_k = attn.grad_w_k_accum.clone();
    let grad_w_v = attn.grad_w_v_accum.clone();
    let grad_w_o = attn.grad_w_o_accum.clone();

    let eps = 1e-3_f32;
    let tol = 2e-2_f32; // 注意力含 softmax，允许略大误差

    // 抽样若干参数位置（避免测试过慢）
    let samples: &[(usize, usize)] = &[
        (0, 0),
        (1, 2),
        (3, 5),
        (7, 8),
        (15, 15),
    ];

    let check = |name: &str,
                 base: &Array2<f32>,
                 grad_ana: &Array2<f32>,
                 loss_fn: &dyn Fn(&Array2<f32>) -> f32| {
        for &(r, c) in samples {
            let mut pos = base.clone();
            pos[[r, c]] += eps;
            let mut neg = base.clone();
            neg[[r, c]] -= eps;
            let num = (loss_fn(&pos) - loss_fn(&neg)) / (2.0 * eps);
            let ana = grad_ana[[r, c]];

            let denom = 1.0_f32.max(num.abs()).max(ana.abs());
            let rel_err = (num - ana).abs() / denom;
            assert!(
                rel_err < tol,
                "{} grad mismatch at ({}, {}): num={}, ana={}, rel_err={}",
                name,
                r,
                c,
                num,
                ana,
                rel_err
            );
        }
    };

    // w_q
    check(
        "w_q",
        &w_q,
        &grad_w_q,
        &|w_q_perturbed| {
            let mut a = make_attn_with_weights(dim, w_q_perturbed, &w_k, &w_v, &w_o);
            compute_loss(&mut a, &input, &grad_out)
        },
    );

    // w_k
    check(
        "w_k",
        &w_k,
        &grad_w_k,
        &|w_k_perturbed| {
            let mut a = make_attn_with_weights(dim, &w_q, w_k_perturbed, &w_v, &w_o);
            compute_loss(&mut a, &input, &grad_out)
        },
    );

    // w_v
    check(
        "w_v",
        &w_v,
        &grad_w_v,
        &|w_v_perturbed| {
            let mut a = make_attn_with_weights(dim, &w_q, &w_k, w_v_perturbed, &w_o);
            compute_loss(&mut a, &input, &grad_out)
        },
    );

    // w_o
    check(
        "w_o",
        &w_o,
        &grad_w_o,
        &|w_o_perturbed| {
            let mut a = make_attn_with_weights(dim, &w_q, &w_k, &w_v, w_o_perturbed);
            compute_loss(&mut a, &input, &grad_out)
        },
    );
}
