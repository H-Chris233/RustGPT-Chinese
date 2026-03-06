use ndarray::{Array1, Array2, Array3};

use llm::{Layer, self_attention::SelfAttention};

/// 当 padding mask 把所有 key 都屏蔽时，注意力权重应为全 0（而不是均匀分布），
/// 从而输出为全 0，避免把 PAD/value 当作有效信号回流。
#[test]
fn self_attention_all_keys_masked_outputs_zero() {
    let seq_len = 3;
    let dim = 16;

    let mut attn = SelfAttention::new(dim);

    // 构造确定性输入（避免随机导致断言不稳定）。
    let input = Array2::from_shape_fn((seq_len, dim), |(r, c)| {
        (r as f32) * 0.1 + (c as f32) * 0.01
    });

    // 全部是 PAD：key_padding_mask[j] = 0 => 该列被置为 -∞
    let key_padding_mask = Array1::zeros(seq_len);

    let out = attn.forward_with_padding_mask(&input, Some(&key_padding_mask));

    // 旧行为：stable_softmax 在 sum_exp 很小会回退均匀分布，从而 out 一般不为 0。
    // 新行为：应为全 0（或极小浮点误差）。
    let max_abs = out.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    assert!(
        max_abs < 1e-6,
        "expected near-zero output when all keys masked; max_abs={}",
        max_abs
    );
}

#[test]
fn self_attention_all_keys_masked_backward_produces_near_zero_param_grads() {
    let seq_len = 3;
    let dim = 8;
    let mut attn = SelfAttention::new(dim);

    // 用单位阵让行为更稳定、可解释。
    let eye = Array2::from_shape_fn((dim, dim), |(r, c)| if r == c { 1.0 } else { 0.0 });
    attn.w_q = eye.clone();
    attn.w_k = eye.clone();
    attn.w_v = eye.clone();
    attn.w_o = eye;

    let input = Array3::from_shape_fn((1, seq_len, dim), |(_, r, c)| (r as f32) + (c as f32) * 0.1);
    let attention_mask = Array2::zeros((1, seq_len));

    let (_out, ctxs) = attn.forward_batch(&input, Some(&attention_mask));
    let grad_out = Array2::ones((seq_len, dim));

    attn.zero_grad_accum();
    let grad_input = attn.backward_accumulate_with_ctx(&ctxs[0], &grad_out);

    let max_abs = |arr: &Array2<f32>| arr.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));

    assert!(
        grad_input.iter().all(|v| v.is_finite()),
        "grad_input should stay finite"
    );
    assert!(
        max_abs(&attn.grad_w_q_accum) < 1e-6,
        "grad_w_q should be near zero when all keys are masked"
    );
    assert!(
        max_abs(&attn.grad_w_k_accum) < 1e-6,
        "grad_w_k should be near zero when all keys are masked"
    );
    assert!(
        max_abs(&attn.grad_w_v_accum) < 1e-6,
        "grad_w_v should be near zero when all keys are masked"
    );
    assert!(
        max_abs(&attn.grad_w_o_accum) < 1e-6,
        "grad_w_o should be near zero when all keys are masked"
    );
}

#[test]
fn self_attention_masked_key_value_rows_do_not_change_wk_wv_grads() {
    let seq_len = 4;
    let dim = 8;
    let attention_mask = Array2::from_shape_vec((1, seq_len), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
    let grad_out = Array2::from_shape_fn((seq_len, dim), |(r, c)| {
        if r >= 2 {
            0.0
        } else {
            0.1 * (r as f32 + 1.0) - 0.03 * (c as f32)
        }
    });

    let eye = Array2::from_shape_fn((dim, dim), |(r, c)| if r == c { 1.0 } else { 0.0 });

    let build_input = |masked_value: f32| {
        let mut input = Array3::zeros((1, seq_len, dim));
        for c in 0..dim {
            input[[0, 0, c]] = 1.0 + c as f32 * 0.01;
            input[[0, 1, c]] = 2.0 + c as f32 * 0.01;
            input[[0, 2, c]] = masked_value;
            input[[0, 3, c]] = masked_value * 2.0;
        }
        input
    };

    let run = |masked_value: f32| {
        let mut attn = SelfAttention::new(dim);
        attn.w_q = eye.clone();
        attn.w_k = eye.clone();
        attn.w_v = eye.clone();
        attn.w_o = eye.clone();

        let input = build_input(masked_value);
        let (_out, ctxs) = attn.forward_batch(&input, Some(&attention_mask));
        attn.zero_grad_accum();
        let _ = attn.backward_accumulate_with_ctx(&ctxs[0], &grad_out);
        (attn.grad_w_k_accum.clone(), attn.grad_w_v_accum.clone())
    };

    let (grad_w_k_a, grad_w_v_a) = run(10.0);
    let (grad_w_k_b, grad_w_v_b) = run(1000.0);

    let max_diff = |a: &Array2<f32>, b: &Array2<f32>| {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f32, |m, (&x, &y)| m.max((x - y).abs()))
    };

    assert!(
        max_diff(&grad_w_k_a, &grad_w_k_b) < 1e-5,
        "masked key rows should not affect W_k gradients"
    );
    assert!(
        max_diff(&grad_w_v_a, &grad_w_v_b) < 1e-5,
        "masked value rows should not affect W_v gradients"
    );
}
