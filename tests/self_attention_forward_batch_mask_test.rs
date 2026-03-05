//! 验证 `Layer::forward_batch()` 路径下的注意力 padding mask 能真正影响注意力前向。
//!
//! 背景：SelfAttention 虽然已有 `forward_with_padding_mask()`，但如果 batch 前向不消费 mask，
//! padding token 仍可能参与注意力，导致 batch 向量化训练/推理时出现系统性噪声。

use llm::{self_attention::SelfAttention, Layer};
use ndarray::{s, Array2, Array3};

#[test]
fn self_attention_forward_batch_respects_padding_mask() {
    let dim = 8;
    let seq_len = 4;

    let mut attn = SelfAttention::new(dim);

    // 将投影矩阵设为单位阵，让 Q/K/V/O = input，保证断言稳定且易于推理。
    let eye = Array2::from_shape_fn((dim, dim), |(r, c)| if r == c { 1.0 } else { 0.0 });
    attn.w_q = eye.clone();
    attn.w_k = eye.clone();
    attn.w_v = eye.clone();
    attn.w_o = eye;

    // 构造“左 padding”输入：前两位是 PAD（值很大），后两位是有效 token（值很小）。
    // 对于最后一个有效 token（位置3），因果掩码允许其 attend 到过去的 PAD key，
    // 因此 padding mask 必须生效，否则输出会被 PAD 主导。
    let mut input_batch = Array3::zeros((1, seq_len, dim));
    for s in 0..seq_len {
        let val = if s < 2 { 100.0 } else { 1.0 };
        for d in 0..dim {
            input_batch[[0, s, d]] = val;
        }
    }

    let attention_mask = Array2::from_shape_vec((1, seq_len), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

    let (out_unmasked, _ctx_unmasked) = attn.forward_batch(&input_batch, None);
    let (out_masked, _ctx_masked) = attn.forward_batch(&input_batch, Some(&attention_mask));

    let max_abs_unmasked = out_unmasked
        .slice(s![0, 3, ..])
        .iter()
        .fold(0.0_f32, |m, &v| m.max(v.abs()));
    let max_abs_masked = out_masked
        .slice(s![0, 3, ..])
        .iter()
        .fold(0.0_f32, |m, &v| m.max(v.abs()));

    assert!(
        max_abs_unmasked > 50.0,
        "expected large output when PAD is not masked; max_abs={}",
        max_abs_unmasked
    );
    assert!(
        max_abs_masked < 10.0,
        "expected small output when PAD is masked; max_abs={}",
        max_abs_masked
    );
}
