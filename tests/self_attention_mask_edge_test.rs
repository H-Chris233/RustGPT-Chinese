use ndarray::{Array1, Array2};

use llm::self_attention::SelfAttention;

/// 当 padding mask 把所有 key 都屏蔽时，注意力权重应为全 0（而不是均匀分布），
/// 从而输出为全 0，避免把 PAD/value 当作有效信号回流。
#[test]
fn self_attention_all_keys_masked_outputs_zero() {
    let seq_len = 3;
    let dim = 16;

    let mut attn = SelfAttention::new(dim);

    // 构造确定性输入（避免随机导致断言不稳定）。
    let input = Array2::from_shape_fn((seq_len, dim), |(r, c)| (r as f32) * 0.1 + (c as f32) * 0.01);

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

