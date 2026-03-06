//! 回归测试：batch API 应当可以执行（不依赖层内 cached_*），并返回正确形状的结果。
//!
//! 教学说明：
//! - 旧版 Layer 的默认 batch 实现会在 batch 内循环 forward 覆盖 self.cached_*，
//!   导致 backward_batch 使用“最后一个样本”的缓存，产生静默错误梯度。
//! - 新版 Layer 通过显式 ctx（forward 返回 ctx，backward 接收 ctx）解决该问题。
//! - 因此：我们期望默认 batch 实现 **能够运行**，且 backward_batch 能在提供 ctx 的前提下正常执行。

use llm::{Layer, EMBEDDING_DIM};
use ndarray::Array3;

#[test]
fn self_attention_forward_backward_batch_runs() {
    let mut attn = llm::self_attention::SelfAttention::new(EMBEDDING_DIM);

    // (batch=2, seq=4, dim)
    let input = Array3::<f32>::ones((2, 4, EMBEDDING_DIM));
    let (out, ctxs) = attn.forward_batch(&input, None);
    assert_eq!(out.dim(), (2, 4, EMBEDDING_DIM));
    assert_eq!(ctxs.len(), 2);

    // 构造与输出同形状的测试梯度。
    let grads = Array3::<f32>::ones(out.dim());
    let grad_input = attn.backward_batch(&ctxs, &grads, 0.001, None);
    assert_eq!(grad_input.dim(), input.dim());
    assert!(grad_input.iter().all(|v| v.is_finite()));
}

#[test]
fn forward_batch_default_mask_contract_zeroes_masked_rows() {
    use llm::layer_norm::LayerNorm;
    use ndarray::{Array2, Array3};

    let mut layer = LayerNorm::new(4);
    let input = Array3::from_shape_fn((1, 3, 4), |(_, s, d)| 1.0 + s as f32 + d as f32 * 0.1);
    let attention_mask = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 1.0]).unwrap();

    let (out, ctxs) = layer.forward_batch(&input, Some(&attention_mask));
    assert_eq!(out.dim(), (1, 3, 4));
    assert_eq!(ctxs.len(), 1);
    assert!(
        out.slice(ndarray::s![0, 1, ..])
            .iter()
            .all(|v| v.abs() < 1e-6)
    );

    let grads = Array3::<f32>::ones(out.dim());
    let grad_input = layer.backward_batch(&ctxs, &grads, 0.001, Some(&attention_mask));
    assert!(
        grad_input
            .slice(ndarray::s![0, 1, ..])
            .iter()
            .all(|v| v.abs() < 1e-6)
    );
}
