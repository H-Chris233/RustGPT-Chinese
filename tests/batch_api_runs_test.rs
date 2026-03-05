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

    // dummy grads: same shape as out
    let grads = Array3::<f32>::ones(out.dim());
    let grad_input = attn.backward_batch(&ctxs, &grads, 0.001, None);
    assert_eq!(grad_input.dim(), input.dim());
    assert!(grad_input.iter().all(|v| v.is_finite()));
}

