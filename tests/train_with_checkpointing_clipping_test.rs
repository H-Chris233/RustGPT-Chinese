//! 回归测试：`train_with_checkpointing` 必须对输出梯度做裁剪（clip=1.0）
//!
//! 说明：
//! - 这里用一个“探针 Layer”截获传入 backward 的梯度，验证其 L2 范数不超过 1.0。
//! - 如果未来有人把裁剪阈值改回 5.0 或移除裁剪，这个测试会立刻失败。

use llm::{Layer, LLM, Vocab};
use llm::LayerContext;
use ndarray::Array2;

struct ProbeLayer {
    logits: Array2<f32>,
    pub seen_grad_norms: Vec<f32>,
}

impl ProbeLayer {
    fn new(logits: Array2<f32>) -> Self {
        Self {
            logits,
            seen_grad_norms: Vec::new(),
        }
    }
}

impl Layer for ProbeLayer {
    fn layer_type(&self) -> &str {
        "ProbeLayer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        // 训练入口输入为 (1, seq_len)，输出 logits 为 (seq_len, vocab_size)。
        let seq_len = input.shape()[1];
        if seq_len == self.logits.nrows() {
            return (self.logits.clone(), Box::new(()));
        }

        // 兼容性：必要时复制第 0 行。
        let mut out = Array2::zeros((seq_len, self.logits.ncols()));
        for i in 0..seq_len {
            out.row_mut(i).assign(&self.logits.row(0));
        }
        (out, Box::new(()))
    }

    fn backward(&mut self, _ctx: &LayerContext, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        self.seen_grad_norms.push(LLM::compute_grad_norm(grads));
        // 返回任意形状一致的输入梯度（该测试中不会被使用）
        Array2::zeros((1, grads.nrows()))
    }

    fn parameters(&self) -> usize {
        0
    }

    fn set_training_mode(&mut self, _training: bool) {}
}

#[test]
fn train_with_checkpointing_clips_grads_to_l2_norm_1() {
    // 构造一个很小的 vocab（仍会包含默认特殊 token）
    let vocab = Vocab::new(vec!["a", "b", "c"]);
    let vocab_size = vocab.len();

    let a_id = vocab.encode("a").unwrap();
    let b_id = vocab.encode("b").unwrap();

    // logits 让模型极度偏向 a，但 target 是 b，使得未裁剪梯度 L2 范数接近 sqrt(2) > 1。
    let mut logits_vec = vec![0.0f32; vocab_size];
    logits_vec[a_id] = 1000.0;
    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec).unwrap();

    let mut llm = LLM::new(vocab, vec![Box::new(ProbeLayer::new(logits))]);

    let tokenized_data = vec![vec![a_id, b_id]]; // input=[a], target=[b]

    llm.train_with_checkpointing(
        tokenized_data,
        1,
        0.001,
        100,
        None,
        "clip_test",
        0,
    );

    let probe = llm.network[0]
        .as_any()
        .downcast_ref::<ProbeLayer>()
        .expect("ProbeLayer downcast failed");

    assert_eq!(probe.seen_grad_norms.len(), 1);
    let norm = probe.seen_grad_norms[0];
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected clipped grad norm ≈ 1.0, got {}",
        norm
    );
}
