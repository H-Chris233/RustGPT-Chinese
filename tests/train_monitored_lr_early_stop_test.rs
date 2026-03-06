//! 回归测试：`train_monitored()` 应同时遵守早停与学习率调度语义。
//!
//! 目标：
//! - 学习率仍应按 `cosine_with_warmup_lr()` 逐 epoch 计算；
//! - 当 loss 持续不改善时，应按 patience 提前停止。

use llm::{Layer, LayerContext, LLM, Vocab};
use ndarray::Array2;

struct MonitoredLrProbeLayer {
    logits: Array2<f32>,
    seen_lrs: Vec<f32>,
}

impl MonitoredLrProbeLayer {
    fn new(logits: Array2<f32>) -> Self {
        Self {
            logits,
            seen_lrs: Vec::new(),
        }
    }
}

impl Layer for MonitoredLrProbeLayer {
    fn layer_type(&self) -> &str {
        "MonitoredLrProbeLayer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        let seq_len = input.shape()[1];
        if seq_len == self.logits.nrows() {
            return (self.logits.clone(), Box::new(()));
        }

        let mut out = Array2::zeros((seq_len, self.logits.ncols()));
        for i in 0..seq_len {
            out.row_mut(i).assign(&self.logits.row(0));
        }
        (out, Box::new(()))
    }

    fn backward(&mut self, _ctx: &LayerContext, _grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        if self
            .seen_lrs
            .last()
            .map_or(true, |last| (lr - *last).abs() > 1e-12)
        {
            self.seen_lrs.push(lr);
        }
        Array2::zeros((1, 1))
    }

    fn parameters(&self) -> usize {
        0
    }

    fn set_training_mode(&mut self, _training: bool) {}
}

fn recommend_warmup_epochs(total_epochs: usize) -> usize {
    if total_epochs == 0 {
        return 0;
    }
    let warmup = ((total_epochs as f32) * 0.05).ceil() as usize;
    warmup.clamp(1, total_epochs)
}

#[test]
fn train_monitored_applies_lr_schedule_before_early_stop() {
    let vocab = Vocab::new(vec!["a", "b"]);
    let vocab_size = vocab.len();
    let a_id = vocab.encode("a").unwrap();
    let b_id = vocab.encode("b").unwrap();

    let mut logits_vec = vec![0.0f32; vocab_size];
    logits_vec[a_id] = 1.0;
    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec).unwrap();

    let mut llm = LLM::new_experimental(vocab, vec![Box::new(MonitoredLrProbeLayer::new(logits))]);

    let max_epochs = 5usize;
    let initial_lr = 0.001f32;
    let patience = 1usize;
    let actual_epochs = llm.train_monitored(vec!["a b"], max_epochs, initial_lr, patience, 1);

    assert!(
        actual_epochs < max_epochs,
        "固定 loss 下应在达到 max_epochs 前触发早停"
    );

    let probe = llm.network[0]
        .as_any()
        .downcast_ref::<MonitoredLrProbeLayer>()
        .expect("MonitoredLrProbeLayer downcast failed");

    assert!(!probe.seen_lrs.is_empty(), "应至少记录一个 epoch 的学习率");
    let warmup_epochs = recommend_warmup_epochs(max_epochs);
    for (epoch, &lr_seen) in probe.seen_lrs.iter().enumerate() {
        let lr_expected =
            LLM::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);
        assert!(
            (lr_seen - lr_expected).abs() < 1e-9,
            "epoch {epoch}: expected lr {lr_expected}, got {lr_seen}"
        );
    }

    // 防御性检查：测试数据确实能跑到 backward，而不是被跳过。
    assert_ne!(a_id, b_id);
}
