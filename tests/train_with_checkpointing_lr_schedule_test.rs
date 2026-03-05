//! 回归测试：`train_with_checkpointing` 的学习率调度必须与 `train_monitored` 对齐
//!
//! 关键约束（当前仓库约定）：余弦退火 + warmup，且 `num_restarts=0`。

use llm::{Layer, LLM, Vocab};
use llm::LayerContext;
use ndarray::Array2;

struct LrProbeLayer {
    logits: Array2<f32>,
    pub seen_lrs: Vec<f32>,
}

impl LrProbeLayer {
    fn new(logits: Array2<f32>) -> Self {
        Self {
            logits,
            seen_lrs: Vec::new(),
        }
    }
}

impl Layer for LrProbeLayer {
    fn layer_type(&self) -> &str {
        "LrProbeLayer"
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
        // 同一 epoch 内（多个样本）lr 通常相同；这里做去重，避免测试与 sample_count 强耦合。
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
fn train_with_checkpointing_uses_cosine_with_warmup_no_restarts() {
    let vocab = Vocab::new(vec!["a", "b"]);
    let vocab_size = vocab.len();

    let a_id = vocab.encode("a").unwrap();
    let b_id = vocab.encode("b").unwrap();

    // 随便给一个 logits（不需要极端），只要能跑通 forward/backward 并触发 lr 记录。
    let mut logits_vec = vec![0.0f32; vocab_size];
    logits_vec[a_id] = 1.0;
    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec).unwrap();

    let mut llm = LLM::new(vocab, vec![Box::new(LrProbeLayer::new(logits))]);
    let tokenized_data = vec![vec![a_id, b_id]];

    let max_epochs = 5usize;
    let resume_epoch = 3usize;
    let initial_lr = 0.001f32;
    let warmup_epochs = recommend_warmup_epochs(max_epochs);

    llm.train_with_checkpointing(
        tokenized_data,
        max_epochs,
        initial_lr,
        100, // 高 patience，避免早停影响观测
        None,
        "lr_schedule_test",
        resume_epoch,
    );

    let probe = llm.network[0]
        .as_any()
        .downcast_ref::<LrProbeLayer>()
        .expect("LrProbeLayer downcast failed");

    // 只包含 epoch=resume_epoch..max_epochs 的 lr 记录（本测试中每个 epoch 只有 1 个样本）
    assert_eq!(probe.seen_lrs.len(), max_epochs - resume_epoch);

    for (i, &lr_seen) in probe.seen_lrs.iter().enumerate() {
        let epoch = resume_epoch + i;
        let lr_expected = LLM::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);
        assert!(
            (lr_seen - lr_expected).abs() < 1e-9,
            "epoch {epoch}: expected lr {lr_expected}, got {lr_seen}"
        );
    }
}
