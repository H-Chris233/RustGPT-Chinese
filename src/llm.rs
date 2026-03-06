use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use rand::{rng, Rng};

use crate::{
    output_projection::OutputProjection,
    transformer::TransformerBlock,
    utils::{log_softmax, softmax},
    Embeddings, PerformanceMonitor, Vocab, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN, SOFTMAX_EPSILON,
};

#[path = "llm/inference_api.rs"]
mod inference_api;
pub use inference_api::{InferenceSession, SessionSnapshot};

#[derive(Clone)]
struct ProbEntry {
    prob: f32,
    idx: usize,
}

impl PartialEq for ProbEntry {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.prob.to_bits() == other.prob.to_bits()
    }
}

impl Eq for ProbEntry {}

impl PartialOrd for ProbEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProbEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .prob
            .total_cmp(&self.prob)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

// 当 temperature 足够接近 0 时，语义上应退化为 greedy / argmax。
// 这里给出一个保守阈值，避免概率幂变换在极端温度下把整行压成 0。
const GREEDY_TEMPERATURE_EPS: f32 = 1e-4;

use std::any::Any;

/// Layer 上下文：用于把 forward 产生的中间量（原本缓存在 self 内部的 cached_*）
/// 以“值”的形式返回给调用方，并在 backward 时显式传回，从而避免 batch 缓存覆盖。
pub type LayerContext = Box<dyn Any>;

/// Layer trait - 支持单样本和批量处理（显式上下文）
///
/// 设计目标：
/// - `forward()` 返回 `(output, ctx)`，其中 `ctx` 保存 backward 所需的中间量
/// - `backward()` 必须显式接收 `ctx`，避免依赖 self 内部缓存
///
/// 批量接口同理：返回/接收每个样本的 ctx，保证 batch 反传语义正确。
pub trait Layer {
    fn layer_type(&self) -> &str;

    /// 用于类型转换的辅助方法
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// 单样本前向传播
    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext);

    /// 单样本反向传播
    fn backward(&mut self, ctx: &LayerContext, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    /// 批量前向传播（默认实现：逐样本循环）
    ///
    /// 默认 mask 契约：
    /// - `attention_mask[b, s] < 0.5` 的位置会在层边界被视为“无效 token”；
    /// - 因此默认实现会把这些位置的**输出行清零**；
    /// - `backward_batch()` 也会对同样的位置把**输入梯度行清零**。
    ///
    /// 说明：
    /// - 这只保证“层边界的一致零化语义”；
    /// - 像注意力这类需要把 mask 参与内部计算（例如 key padding mask）的层，仍应覆写本方法。
    fn forward_batch(
        &mut self,
        input: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
    ) -> (Array3<f32>, Vec<LayerContext>) {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_dim = input.shape()[2];

        let mut output = Array3::zeros((batch_size, seq_len, hidden_dim));
        let mut ctxs: Vec<LayerContext> = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let sample = input.slice(ndarray::s![b, .., ..]).to_owned();
            let (mut sample_out, ctx) = self.forward(&sample);

            if let Some(mask) = attention_mask {
                for s in 0..seq_len.min(sample_out.nrows()) {
                    if mask[[b, s]] < 0.5 {
                        sample_out.row_mut(s).fill(0.0);
                    }
                }
            }

            output.slice_mut(ndarray::s![b, .., ..]).assign(&sample_out);
            ctxs.push(ctx);
        }

        (output, ctxs)
    }

    /// 批量反向传播（默认实现：逐样本循环）
    fn backward_batch(
        &mut self,
        ctxs: &[LayerContext],
        grads: &Array3<f32>,
        lr: f32,
        attention_mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        let batch_size = grads.shape()[0];
        let seq_len = grads.shape()[1];
        let hidden_dim = grads.shape()[2];

        assert_eq!(
            ctxs.len(),
            batch_size,
            "backward_batch: ctxs.len() must equal batch_size"
        );

        let mut grad_input = Array3::zeros((batch_size, seq_len, hidden_dim));
        for b in 0..batch_size {
            let mut sample_grad = grads.slice(ndarray::s![b, .., ..]).to_owned();

            if let Some(mask) = attention_mask {
                for s in 0..seq_len {
                    if mask[[b, s]] < 0.5 {
                        sample_grad.row_mut(s).fill(0.0);
                    }
                }
            }

            let sample_grad_input = self.backward(&ctxs[b], &sample_grad, lr);
            grad_input
                .slice_mut(ndarray::s![b, .., ..])
                .assign(&sample_grad_input);
        }
        grad_input
    }

    fn parameters(&self) -> usize;

    fn set_training_mode(&mut self, _training: bool) {}
}

#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
    pub context_window: Vec<usize>,
    pub max_context_length: usize,
    pub training: bool,
    // 性能优化：可重用的采样缓冲区（public以便序列化）
    pub sampling_prob_buffer: Vec<f32>,
    pub sampling_idx_buffer: Vec<(f32, usize)>,
    pub beam_candidates_buffer: Vec<(Vec<usize>, f32)>,
}

/// 训练信号（loss/grad）计算错误。
///
/// 一旦触发，说明上游对齐或词表尺寸已损坏；继续执行 optimizer step 会导致 Adam 动量“漂移”，
/// 破坏可复现性。因此调用方应 **跳过本步参数更新** 并优先定位根因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingSignalError {
    /// `probs.shape()[0] != target.len()`（常见于 target 对齐错误或 logits/softmax shape 错误）。
    ShapeMismatch {
        probs_rows: usize,
        target_len: usize,
    },
    /// target id 超出 vocab_size（常见于词表/输出层维度不一致）。
    TargetOutOfRange {
        row_idx: usize,
        target_idx: usize,
        vocab_size: usize,
    },
}

/// 单样本训练步的中间结果。
///
/// 说明：
/// - `layer_ctxs`：逐层保存前向传播上下文，供 `backward_with_ctx()` 或梯度累积流程复用；
/// - `grads_output`：loss 对 logits 的梯度，且当前口径已经按有效 token 做过平均；
/// - `loss_mean`：按有效 token 平均后的交叉熵；
/// - `n_targets`：有效 target（非 PAD）数量。
pub(crate) struct PreparedTrainingStep {
    pub(crate) layer_ctxs: Vec<LayerContext>,
    pub(crate) grads_output: Array2<f32>,
    pub(crate) loss_mean: f32,
    pub(crate) n_targets: usize,
}

#[derive(Default)]
pub(crate) struct EpochAccumulator {
    pub(crate) total_nll: f32,
    pub(crate) total_tokens: usize,
    pub(crate) total_grad_norm: f32,
    pub(crate) sample_count: usize,
}

impl EpochAccumulator {
    pub(crate) fn record_step(&mut self, step: &PreparedTrainingStep, grad_norm: f32) {
        self.total_nll += step.loss_mean * (step.n_targets as f32);
        self.total_tokens += step.n_targets;
        self.total_grad_norm += grad_norm;
        self.sample_count += 1;
    }

    pub(crate) fn has_valid_samples(&self) -> bool {
        self.sample_count > 0 && self.total_tokens > 0
    }

    pub(crate) fn avg_loss(&self) -> Option<f32> {
        self.has_valid_samples()
            .then_some(self.total_nll / self.total_tokens as f32)
    }

    pub(crate) fn avg_grad_norm(&self) -> Option<f32> {
        (self.sample_count > 0).then_some(self.total_grad_norm / self.sample_count as f32)
    }
}

/// 早停机制
///
/// 监控训练loss，如果长时间不改善则自动停止训练
pub struct EarlyStopping {
    /// 容忍多少个epoch loss不改善
    patience: usize,

    /// 当前最佳loss
    best_loss: f32,

    /// 已经多少个epoch没有改善
    counter: usize,

    /// 最小改善幅度（小于这个值不算改善）
    min_delta: f32,

    /// 最佳模型所在的epoch
    best_epoch: usize,
}

impl EarlyStopping {
    /// 创建早停监控器
    ///
    /// # 参数
    /// - `patience`: 容忍epoch数（推荐30-50）
    /// - `min_delta`: 最小改善幅度（推荐0.001）
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            best_loss: f32::INFINITY,
            counter: 0,
            min_delta,
            best_epoch: 0,
        }
    }

    /// 检查是否应该停止训练
    ///
    /// # 返回值
    /// - `true`: 应该停止训练
    /// - `false`: 继续训练
    pub fn should_stop(&mut self, current_loss: f32, current_epoch: usize) -> bool {
        // 如果loss有明显改善
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.best_epoch = current_epoch;
            self.counter = 0;
            false
        } else {
            // loss没有改善
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// 获取最佳loss和对应的epoch
    pub fn best_state(&self) -> (f32, usize) {
        (self.best_loss, self.best_epoch)
    }
}

impl Default for LLM {
    fn default() -> Self {
        let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let transformer_block_4 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::default_words().len());
        let vocab_size = Vocab::default_words().len();
        Self {
            vocab: Vocab::default(),
            network: vec![
                Box::new(Embeddings::default()),
                Box::new(transformer_block_1),
                Box::new(transformer_block_2),
                Box::new(transformer_block_3),
                Box::new(transformer_block_4),
                Box::new(output_projection),
            ],
            context_window: Vec::new(),
            max_context_length: MAX_SEQ_LEN,
            training: true,
            sampling_prob_buffer: Vec::with_capacity(vocab_size),
            sampling_idx_buffer: Vec::with_capacity(vocab_size),
            beam_candidates_buffer: Vec::with_capacity(50),
        }
    }
}

impl LLM {
    pub(crate) fn validate_network_topology(network: &[Box<dyn Layer>]) -> Result<(), String> {
        if network.len() < 2 {
            return Err(
                "当前教学主路径要求网络至少包含 Embeddings 和 OutputProjection 两层".to_string(),
            );
        }

        if !network
            .first()
            .and_then(|layer| layer.as_any().downcast_ref::<Embeddings>())
            .is_some()
        {
            return Err("第 0 层必须是 Embeddings".to_string());
        }

        if !network
            .last()
            .and_then(|layer| layer.as_any().downcast_ref::<OutputProjection>())
            .is_some()
        {
            return Err("最后一层必须是 OutputProjection".to_string());
        }

        for (idx, layer) in network
            .iter()
            .enumerate()
            .skip(1)
            .take(network.len().saturating_sub(2))
        {
            if !layer.as_any().is::<TransformerBlock>() {
                return Err(format!(
                    "当前教学主路径仅支持 Embeddings -> TransformerBlock* -> OutputProjection；第 {} 层实际为 {}",
                    idx,
                    layer.layer_type()
                ));
            }
        }

        Ok(())
    }

    pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        Self::validate_network_topology(&network)
            .unwrap_or_else(|error| panic!("LLM::new 收到不受支持的网络拓扑: {}", error));

        let vocab_size = vocab.words.len();
        Self {
            vocab,
            network,
            context_window: Vec::new(),
            max_context_length: MAX_SEQ_LEN,
            training: true,
            sampling_prob_buffer: Vec::with_capacity(vocab_size),
            sampling_idx_buffer: Vec::with_capacity(vocab_size),
            beam_candidates_buffer: Vec::with_capacity(50),
        }
    }
}

impl LLM {
    fn for_each_transformer_block_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut TransformerBlock),
    {
        for layer in &mut self.network {
            if let Some(block) = layer.as_any_mut().downcast_mut::<TransformerBlock>() {
                f(block);
            }
        }
    }

    fn for_each_transformer_block<F>(&self, mut f: F)
    where
        F: FnMut(&TransformerBlock),
    {
        for layer in &self.network {
            if let Some(block) = layer.as_any().downcast_ref::<TransformerBlock>() {
                f(block);
            }
        }
    }

    fn is_kv_cache_enabled(&self) -> bool {
        for layer in &self.network {
            if let Some(block) = layer.as_any().downcast_ref::<TransformerBlock>() {
                return block.attention.use_kv_cache;
            }
        }
        false
    }

    fn capture_kv_cache(&self) -> Vec<Option<(Array2<f32>, Array2<f32>)>> {
        let mut caches = Vec::new();
        self.for_each_transformer_block(|block| {
            caches.push(
                block
                    .attention
                    .kv_cache
                    .as_ref()
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
        });
        caches
    }

    fn restore_kv_cache(&mut self, caches: &[Option<(Array2<f32>, Array2<f32>)>]) {
        let mut idx = 0usize;
        self.for_each_transformer_block_mut(|block| {
            if let Some(cache) = caches.get(idx) {
                block.attention.kv_cache = cache.as_ref().map(|(k, v)| (k.clone(), v.clone()));
            } else {
                block.attention.kv_cache = None;
            }
            idx += 1;
        });
    }

    fn inference_step(&mut self, token_id: usize, position: usize) -> Array2<f32> {
        let embedding_output = {
            let embeddings = self
                .network
                .first_mut()
                .expect("网络至少包含嵌入层")
                .as_any_mut()
                .downcast_mut::<Embeddings>()
                .expect("首层必须是 Embeddings");
            embeddings.embed_tokens_with_offset(&[token_id], position)
        };

        let mut hidden = embedding_output;

        for layer in self.network.iter_mut().skip(1) {
            if let Some(block) = layer.as_any_mut().downcast_mut::<TransformerBlock>() {
                hidden = block.forward_inference(&hidden);
            } else {
                let (out, _ctx) = layer.forward(&hidden);
                hidden = out;
            }
        }

        hidden
    }

    fn select_top_k_from_row(&mut self, row: ArrayView1<'_, f32>, k: usize) -> Vec<(usize, f32)> {
        self.sampling_idx_buffer.clear();
        self.sampling_idx_buffer
            .extend(row.iter().enumerate().map(|(idx, &prob)| (prob, idx)));

        let top_k = k.min(self.sampling_idx_buffer.len());
        if top_k == 0 {
            return Vec::new();
        }

        let nth = top_k - 1;
        self.sampling_idx_buffer
            .select_nth_unstable_by(nth, |a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let top_slice = &mut self.sampling_idx_buffer[..top_k];
        top_slice.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        top_slice.iter().map(|&(prob, idx)| (idx, prob)).collect()
    }

    fn argmax_index(values: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_value = f32::NEG_INFINITY;

        for (idx, &value) in values.iter().enumerate() {
            if value > best_value {
                best_idx = idx;
                best_value = value;
            }
        }

        best_idx
    }

    fn sample_token_from_probs(
        &mut self,
        probs: &Array2<f32>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> usize {
        let last_row = probs.row(probs.nrows().saturating_sub(1));
        if temperature <= GREEDY_TEMPERATURE_EPS {
            let greedy: Vec<f32> = last_row.iter().copied().collect();
            return Self::argmax_index(&greedy);
        }

        let adjusted = Self::apply_temperature(probs, temperature);
        let candidates = if top_k > 0 {
            self.top_k_sampling(&adjusted, top_k)
        } else {
            self.top_p_sampling(&adjusted, top_p)
        };

        candidates
            .into_iter()
            .next()
            .unwrap_or_else(|| Self::argmax_index(&last_row.iter().copied().collect::<Vec<_>>()))
    }

    /// 当前增量推理的最低要求：首层必须是 Embeddings。
    ///
    /// 说明：
    /// - 当前默认网络满足该条件，因此正常会走增量 / KV cache 主路径；
    /// - 保留该判断，是为了让教学实验中的“非常规网络拼装”仍能退回完整前向路径。
    fn supports_incremental(&self) -> bool {
        self.network
            .first()
            .and_then(|layer| layer.as_any().downcast_ref::<Embeddings>())
            .is_some()
    }

    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|layer| layer.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        self.network
            .iter()
            .map(|layer| layer.parameters())
            .sum::<usize>()
    }

    pub fn set_training_mode(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.network {
            layer.set_training_mode(training);
        }
    }

    /// 最基础的教学训练入口。
    ///
    /// 说明：
    /// - 保留最少参数，便于讲解“tokenize -> forward -> loss -> backward”主线；
    /// - 学习率使用简单指数衰减；
    /// - 每个样本独立更新一次参数，不包含早停、checkpoint 或梯度累积。
    pub fn train(&mut self, data: Vec<&str>, epochs: usize, initial_lr: f32) {
        self.set_training_mode(true);

        let pad_token_id = self.vocab.pad_token_id();

        let tokenized_data = data
            .iter()
            // 教学要点：训练序列应包含 BOS/EOS，否则模型难以学会“什么时候结束”。
            .map(|input| Self::tokenize_training_with_vocab(&self.vocab, input))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let decay_rate: f32 = 0.95;
            let decay_steps = 10.0;
            let current_lr = initial_lr * decay_rate.powf(epoch as f32 / decay_steps);

            // 训练指标口径：统一使用 token-weighted mean，避免短序列被隐式加权。
            let mut epoch_accumulator = EpochAccumulator::default();
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                // 训练样本按“前 N-1 个 token 预测后 N-1 个 token”切分。
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];
                let _ = self.run_standard_training_step(
                    input_ids,
                    target_ids,
                    pad_token_id,
                    current_lr,
                    &mut epoch_accumulator,
                );
            }

            println!(
                "Epoch {}: Loss = {:.4}, LR = {:.6}",
                epoch,
                epoch_accumulator.avg_loss().unwrap_or(0.0),
                current_lr
            );
        }

        self.set_training_mode(false);
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // 🚀 阶段1训练优化 - 性能优化方法
    // ═════════════════════════════════════════════════════════════════════════════

    /// 余弦退火学习率调度（带重启）
    ///
    /// # 参数
    /// - `initial_lr`: 初始学习率（如 0.001）
    /// - `epoch`: 当前epoch
    /// - `total_epochs`: 总epoch数
    /// - `num_restarts`: 重启次数（如2表示训练分为3个周期）
    ///
    /// # 示例
    /// ```rust
    /// use llm::LLM;
    /// // 500 epochs, 2次重启，每个周期约166 epochs
    /// let lr = LLM::cosine_annealing_lr(0.001, 100, 500, 2);
    /// ```
    pub fn cosine_annealing_lr(
        initial_lr: f32,
        epoch: usize,
        total_epochs: usize,
        num_restarts: usize,
    ) -> f32 {
        if total_epochs == 0 {
            return initial_lr;
        }

        // 计算每个周期的长度
        //
        // 注意：当 total_epochs < (num_restarts+1) 时，整数除法会得到 0，进而导致：
        // - epoch % cycle_length 取模 0 panic
        // - progress 计算除以 0
        // 这里用 max(1) 做鲁棒性保护：宁可退化为“无退火”（cycle_length=1），也不要训练直接崩溃。
        // 额外安全性：
        // - `num_restarts + 1` 在极端输入下可能发生 usize 溢出并回绕为 0（release 模式），导致除以 0 panic；
        // - 这里使用 saturating_add 保证分母至少为 1。
        let denom = num_restarts.saturating_add(1).max(1);
        let cycle_length = (total_epochs / denom).max(1);

        // 当前在周期内的位置
        let cycle_epoch = epoch % cycle_length;

        // 周期内的进度 [0, 1]
        let progress = cycle_epoch as f32 / cycle_length as f32;

        // 最小学习率为初始值的1%
        let min_lr = initial_lr * 0.01;

        // 余弦退火公式
        min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    /// 余弦退火学习率调度（带热身）
    ///
    /// # 核心思想
    /// - **热身阶段 (Warmup)**：学习率从 0 线性升到 initial_lr，避免训练初期震荡
    /// - **退火阶段 (Cosine)**：热身结束后进入余弦退火
    ///
    /// # 参数
    /// - `initial_lr`: 初始学习率上限
    /// - `epoch`: 当前 epoch
    /// - `total_epochs`: 总 epoch 数
    /// - `num_restarts`: 余弦退火的重启次数
    /// - `warmup_epochs`: 热身轮数（建议占总训练 3%-10%）
    pub fn cosine_with_warmup_lr(
        initial_lr: f32,
        epoch: usize,
        total_epochs: usize,
        num_restarts: usize,
        warmup_epochs: usize,
    ) -> f32 {
        if total_epochs == 0 {
            return initial_lr;
        }

        let warmup_epochs = warmup_epochs.min(total_epochs);
        if warmup_epochs == 0 {
            return Self::cosine_annealing_lr(initial_lr, epoch, total_epochs, num_restarts);
        }

        if epoch < warmup_epochs {
            return initial_lr * (epoch + 1) as f32 / warmup_epochs as f32;
        }

        let adjusted_epoch = epoch - warmup_epochs;
        let adjusted_total = total_epochs.saturating_sub(warmup_epochs).max(1);
        Self::cosine_annealing_lr(initial_lr, adjusted_epoch, adjusted_total, num_restarts)
    }

    /// 计算推荐的 warmup 轮数（默认 5%）
    pub(crate) fn recommend_warmup_epochs(total_epochs: usize) -> usize {
        if total_epochs == 0 {
            return 0;
        }
        let warmup = ((total_epochs as f32) * 0.05).ceil() as usize;
        warmup.clamp(1, total_epochs)
    }

    /// 计算梯度L2范数
    pub fn compute_grad_norm(grads: &Array2<f32>) -> f32 {
        grads.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// 将“token-mean loss”的 logits 梯度还原为“sum NLL”的梯度。
    ///
    /// 训练语义：`compute_gradients_step()` 已经按 `n_targets` 做了平均；
    /// 梯度累积阶段必须先乘回 `n_targets`，否则多个不同长度序列会退化为 sequence-weighted。
    pub(crate) fn rescale_logits_grads_for_accumulation(grads: &mut Array2<f32>, n_targets: usize) {
        debug_assert!(n_targets > 0, "n_targets must be > 0 for accumulation");
        grads.mapv_inplace(|x| x * (n_targets as f32));
    }

    /// 计算当前累积窗口内的 token-weighted 平均缩放系数。
    pub(crate) fn token_weighted_accum_scale(accum_tokens: usize) -> f32 {
        assert!(
            accum_tokens > 0,
            "token_weighted_accum_scale requires accum_tokens > 0"
        );
        1.0 / accum_tokens as f32
    }

    /// 当前 micro-batch 在整个累积窗口中的 token 权重。
    #[cfg(test)]
    pub(crate) fn token_weighted_micro_batch_weight(n_targets: usize, accum_tokens: usize) -> f32 {
        (n_targets as f32) * Self::token_weighted_accum_scale(accum_tokens)
    }

    // =====================================================================
    // 梯度累积：跨层统一调度（网络级别）
    // =====================================================================
    //
    // 设计说明（教育项目优先）：
    // - 本项目的 `Layer::backward()` 设计为“反向传播 + 立刻更新参数（Adam step）”。
    // - 为了实现正确的梯度累积，我们不能简单把多个样本的 logits 梯度攒起来再 backward 一次，
    //   因为**旧版**各层 backward 会从 `self.cached_*` 读取 forward 的中间量；如果在同一层实例上
    //   对多个样本 forward 而不保存每个样本的中间量，就会发生“缓存覆盖”，导致数学错误。
    // - 本轮重构已将中间量改为 **ctx 驱动**（forward 返回 ctx，反传显式消费 ctx），但正确的
    //   梯度累积仍然需要“每个 micro-batch 对应一份 ctx”。
    // - 正确做法是：每个 micro-batch 立即 backward（保证 ctx 对应），但把 *参数梯度* 累积，
    //   最终对平均梯度做一次参数更新。
    //
    // 由于网络里使用 `Box<dyn Layer>`（动态分发），我们在这里对已知层类型做 downcast，
    // 并调用其 ctx 驱动的梯度累积实现与 `step_accumulated()`。
    //
    // 当前网络结构（src/main.rs）固定为：
    // Embeddings -> TransformerBlock* -> OutputProjection
    fn zero_grad_accum(&mut self) {
        for layer in &mut self.network {
            if let Some(emb) = layer.as_any_mut().downcast_mut::<Embeddings>() {
                emb.zero_grad_accum();
            } else if let Some(tb) = layer.as_any_mut().downcast_mut::<TransformerBlock>() {
                tb.zero_grad_accum();
            } else if let Some(op) = layer.as_any_mut().downcast_mut::<OutputProjection>() {
                op.zero_grad_accum();
            }
        }
    }

    /// 梯度累积（网络级别调度，ctx 驱动）。
    ///
    /// 说明：
    /// - 梯度累积阶段不能依赖层内隐式缓存，否则 batch/并发下容易发生样本错配；
    /// - 这里统一要求：forward 收集每层 ctx，反传（累积）时按层把 ctx 逐一归还。
    fn backward_accumulate_with_ctx(
        &mut self,
        layer_ctxs: &[LayerContext],
        grads_output: &Array2<f32>,
    ) -> Array2<f32> {
        // 这是结构性错误：ctx 与层必须一一对应，否则 zip 会截断并导致“部分层不反传/层与 ctx 错配”，
        // 训练会静默算错。因此这里直接 fail-fast。
        if layer_ctxs.len() != self.network.len() {
            panic!(
                "LLM.backward_accumulate_with_ctx: layer_ctxs.len()={} 与 network.len()={} 不一致，无法保证梯度正确性",
                layer_ctxs.len(),
                self.network.len()
            );
        }

        let mut grads = grads_output.clone();
        for (layer, ctx) in self.network.iter_mut().rev().zip(layer_ctxs.iter().rev()) {
            grads = if let Some(op) = layer.as_any_mut().downcast_mut::<OutputProjection>() {
                op.backward_accumulate_with_ctx(ctx, &grads)
            } else if let Some(tb) = layer.as_any_mut().downcast_mut::<TransformerBlock>() {
                tb.backward_accumulate_with_ctx(ctx, &grads)
            } else if let Some(emb) = layer.as_any_mut().downcast_mut::<Embeddings>() {
                emb.backward_accumulate_with_ctx(ctx, &grads)
            } else {
                // 理论上不会发生（网络结构在 src/main.rs 中固定）。
                //
                // 旧版这里会 fallback 到 `backward(lr=0.0)`，但 Adam 即使 lr=0 也会推进内部状态
                //（timestep/m/v），导致训练“看似没更新参数但状态漂移”，并破坏梯度累积的等价性。
                //
                // 因此这里选择 fail-fast：发现不支持累积的层就直接终止，避免静默错误。
                panic!(
                    "Layer {} 未实现 ctx 驱动的梯度累积接口，无法启用 accumulation_steps。",
                    layer.layer_type()
                );
            };
        }
        grads
    }

    fn step_accumulated(&mut self, lr: f32, scale: f32) {
        for layer in &mut self.network {
            if let Some(emb) = layer.as_any_mut().downcast_mut::<Embeddings>() {
                emb.step_accumulated(lr, scale);
            } else if let Some(tb) = layer.as_any_mut().downcast_mut::<TransformerBlock>() {
                tb.step_accumulated(lr, scale);
            } else if let Some(op) = layer.as_any_mut().downcast_mut::<OutputProjection>() {
                op.step_accumulated(lr, scale);
            }
        }
    }

    /// 单样本训练步：执行前向传播，收集 ctx，并计算 loss 与 `dL/dlogits`。
    ///
    /// 教学说明：
    /// - 这是当前训练主线的核心拼装点，负责把“一个样本”整理成可反传的数据包；
    /// - 它只做 `forward + ctx 收集 + loss/梯度构造`，不直接执行参数更新；
    /// - 调用方随后还需要决定是立刻 `backward_with_ctx()`，还是先走梯度累积路径；
    /// - 返回 `None` 表示当前样本没有有效训练信号（例如输入为空、target 全 PAD，或训练信号非法）。
    pub(crate) fn prepare_training_step(
        &mut self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_token_id: usize,
    ) -> Option<PreparedTrainingStep> {
        let n_targets = target_ids.iter().filter(|&&t| t != pad_token_id).count();
        if input_ids.is_empty() || n_targets == 0 {
            return None;
        }

        let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
        input
            .row_mut(0)
            .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

        let mut layer_ctxs: Vec<LayerContext> = Vec::with_capacity(self.network.len());
        for layer in &mut self.network {
            let (out, ctx) = layer.forward(&input);
            layer_ctxs.push(ctx);
            input = out;
        }

        let log_probs = log_softmax(&input);
        let probs = log_probs.mapv(|x| x.exp());
        let grads_output = match Self::compute_gradients_step(&probs, target_ids, pad_token_id) {
            Ok(Some(grads)) => grads,
            Ok(None) => return None,
            Err(err) => {
                log::error!("训练信号错误({err:?})，已跳过 optimizer step");
                return None;
            }
        };

        let loss_mean = Self::cross_entropy_from_log_probs(&log_probs, target_ids, pad_token_id);

        Some(PreparedTrainingStep {
            layer_ctxs,
            grads_output,
            loss_mean,
            n_targets,
        })
    }

    pub(crate) fn run_standard_training_step(
        &mut self,
        input_ids: &[usize],
        target_ids: &[usize],
        pad_token_id: usize,
        current_lr: f32,
        epoch_accumulator: &mut EpochAccumulator,
    ) -> bool {
        let Some(mut step) = self.prepare_training_step(input_ids, target_ids, pad_token_id) else {
            return false;
        };

        epoch_accumulator.record_step(&step, Self::compute_grad_norm(&step.grads_output));
        Self::clip_gradients(&mut step.grads_output, 1.0);
        self.backward_with_ctx(&step.layer_ctxs, &step.grads_output, current_lr);
        true
    }

    /// 逐层取回 ctx 并执行标准反向传播。
    ///
    /// 与旧版 `cached_*` 隐式缓存不同，这里显式消费每层 forward 返回的 `ctx`：
    /// - 这样更容易看清数据流；
    /// - 也避免多个样本/多次 forward 之间发生中间量覆盖；
    /// - 该路径会在各层 `backward()` 中立即应用参数更新，对应“标准单步训练”。
    pub(crate) fn backward_with_ctx(
        &mut self,
        layer_ctxs: &[LayerContext],
        grads_output: &Array2<f32>,
        lr: f32,
    ) -> Array2<f32> {
        assert_eq!(
            layer_ctxs.len(),
            self.network.len(),
            "LLM.backward_with_ctx: layer_ctxs.len() must equal network.len()"
        );

        let mut grads = grads_output.clone();
        for (layer, ctx) in self.network.iter_mut().rev().zip(layer_ctxs.iter().rev()) {
            grads = layer.backward(ctx, &grads, lr);
        }
        grads
    }

    /// 监控型训练方法（集成预处理、学习率调度、早停与梯度累积）
    ///
    /// # 特性
    /// - ✅ 单线程 tokenization 与预处理计时
    /// - ✅ 余弦退火学习率调度
    /// - ✅ 早停机制
    /// - ✅ 增强训练监控（困惑度、梯度范数、训练速度）
    /// - ✅ 正确的梯度累积（accumulation_steps > 1 时：micro-batch 逐次 backward + 参数梯度累加，最终一次更新）
    ///
    /// # 参数
    /// - `data`: 训练数据
    /// - `max_epochs`: 最大 epoch 数
    /// - `initial_lr`: 初始学习率
    /// - `patience`: 早停容忍 epoch 数
    /// - `accumulation_steps`: 梯度累积步数（推荐 4-8）
    ///
    /// # 返回值
    /// 实际训练的 epoch 数
    pub fn train_monitored(
        &mut self,
        data: Vec<&str>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
        accumulation_steps: usize,
    ) -> usize {
        self.set_training_mode(true);

        let pad_token_id = self.vocab.pad_token_id();

        let mut perf_monitor = PerformanceMonitor::new();
        let effective_accum_steps = accumulation_steps.max(1);

        println!("📝 正在预处理训练数据...");
        let preprocess_start = std::time::Instant::now();

        // 说明：
        // - 早期版本的注释中提到“Rayon 并行 tokenization”，但当前仓库并未引入 rayon 依赖；
        // - 为避免教学误导，我们在这里采用明确的单线程实现，并保留计时监控。
        perf_monitor.start("tokenization_single_thread");
        let tokenized_data: Vec<Vec<usize>> = data
            .iter()
            // 训练序列必须包含 BOS/EOS（否则模型学不到“何时结束”）
            .map(|input| Self::tokenize_training_with_vocab(&self.vocab, input))
            .collect();
        perf_monitor.stop("tokenization_single_thread");

        println!(
            "✅ 数据预处理完成，共 {} 个序列（耗时 {:.2}s）",
            tokenized_data.len(),
            preprocess_start.elapsed().as_secs_f32()
        );

        println!(
            "🧮 梯度累积: {} (accumulation_steps={})",
            if effective_accum_steps > 1 {
                "启用：micro-batch 累积后一次更新"
            } else {
                "关闭：每个样本一次更新"
            },
            effective_accum_steps
        );

        let mut early_stopping = EarlyStopping::new(patience, 0.01);
        let training_start_time = std::time::Instant::now();

        for epoch in 0..max_epochs {
            let epoch_start = std::time::Instant::now();

            // 🔥 余弦退火 + Warmup（禁用重启以提升稳定性）
            let warmup_epochs = Self::recommend_warmup_epochs(max_epochs);
            let current_lr =
                Self::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);

            // 训练指标口径（重要）：
            // - 旧实现：先对每个序列做 token-mean（loss / n_targets），再对序列做 mean（/ sample_count）
            //   → 会把短序列“隐式加权”得更重。
            // - 新实现：对所有有效 token 的 NLL 求和后再除以总 token 数
            //   → 与常见 LM/困惑度定义一致（token-weighted）。
            let mut epoch_accumulator = EpochAccumulator::default();
            let mut accum_counter = 0usize; // micro-batch 计数（用于触发 step）
            let mut accum_tokens = 0usize; // micro-batch 内有效 token 数（用于 token-weighted scale）
            // 每个 epoch 重新开始累积
            self.zero_grad_accum();

            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                // 前向传播
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];
                let Some(mut step) =
                    self.prepare_training_step(input_ids, target_ids, pad_token_id)
                else {
                    continue;
                };

                epoch_accumulator.record_step(&step, Self::compute_grad_norm(&step.grads_output));
                Self::clip_gradients(&mut step.grads_output, 1.0);

                // ==========================
                // 正确的梯度累积实现
                // ==========================
                //
                // 关键点：
                // - 必须对每个 micro-batch 保留并及时消费其对应的 ctx（常见做法：forward 后立刻累积反传）；
                // - 但不立刻更新参数，而是把参数梯度累加到各层的累积 buffer；
                // - 当累积步数到达阈值时，再统一对“平均梯度”执行一次参数更新（scale=1/steps）。
                //
                // 这里额外做一个关键修复（token-weighted）：
                // - `compute_gradients_step()` 输出的是 “token-mean loss” 对 logits 的梯度（已除以 n_targets）；
                // - 若直接对 micro-batch 求平均，会变成 sequence-weighted（每条序列权重相同）；
                // - 我们把 logits 梯度乘回 n_targets，使其对应 “sum NLL” 的梯度，
                //   然后在 step 时用 `scale = 1/accum_tokens` 做统一平均，得到 token-weighted 梯度。
                Self::rescale_logits_grads_for_accumulation(&mut step.grads_output, step.n_targets);
                self.backward_accumulate_with_ctx(&step.layer_ctxs, &step.grads_output);
                accum_counter += 1;
                accum_tokens += step.n_targets;
                if accum_counter >= effective_accum_steps {
                    self.step_accumulated(
                        current_lr,
                        Self::token_weighted_accum_scale(accum_tokens),
                    );
                    accum_counter = 0;
                    accum_tokens = 0;
                }
            }

            // 处理最后一个“不满 accumulation_steps”的尾批次
            if accum_counter > 0 {
                self.step_accumulated(current_lr, Self::token_weighted_accum_scale(accum_tokens));
            }

            if !epoch_accumulator.has_valid_samples() {
                log::error!(
                    "train_monitored: 没有有效训练样本（所有序列长度 < 2 或全部被跳过），无法继续训练。epoch={}",
                    epoch
                );
                self.set_training_mode(false);
                perf_monitor.print_report();
                return epoch;
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            let avg_loss = epoch_accumulator.avg_loss().unwrap_or(0.0);
            let avg_grad_norm = epoch_accumulator.avg_grad_norm().unwrap_or(0.0);
            let perplexity = avg_loss.exp();
            let samples_per_sec = if epoch_time > 0.0 {
                epoch_accumulator.sample_count as f32 / epoch_time
            } else {
                0.0
            };

            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                let progress = (epoch + 1) as f32 / max_epochs as f32 * 100.0;
                let elapsed = training_start_time.elapsed().as_secs();
                let eta = if epoch + 1 > 0 {
                    (elapsed as f32 / (epoch + 1) as f32 * (max_epochs - epoch - 1) as f32) as u64
                } else {
                    0
                };

                println!(
                    "[{:3}/{}] {:6.1}% | Loss: {:.4} | PPL: {:6.2} | LR: {:.6} | Grad: {:6.4} | Speed: {:5.1} samples/s | ETA: {}s",
                    epoch + 1,
                    max_epochs,
                    progress,
                    avg_loss,
                    perplexity,
                    current_lr,
                    avg_grad_norm,
                    samples_per_sec,
                    eta
                );
            }

            if early_stopping.should_stop(avg_loss, epoch) {
                let (best_loss, best_epoch) = early_stopping.best_state();
                println!("\n🛑 早停触发:");
                println!("   • 最佳epoch: {}", best_epoch + 1);
                println!("   • 最佳loss: {:.4}", best_loss);
                println!("   • 停止epoch: {}", epoch + 1);
                println!("   • 节省时间: {} epochs", max_epochs - epoch);

                self.set_training_mode(false);
                perf_monitor.print_report();
                return epoch + 1;
            }
        }

        self.set_training_mode(false);
        perf_monitor.print_report();
        max_epochs
    }

    /// 分桶批次训练方法（支持动态掩码）
    ///
    /// # 语义说明
    /// - 该方法使用 batch 组织样本、padding mask 与 bucketing 来减少填充开销；
    /// - 但参数更新仍按“批内逐样本顺序执行”，不是严格的 batch 梯度平均后统一更新。
    ///
    /// # 特性
    /// - ✅ 批次组织：按小批次分组样本并减少无效填充
    /// - ✅ 动态填充：每个批次填充到该批次的最大长度
    /// - ✅ 注意力掩码：确保 PAD 不参与梯度计算
    /// - ✅ 前向裁剪：每个样本只前向真实 token，避免 PAD 干扰注意力
    /// - ✅ 数据分桶：减少填充开销
    /// - ✅ 所有监控和优化特性（余弦退火、早停等）
    ///
    /// # 参数
    /// - `data`: 训练数据
    /// - `max_epochs`: 最大 epoch 数
    /// - `initial_lr`: 初始学习率
    /// - `patience`: 早停容忍 epoch 数
    /// - `batch_size`: 批次大小（推荐 2-8）
    ///
    /// # 返回值
    /// 实际训练的 epoch 数
    pub fn train_bucketed_sequential(
        &mut self,
        data: Vec<&str>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
        batch_size: usize,
    ) -> usize {
        use crate::batch_loader::{BatchLoader, create_training_batches};

        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        let perf_monitor = PerformanceMonitor::new();

        println!("📝 正在预处理训练数据...");
        let preprocess_start = std::time::Instant::now();

        // 先把所有训练文本转成带 BOS/EOS 的 token 序列。
        let tokenized_data: Vec<Vec<usize>> = data
            .iter()
            // 训练序列必须包含 BOS/EOS（否则模型学不到“何时结束”）
            .map(|input| Self::tokenize_training_with_vocab(&self.vocab, input))
            .collect();

        println!(
            "✅ 数据预处理完成，共 {} 个序列（耗时 {:.2}s）",
            tokenized_data.len(),
            preprocess_start.elapsed().as_secs_f32()
        );

        // 创建批量加载器
        let batch_loader = BatchLoader::new_with_pad_token_id(batch_size, true, 16, pad_token_id);

        let mut early_stopping = EarlyStopping::new(patience, 0.01);
        let training_start_time = std::time::Instant::now();

        for epoch in 0..max_epochs {
            let epoch_start = std::time::Instant::now();

            // 余弦退火 + Warmup
            let warmup_epochs = Self::recommend_warmup_epochs(max_epochs);
            let current_lr =
                Self::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);

            // 训练指标口径：统一使用 token-weighted mean，避免短序列被隐式加权。
            let mut epoch_accumulator = EpochAccumulator::default();

            // 创建训练批次
            let training_batches = create_training_batches(&batch_loader, &tokenized_data);

            for (input_batch, targets) in training_batches {
                // 跳过空批次
                if input_batch.batch_size == 0 {
                    continue;
                }

                // 重要（历史背景 + 设计约束）：
                // - **旧版**各层 backward 会从 `self.cached_*` 读取 forward 的中间量，因此如果你在
                //   “同一模型实例/同一层实例”上先对多个样本 forward、再统一 backward，就会发生
                //   “缓存覆盖”，从而出现梯度与激活不匹配的严重错误。
                // - 本轮重构已将中间量改为 **ctx 驱动**（forward 返回 ctx，反传显式消费 ctx），因此
                //   不再依赖 `cached_*` 字段；但如果想实现“先对整个 batch forward、再统一 backward”，
                //   仍然必须为 batch 内每个样本保存其对应的 `layer_ctxs`，否则中间量会丢失/错配。
                //
                // 为了保证正确性，这里采用最清晰的做法：
                // - 对 batch 内每个样本执行：forward -> loss -> backward（立刻更新参数）。
                //
                // 这在数学上等价于“批内样本的顺序 SGD”，不是严格的“batch 梯度平均后更新一次”。
                // 但它是正确的，并且符合教学项目“先正确、再优化”的原则。
                for b in 0..input_batch.batch_size {
                    let target_ids = targets.get(b).cloned().unwrap_or_default();
                    let target_len = target_ids.len();
                    if target_len == 0 {
                        continue;
                    }

                    // 只取真实 token（避免 PAD 参与前向计算）
                    let sample_tokens = input_batch.tokens.row(b);
                    let sample_ids: Vec<usize> =
                        sample_tokens.iter().take(target_len).copied().collect();

                    let Some(mut step) =
                        self.prepare_training_step(&sample_ids, &target_ids, pad_token_id)
                    else {
                        continue;
                    };

                    epoch_accumulator
                        .record_step(&step, Self::compute_grad_norm(&step.grads_output));
                    Self::clip_gradients(&mut step.grads_output, 1.0);
                    self.backward_with_ctx(&step.layer_ctxs, &step.grads_output, current_lr);
                }
            }

            if !epoch_accumulator.has_valid_samples() {
                log::error!(
                    "train_bucketed_sequential: 没有有效训练样本（所有序列长度 < 2 或全部被跳过），无法继续训练。epoch={}",
                    epoch
                );
                self.set_training_mode(false);
                perf_monitor.print_report();
                return epoch;
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            let avg_loss = epoch_accumulator.avg_loss().unwrap_or(0.0);
            let avg_grad_norm = epoch_accumulator.avg_grad_norm().unwrap_or(0.0);
            let perplexity = avg_loss.exp();
            let samples_per_sec = if epoch_time > 0.0 {
                epoch_accumulator.sample_count as f32 / epoch_time
            } else {
                0.0
            };

            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                let progress = (epoch + 1) as f32 / max_epochs as f32 * 100.0;
                let elapsed = training_start_time.elapsed().as_secs();
                let eta = if epoch + 1 > 0 {
                    (elapsed as f32 / (epoch + 1) as f32 * (max_epochs - epoch - 1) as f32) as u64
                } else {
                    0
                };

                println!(
                    "[{:3}/{}] {:6.1}% | Loss: {:.4} | PPL: {:6.2} | LR: {:.6} | Grad: {:6.4} | Speed: {:5.1} samples/s | ETA: {}s | Batch: {}",
                    epoch + 1,
                    max_epochs,
                    progress,
                    avg_loss,
                    perplexity,
                    current_lr,
                    avg_grad_norm,
                    samples_per_sec,
                    eta,
                    batch_size
                );
            }

            if early_stopping.should_stop(avg_loss, epoch) {
                let (best_loss, best_epoch) = early_stopping.best_state();
                println!("\n🛑 早停触发:");
                println!("   • 最佳epoch: {}", best_epoch + 1);
                println!("   • 最佳loss: {:.4}", best_loss);
                println!("   • 停止epoch: {}", epoch + 1);
                println!("   • 节省时间: {} epochs", max_epochs - epoch);

                self.set_training_mode(false);
                perf_monitor.print_report();
                return epoch + 1;
            }
        }

        self.set_training_mode(false);
        perf_monitor.print_report();
        max_epochs
    }

    /// 计算 3D 梯度张量的 L2 范数
    ///
    /// 教学/备用说明：
    /// - 当前训练主链路主要使用 1D/2D 的梯度（例如线性层权重与偏置）；
    /// - 这里保留 3D 版本，便于以后处理真正的 3D 激活或梯度张量时复用；
    /// - 为避免 `dead_code` 警告影响阅读，这里显式允许未使用。
    #[allow(dead_code)]
    fn compute_grad_norm_3d(grads: &Array3<f32>) -> f32 {
        grads.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// 3D 梯度裁剪
    ///
    /// 见 `compute_grad_norm_3d` 的教学/备用说明。
    #[allow(dead_code)]
    fn clip_gradients_3d(grads: &mut Array3<f32>, max_norm: f32) {
        let norm = Self::compute_grad_norm_3d(grads);
        if norm > max_norm {
            let scale = max_norm / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }

    /// Add tokens to the context window, maintaining the maximum length
    pub fn add_to_context(&mut self, tokens: &[usize]) {
        // Add new tokens to the context window
        self.context_window.extend_from_slice(tokens);

        // If context exceeds maximum length, remove oldest tokens
        if self.context_window.len() > self.max_context_length {
            let excess = self.context_window.len() - self.max_context_length;
            self.context_window.drain(0..excess);
        }
    }

    /// Clear the context window
    pub fn clear_context(&mut self) {
        self.context_window.clear();
    }

    /// 启用所有transformer层的KV缓存
    ///
    /// KV 缓存在逐 token 推理中通常能明显减少重复计算，但不能用于训练。
    /// 适用场景：交互式对话生成、逐token生成等
    pub fn enable_kv_cache(&mut self) {
        self.for_each_transformer_block_mut(|block| block.attention.enable_kv_cache());
    }

    /// 禁用所有transformer层的KV缓存
    pub fn disable_kv_cache(&mut self) {
        self.for_each_transformer_block_mut(|block| block.attention.disable_kv_cache());
    }

    /// 清空所有transformer层的KV缓存（保持启用状态）
    pub fn clear_kv_cache(&mut self) {
        self.for_each_transformer_block_mut(|block| block.attention.clear_kv_cache());
    }

    /// 统一设置所有 TransformerBlock 的 SelfAttention 是否冻结参数更新
    /// 用于在不修改网络结构的前提下，快速排查训练不稳定问题
    pub fn set_attention_freeze_updates(&mut self, freeze: bool) {
        self.for_each_transformer_block_mut(|block| block.attention.freeze_updates = freeze);
    }

    /// Get current context as token IDs
    #[allow(dead_code)]
    pub fn get_context(&self) -> &[usize] {
        &self.context_window
    }

    /// Set a fixed context
    #[allow(dead_code)]
    pub fn set_context(&mut self, tokens: Vec<usize>) {
        self.context_window = tokens;
        // Ensure context doesn't exceed maximum length
        if self.context_window.len() > self.max_context_length {
            let excess = self.context_window.len() - self.max_context_length;
            self.context_window.drain(0..excess);
        }
    }

    pub fn tokenize_with_vocab(vocab: &Vocab, text: &str) -> Vec<usize> {
        let has_chinese = text
            .chars()
            .any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);

        if has_chinese {
            return vocab.encode_sequence(text);
        }

        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            if word == "</s>" {
                if let Some(token_id) = vocab.encode(word) {
                    tokens.push(token_id);
                }
                continue;
            }

            let mut current_word = String::new();

            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current_word.is_empty() {
                        if let Some(token_id) = vocab.encode(&current_word) {
                            tokens.push(token_id);
                        }
                        current_word.clear();
                    }

                    if let Some(token_id) = vocab.encode(&c.to_string()) {
                        tokens.push(token_id);
                    }
                } else {
                    current_word.push(c);
                }
            }

            if !current_word.is_empty() {
                if let Some(token_id) = vocab.encode(&current_word) {
                    tokens.push(token_id);
                }
            }
        }

        tokens
    }

    /// 将文本编码为“训练用 token 序列”：自动注入 BOS/EOS，并做长度截断。
    ///
    /// # 为什么训练必须有 BOS/EOS？
    /// - 生成阶段通常用 `eos_token_id()` 作为停止条件；
    /// - 如果训练数据里从不出现 EOS，模型就学不到“何时结束”，只能靠 max_len 强行截断；
    /// - 对教学项目来说，这是非常典型、也非常容易被忽略的坑。
    ///
    /// # 具体策略（KISS 版本）
    /// 1. 先按 `tokenize_with_vocab()` 做基础分词/编码；
    /// 2. 序列非空时：开头插入 `<|bos|>`，末尾追加 `</s>`；
    /// 3. 若超过 `MAX_SEQ_LEN`：保留 `BOS + 最后 N 个内容 token + EOS`（更偏向保留“最近上下文”）。
    pub fn tokenize_training_with_vocab(vocab: &Vocab, text: &str) -> Vec<usize> {
        let mut tokens = Self::tokenize_with_vocab(vocab, text);
        if tokens.is_empty() {
            return tokens;
        }

        let bos = vocab.bos_token_id();
        let eos = vocab.eos_token_id();

        if tokens.first().copied() != Some(bos) {
            tokens.insert(0, bos);
        }
        if tokens.last().copied() != Some(eos) {
            tokens.push(eos);
        }

        if tokens.len() <= MAX_SEQ_LEN {
            return tokens;
        }

        // tokens: [BOS, ...content..., EOS]
        let keep_content = MAX_SEQ_LEN.saturating_sub(2);
        let content = &tokens[1..tokens.len().saturating_sub(1)];
        let start = content.len().saturating_sub(keep_content);

        let mut truncated = Vec::with_capacity(MAX_SEQ_LEN);
        truncated.push(bos);
        truncated.extend_from_slice(&content[start..]);
        truncated.push(eos);
        truncated
    }

    /// 将文本编码为“推理/提示词用 token 序列”：可选注入 BOS，但不自动追加 EOS。
    ///
    /// 说明：
    /// - 训练时我们强制注入 BOS/EOS；推理时如果也注入 BOS，模型行为更一致；
    /// - 但不注入 EOS：否则提示词会被模型理解为“已经结束”，影响生成。
    pub fn tokenize_prompt_with_vocab(vocab: &Vocab, text: &str) -> Vec<usize> {
        let mut tokens = Self::tokenize_with_vocab(vocab, text);
        if tokens.is_empty() {
            return tokens;
        }

        let bos = vocab.bos_token_id();
        if tokens.first().copied() != Some(bos) {
            tokens.insert(0, bos);
        }

        if tokens.len() > MAX_SEQ_LEN {
            // 推理场景：保留末尾上下文（更贴近对话系统）
            tokens = tokens[tokens.len() - MAX_SEQ_LEN..].to_vec();
        }
        tokens
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // 保持行为最小惊讶原则（KISS）：
        // - `tokenize()` 只负责“把文本编码成 token ids”，不隐式注入特殊 token；
        // - 训练场景请使用 `tokenize_training_with_vocab()`（显式注入 BOS/EOS）。
        Self::tokenize_with_vocab(&self.vocab, text)
    }

    fn apply_temperature(probs: &Array2<f32>, temperature: f32) -> Array2<f32> {
        if temperature <= 0.0 {
            return probs.clone();
        }

        // 直接对概率做 `powf(1 / temperature)` 在极小 temperature 下容易整体下溢为 0，
        // 最终让采样退化到“sum == 0 -> 随机 token”。
        //
        // 这里改为在 log 概率空间完成同等变换：
        //   p_i^(1/T) / Z  ==  softmax(log(p_i) / T)
        // 并通过“减去行内最大 logit”保持数值稳定。
        let mut adjusted = probs.clone();

        for mut row in adjusted.rows_mut() {
            let original_row: Vec<f32> = row.iter().copied().collect();
            let scaled_logs: Vec<f32> = original_row
                .iter()
                .map(|&value| value.max(SOFTMAX_EPSILON).ln() / temperature)
                .collect();

            let max_scaled_log = scaled_logs
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            if !max_scaled_log.is_finite() {
                continue;
            }

            let mut sum = 0.0_f32;
            for (value, scaled_log) in row.iter_mut().zip(scaled_logs.iter().copied()) {
                *value = (scaled_log - max_scaled_log).exp();
                sum += *value;
            }

            if sum > 0.0 && sum.is_finite() {
                for value in row.iter_mut() {
                    *value /= sum;
                }
            } else {
                let argmax = Self::argmax_index(&original_row);
                row.fill(0.0);
                row[argmax] = 1.0;
            }
        }

        adjusted
    }

    /// Top-k sampling: only consider the k most probable tokens
    /// 优化版本：复用内部缓冲区减少分配
    fn top_k_sampling(&mut self, probs: &Array2<f32>, k: usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(probs.nrows());

        for row in probs.rows() {
            let top_entries = self.select_top_k_from_row(row, k);

            self.sampling_prob_buffer.clear();
            self.sampling_prob_buffer
                .resize(self.vocab.words.len(), 0.0);

            let mut sum = 0.0;
            for (idx, prob) in &top_entries {
                self.sampling_prob_buffer[*idx] = *prob;
                sum += *prob;
            }

            if sum > 0.0 {
                for value in &mut self.sampling_prob_buffer {
                    *value /= sum;
                }
            }

            result.push(self.sample_from_probs(&self.sampling_prob_buffer));
        }

        result
    }

    /// Top-p (nucleus) sampling: consider the smallest set of tokens whose cumulative probability
    /// exceeds p
    /// 优化版本：使用部分排序避免全量排序
    fn top_p_sampling(&mut self, probs: &Array2<f32>, p: f32) -> Vec<usize> {
        let mut result = Vec::with_capacity(probs.nrows());
        let target_p = p.clamp(SOFTMAX_EPSILON, 1.0);

        for row in probs.rows() {
            let mut heap = BinaryHeap::new();
            let mut best_entry: Option<ProbEntry> = None;

            for (idx, &prob) in row.iter().enumerate() {
                if prob.is_nan() {
                    continue;
                }
                let entry = ProbEntry { prob, idx };
                if best_entry.as_ref().map_or(true, |best| entry > *best) {
                    best_entry = Some(entry.clone());
                }
                if prob > 0.0 {
                    heap.push(entry);
                }
            }

            self.sampling_prob_buffer.clear();
            self.sampling_prob_buffer
                .resize(self.vocab.words.len(), 0.0);

            let mut cumulative = 0.0;
            let mut selected_entries: Vec<ProbEntry> = Vec::new();

            while cumulative < target_p {
                match heap.pop() {
                    Some(entry) => {
                        cumulative += entry.prob;
                        selected_entries.push(entry);
                    }
                    None => break,
                }
            }

            if selected_entries.is_empty() {
                if let Some(entry) = best_entry {
                    selected_entries.push(entry);
                }
            }

            let mut sum = 0.0;
            for entry in &selected_entries {
                self.sampling_prob_buffer[entry.idx] = entry.prob;
                sum += entry.prob;
            }

            if sum > 0.0 {
                for value in &mut self.sampling_prob_buffer {
                    *value /= sum;
                }
            }

            result.push(self.sample_from_probs(&self.sampling_prob_buffer));
        }

        result
    }

    /// Sample from a probability distribution
    fn sample_from_probs(&self, probs: &[f32]) -> usize {
        let mut rng = rng();
        let sum: f32 = probs.iter().sum();

        if !(sum.is_finite() && sum > 0.0) {
            return Self::argmax_index(probs);
        }

        let mut normalized_probs = Vec::new();
        let mut cumsum = 0.0;

        for &prob in probs {
            cumsum += prob / sum;
            normalized_probs.push(cumsum);
        }

        let rand_val: f32 = rng.random();
        for (i, &cum_prob) in normalized_probs.iter().enumerate() {
            if rand_val <= cum_prob {
                return i;
            }
        }

        probs.len() - 1
    }

    fn generate_tokens_full(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Vec<usize> {
        if prompt_tokens.is_empty() || max_new_tokens == 0 {
            return Vec::new();
        }

        let mut perf_monitor = PerformanceMonitor::new();
        perf_monitor.start("inference_generation");
        let generation_start = Instant::now();

        let mut tokenized = prompt_tokens.to_vec();
        let mut output_tokens = Vec::with_capacity(max_new_tokens);
        let eos_id = self.vocab.eos_token_id();

        for _ in 0..max_new_tokens {
            if tokenized.len() >= self.max_context_length {
                break;
            }

            let input_vec: Vec<f32> = tokenized.iter().map(|&x| x as f32).collect();
            let token_input = match Array2::from_shape_vec((1, tokenized.len()), input_vec) {
                Ok(matrix) => matrix,
                Err(err) => {
                    log::error!("构造输入张量失败: {}", err);
                    break;
                }
            };

            let mut input = token_input;
            for layer in &mut self.network {
                // 生成阶段只做推理，不需要 ctx。
                let (out, _ctx) = layer.forward(&input);
                input = out;
            }

            if input.shape()[0] == 0 {
                break;
            }

            let last_logit = input
                .row(input.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            let probs = softmax(&last_logit);
            let next_token = self.sample_token_from_probs(&probs, temperature, top_p, top_k);

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == eos_id {
                break;
            }
        }

        perf_monitor.stop("inference_generation");
        let tokens_generated = output_tokens.len();
        let elapsed = generation_start.elapsed().as_secs_f32();
        if tokens_generated > 0 && elapsed > 0.0 {
            println!(
                "⚡ 推理吞吐量: {:.2} tokens/s ({} tokens)",
                tokens_generated as f32 / elapsed,
                tokens_generated
            );
        }

        output_tokens
    }

    fn generate_tokens_incremental(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Vec<usize> {
        if prompt_tokens.is_empty() || max_new_tokens == 0 {
            return Vec::new();
        }

        let max_context = self.max_context_length;
        let prompt_len = prompt_tokens.len();
        let start_idx = prompt_len.saturating_sub(max_context);
        let trimmed_prompt = &prompt_tokens[start_idx..];

        if trimmed_prompt.is_empty() {
            return Vec::new();
        }

        let max_allowed_by_context = max_context.saturating_sub(trimmed_prompt.len());
        if max_allowed_by_context == 0 {
            return Vec::new();
        }
        let generation_limit = max_new_tokens.min(max_allowed_by_context);
        if generation_limit == 0 {
            return Vec::new();
        }

        if !self.supports_incremental() {
            let mut tokens = self.generate_tokens_full(
                trimmed_prompt,
                generation_limit,
                temperature,
                top_p,
                top_k,
            );
            let eos_id = self.vocab.eos_token_id();
            if !tokens.iter().any(|&token| token == eos_id) {
                tokens.push(eos_id);
            }
            return tokens;
        }

        let mut perf_monitor = PerformanceMonitor::new();
        perf_monitor.start("inference_generation");
        let generation_start = Instant::now();

        let eos_id = self.vocab.eos_token_id();
        let mut session = InferenceSession::new(self, temperature, top_p, top_k);
        let mut logits = match session.prime_tokens(trimmed_prompt) {
            Some(logits) => logits,
            None => {
                perf_monitor.stop("inference_generation");
                drop(session);
                return Vec::new();
            }
        };

        let mut generated_tokens = Vec::with_capacity(generation_limit);

        for _ in 0..generation_limit {
            let next_token = session.sample_next_token(&logits);
            generated_tokens.push(next_token);

            logits = session.advance_with_token(next_token);
            if next_token == eos_id {
                break;
            }
        }

        perf_monitor.stop("inference_generation");

        let tokens_generated = generated_tokens.len();
        let elapsed = generation_start.elapsed().as_secs_f32();
        if tokens_generated > 0 && elapsed > 0.0 {
            println!(
                "⚡ 推理吞吐量: {:.2} tokens/s ({} tokens)",
                tokens_generated as f32 / elapsed,
                tokens_generated
            );
        }

        generated_tokens
    }

    /// 旧式 beam search 回退路径。
    ///
    /// 只有在网络不满足增量推理前提时才会进入这里；
    /// 当前默认模型一般不会走这条路径，因此暂时保留为教学/回退用途。
    fn beam_search_legacy(&mut self, text: &str, beam_width: usize, max_length: usize) -> String {
        if beam_width == 0 {
            return String::new();
        }

        let initial_tokens = self.tokenize(text);
        if initial_tokens.is_empty() {
            return String::new();
        }

        let mut current_beams = vec![(initial_tokens.clone(), 0.0f32)];

        for _ in initial_tokens.len()..max_length {
            self.beam_candidates_buffer.clear();

            for (seq, log_prob) in &current_beams {
                let input = match Array2::from_shape_vec(
                    (1, seq.len()),
                    seq.iter().map(|&x| x as f32).collect(),
                ) {
                    Ok(matrix) => matrix,
                    Err(err) => {
                        log::error!("构造输入张量失败: {}", err);
                        continue;
                    }
                };

                let mut input_tensor = input;
                for layer in &mut self.network {
                    // beam_search 属于推理路径，不需要 ctx。
                    let (out, _ctx) = layer.forward(&input_tensor);
                    input_tensor = out;
                }

                let probs = softmax(&input_tensor);
                let last_token_probs = probs.row(probs.nrows() - 1);

                self.sampling_idx_buffer.clear();
                self.sampling_idx_buffer.extend(
                    last_token_probs
                        .iter()
                        .enumerate()
                        .map(|(idx, &prob)| (prob, idx)),
                );

                self.sampling_idx_buffer
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

                for i in 0..beam_width.min(self.sampling_idx_buffer.len()) {
                    let (prob, token_id) = self.sampling_idx_buffer[i];
                    if prob > 0.0 {
                        let mut new_seq = seq.clone();
                        new_seq.push(token_id);
                        let new_log_prob = log_prob + prob.ln();
                        self.beam_candidates_buffer.push((new_seq, new_log_prob));
                    }
                }
            }

            if self.beam_candidates_buffer.is_empty() {
                break;
            }

            self.beam_candidates_buffer
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            current_beams = self
                .beam_candidates_buffer
                .iter()
                .take(beam_width)
                .cloned()
                .collect();

            if current_beams
                .iter()
                .any(|(seq, _)| seq.last() == Some(&self.vocab.eos_token_id()))
            {
                break;
            }
        }

        if let Some((best_seq, _)) = current_beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        {
            self.tokens_to_text(best_seq)
        } else {
            String::new()
        }
    }

    /// Beam search implementation
    /// 使用推理会话与 KV 缓存避免重复计算
    fn beam_search(&mut self, text: &str, beam_width: usize, max_length: usize) -> String {
        if !self.supports_incremental() {
            return self.beam_search_legacy(text, beam_width, max_length);
        }

        if beam_width == 0 {
            return String::new();
        }

        let initial_tokens = self.tokenize(text);
        if initial_tokens.is_empty() {
            return String::new();
        }

        let max_context = self.max_context_length;
        let prompt_len = initial_tokens.len();
        let start_idx = prompt_len.saturating_sub(max_context);
        let trimmed_prompt = &initial_tokens[start_idx..];

        let prompt_base_len = trimmed_prompt.len();
        let target_length = max_length.max(prompt_base_len).min(self.max_context_length);

        if target_length <= prompt_base_len {
            return self.tokens_to_text(trimmed_prompt);
        }

        #[derive(Clone)]
        struct BeamCandidate {
            tokens: Vec<usize>,
            log_prob: f32,
            state: SessionSnapshot,
            logits: Array2<f32>,
        }

        let mut perf_monitor = PerformanceMonitor::new();
        perf_monitor.start("inference_generation");
        let generation_start = Instant::now();

        let eos_id = self.vocab.eos_token_id();
        let mut session = InferenceSession::new(self, 1.0, 1.0, 0);
        let initial_logits = match session.prime_tokens(trimmed_prompt) {
            Some(logits) => logits,
            None => {
                perf_monitor.stop("inference_generation");
                drop(session);
                return self.tokens_to_text(trimmed_prompt);
            }
        };

        let mut beams = vec![BeamCandidate {
            tokens: trimmed_prompt.to_vec(),
            log_prob: 0.0,
            state: session.snapshot(),
            logits: initial_logits,
        }];

        let max_steps = target_length - trimmed_prompt.len();
        let mut buffer: Vec<(f32, usize)> = Vec::new();

        for _ in 0..max_steps {
            let mut expanded: Vec<BeamCandidate> = Vec::new();

            for candidate in &beams {
                let is_complete = candidate.tokens.len() >= target_length
                    || candidate.tokens.last() == Some(&eos_id);
                if is_complete {
                    expanded.push(candidate.clone());
                    continue;
                }

                session.restore(&candidate.state);
                let probs = softmax(&candidate.logits);
                let last_row = probs.row(probs.nrows() - 1);

                buffer.clear();
                buffer.extend(last_row.iter().enumerate().map(|(idx, &prob)| (prob, idx)));

                let top = beam_width.min(buffer.len());
                if top == 0 {
                    expanded.push(candidate.clone());
                    continue;
                }

                let nth = top - 1;
                buffer.select_nth_unstable_by(nth, |a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
                });
                buffer[..top]
                    .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

                for &(prob, token_idx) in buffer[..top].iter() {
                    if prob <= 0.0 {
                        continue;
                    }

                    let log_prob = candidate.log_prob + prob.max(SOFTMAX_EPSILON).ln();

                    session.restore(&candidate.state);
                    let next_logits = session.advance_with_token(token_idx);
                    let snapshot = session.snapshot();

                    let mut new_tokens = candidate.tokens.clone();
                    new_tokens.push(token_idx);

                    expanded.push(BeamCandidate {
                        tokens: new_tokens,
                        log_prob,
                        state: snapshot,
                        logits: next_logits,
                    });
                }
            }

            if expanded.is_empty() {
                break;
            }

            expanded.sort_by(|a, b| {
                b.log_prob
                    .partial_cmp(&a.log_prob)
                    .unwrap_or(Ordering::Equal)
            });
            beams = expanded.into_iter().take(beam_width).collect();

            let all_complete = beams.iter().all(|candidate| {
                candidate.tokens.len() >= target_length || candidate.tokens.last() == Some(&eos_id)
            });
            if all_complete {
                break;
            }
        }

        perf_monitor.stop("inference_generation");

        let best_candidate = beams
            .iter()
            .max_by(|a, b| {
                a.log_prob
                    .partial_cmp(&b.log_prob)
                    .unwrap_or(Ordering::Equal)
            })
            .cloned();

        drop(session);

        if let Some(best) = best_candidate {
            let generated_tokens = best.tokens.len().saturating_sub(prompt_base_len);
            let elapsed = generation_start.elapsed().as_secs_f32();
            if generated_tokens > 0 && elapsed > 0.0 {
                println!(
                    "⚡ 推理吞吐量: {:.2} tokens/s ({} tokens)",
                    generated_tokens as f32 / elapsed,
                    generated_tokens
                );
            }
            self.tokens_to_text(&best.tokens)
        } else {
            String::new()
        }
    }

    fn tokens_to_text(&self, tokens: &[usize]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let token_strs: Vec<String> = tokens
            .iter()
            .filter_map(|&token| self.vocab.decode.get(&token).cloned())
            .collect();

        let raw_output = token_strs.join(" ");
        self.post_process_chinese_text(&raw_output)
    }

    /// Post-process generated Chinese text to improve fluency and accuracy
    pub fn post_process_chinese_text(&self, text: &str) -> String {
        // Remove extra spaces between Chinese characters
        let mut result = String::new();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            result.push(ch);

            // If current and next characters are both Chinese, don't add space
            if let Some(&next_ch) = chars.peek() {
                if self.is_chinese_char(ch) && self.is_chinese_char(next_ch) {
                    // Skip any space between Chinese characters
                    if next_ch == ' ' {
                        chars.next(); // consume the space
                    }
                }
            }
        }

        // Additional processing could include:
        // - Grammar pattern correction
        // - Ensuring proper sentence structure
        // - Removing repetitive patterns

        result
    }

    /// Helper function to check if a character is Chinese
    fn is_chinese_char(&self, ch: char) -> bool {
        (ch as u32) >= 0x4E00 && (ch as u32) <= 0x9FFF
    }

    pub(crate) fn cross_entropy_from_log_probs(
        log_probs: &Array2<f32>,
        target: &[usize],
        pad_token_id: usize,
    ) -> f32 {
        // 使用 log_softmax 输出计算交叉熵，避免对概率取对数的数值不稳定
        let mut loss = 0.0;
        let mut n_targets = 0usize;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            // PAD 不应该参与 loss（否则会把“补齐的空白”当成训练信号）。
            if target_idx == pad_token_id {
                continue;
            }

            if row_idx >= log_probs.shape()[0] || target_idx >= log_probs.shape()[1] {
                log::warn!(
                    "cross_entropy_from_log_probs 越界：row_idx={}, target_idx={}, log_probs_shape={:?}",
                    row_idx,
                    target_idx,
                    log_probs.dim()
                );
                continue;
            }

            let lp = log_probs[[row_idx, target_idx]];
            loss -= lp; // NLL: -log p(target)
            n_targets += 1;
        }

        if n_targets == 0 {
            0.0
        } else {
            loss / (n_targets as f32)
        }
    }

    /// 计算交叉熵损失（从softmax概率）
    pub fn cross_entropy_loss_step(
        probs: &Array2<f32>,
        target: &[usize],
        pad_token_id: usize,
    ) -> f32 {
        use crate::LOG_EPSILON;
        let mut loss = 0.0f32;
        let mut n_targets = 0usize;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            if target_idx == pad_token_id {
                continue;
            }

            if row_idx >= probs.shape()[0] || target_idx >= probs.shape()[1] {
                log::warn!(
                    "cross_entropy_loss_step 越界：row_idx={}, target_idx={}, probs_shape={:?}",
                    row_idx,
                    target_idx,
                    probs.dim()
                );
                continue;
            }

            let prob = probs[[row_idx, target_idx]].max(LOG_EPSILON);
            loss -= prob.ln();
            n_targets += 1;
        }

        if n_targets == 0 {
            0.0
        } else {
            loss / (n_targets as f32)
        }
    }

    pub fn compute_gradients_step(
        probs: &Array2<f32>,
        target: &[usize],
        pad_token_id: usize,
    ) -> Result<Option<Array2<f32>>, TrainingSignalError> {
        let mut grads = probs.clone(); // softmax - one_hot(target)

        if probs.shape()[0] != target.len() {
            log::error!(
                "梯度计算输入不匹配：probs行数={}，target长度={}",
                probs.shape()[0],
                target.len()
            );
            // 关键：如果继续 backward（哪怕传入 0 梯度），Adam 的动量也可能导致参数漂移。
            // 因此这里返回 Err，强制调用方跳过本步 optimizer step。
            return Err(TrainingSignalError::ShapeMismatch {
                probs_rows: probs.shape()[0],
                target_len: target.len(),
            });
        }

        let mut n_targets = 0usize;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            if target_idx == pad_token_id {
                // 与 loss 计算保持一致：PAD 不参与训练信号，因此该位置的梯度应为 0。
                grads.row_mut(row_idx).fill(0.0);
                continue;
            }

            if target_idx >= grads.shape()[1] {
                log::error!(
                    "compute_gradients_step target 越界：row_idx={}, target_idx={}, probs_shape={:?}",
                    row_idx,
                    target_idx,
                    probs.dim()
                );
                return Err(TrainingSignalError::TargetOutOfRange {
                    row_idx,
                    target_idx,
                    vocab_size: grads.shape()[1],
                });
            }

            grads[[row_idx, target_idx]] -= 1.0;
            n_targets += 1;
        }

        if n_targets == 0 {
            // 全部是 PAD：没有训练信号。必须跳过 optimizer step，避免 Adam 动量导致“0 梯度漂移”。
            return Ok(None);
        }

        grads.mapv_inplace(|x| x / (n_targets as f32));
        Ok(Some(grads))
    }

    pub fn clip_gradients(grads: &mut Array2<f32>, max_norm: f32) {
        // 计算L2范数并裁剪
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }

    // 说明：旧版实现中存在 “logits 梯度累积后只 backward 一次” 的数学错误（缓存覆盖）。
    // 为了避免误用，我们已完全移除旧的梯度桶聚合函数。
}

#[cfg(test)]
mod tests {
    use super::{EpochAccumulator, LLM};
    use crate::{
        embeddings::Embeddings, output_projection::OutputProjection, vocab::Vocab, EMBEDDING_DIM,
    };
    use ndarray::{arr2, Array2};

    fn make_two_layer_training_model(vocab: Vocab) -> LLM {
        let vocab_size = vocab.len();

        let mut embeddings = Embeddings::new(vocab.clone());
        embeddings.token_embeddings =
            Array2::from_shape_fn((vocab_size, EMBEDDING_DIM), |(r, c)| {
                0.002 * (r as f32) - 0.0001 * (c as f32)
            });

        let mut output = OutputProjection::new(EMBEDDING_DIM, vocab_size);
        output.w_out = Array2::from_shape_fn((EMBEDDING_DIM, vocab_size), |(r, c)| {
            0.0003 * (r as f32 + 1.0) - 0.0004 * (c as f32)
        });
        output.b_out = Array2::from_shape_fn((1, vocab_size), |(_, c)| -0.01 * (c as f32));

        LLM::new(vocab, vec![Box::new(embeddings), Box::new(output)])
    }

    fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f32, |m, (&x, &y)| m.max((x - y).abs()))
    }

    fn embedding_weights(model: &LLM) -> Array2<f32> {
        model.network[0]
            .as_any()
            .downcast_ref::<Embeddings>()
            .expect("expected Embeddings layer")
            .token_embeddings
            .clone()
    }

    fn output_weights(model: &LLM) -> Array2<f32> {
        model.network[1]
            .as_any()
            .downcast_ref::<OutputProjection>()
            .expect("expected OutputProjection layer")
            .w_out
            .clone()
    }

    fn output_bias(model: &LLM) -> Array2<f32> {
        model.network[1]
            .as_any()
            .downcast_ref::<OutputProjection>()
            .expect("expected OutputProjection layer")
            .b_out
            .clone()
    }

    #[test]
    fn accumulation_single_step_keeps_full_weight() {
        let weight = LLM::token_weighted_micro_batch_weight(3, 3);
        assert!((weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accumulation_weights_are_token_weighted_not_sequence_weighted() {
        let short = LLM::token_weighted_micro_batch_weight(1, 3);
        let long = LLM::token_weighted_micro_batch_weight(2, 3);

        assert!((short - (1.0 / 3.0)).abs() < 1e-6);
        assert!((long - (2.0 / 3.0)).abs() < 1e-6);
        assert!((short + long - 1.0).abs() < 1e-6);
        assert!(
            (long - 0.5).abs() > 1e-6,
            "long sequence should not collapse to sequence-weighted 0.5"
        );
    }

    #[test]
    fn accumulation_tail_batch_uses_remaining_tokens() {
        let tail_short = LLM::token_weighted_micro_batch_weight(1, 4);
        let tail_long = LLM::token_weighted_micro_batch_weight(3, 4);

        assert!((tail_short - 0.25).abs() < 1e-6);
        assert!((tail_long - 0.75).abs() < 1e-6);
    }

    #[test]
    fn rescale_logits_grads_for_accumulation_restores_sum_nll_scale() {
        let mut grads = arr2(&[[0.25_f32, -0.25_f32], [0.75_f32, -0.75_f32]]);
        LLM::rescale_logits_grads_for_accumulation(&mut grads, 2);

        let expected = arr2(&[[0.5_f32, -0.5_f32], [1.5_f32, -1.5_f32]]);
        assert_eq!(grads, expected);
    }

    #[test]
    fn train_monitored_accumulation_matches_manual_epoch_replay() {
        let texts = vec!["a".to_string(), "a b".to_string()];
        let vocab = Vocab::build_from_texts(&texts);

        let mut monitored = make_two_layer_training_model(vocab.clone());
        let mut manual = make_two_layer_training_model(vocab.clone());

        let epochs = monitored.train_monitored(vec!["a", "a b"], 1, 0.05, 10, 2);
        assert_eq!(epochs, 1);

        let tokenized_data: Vec<Vec<usize>> = ["a", "a b"]
            .iter()
            .map(|input| LLM::tokenize_training_with_vocab(&manual.vocab, input))
            .collect();
        let pad_token_id = manual.vocab.pad_token_id();
        let warmup_epochs = LLM::recommend_warmup_epochs(1);
        let current_lr = LLM::cosine_with_warmup_lr(0.05, 0, 1, 0, warmup_epochs);

        manual.set_training_mode(true);
        manual.zero_grad_accum();
        let mut accum_counter = 0usize;
        let mut accum_tokens = 0usize;

        for training_row in &tokenized_data {
            let input_ids = &training_row[..training_row.len() - 1];
            let target_ids = &training_row[1..];
            let Some(mut step) = manual.prepare_training_step(input_ids, target_ids, pad_token_id)
            else {
                continue;
            };

            LLM::clip_gradients(&mut step.grads_output, 1.0);
            LLM::rescale_logits_grads_for_accumulation(&mut step.grads_output, step.n_targets);
            manual.backward_accumulate_with_ctx(&step.layer_ctxs, &step.grads_output);
            accum_counter += 1;
            accum_tokens += step.n_targets;

            if accum_counter >= 2 {
                manual.step_accumulated(current_lr, LLM::token_weighted_accum_scale(accum_tokens));
                accum_counter = 0;
                accum_tokens = 0;
            }
        }

        if accum_counter > 0 {
            manual.step_accumulated(current_lr, LLM::token_weighted_accum_scale(accum_tokens));
        }
        manual.set_training_mode(false);

        let tol = 1e-6_f32;
        assert!(max_diff(&embedding_weights(&monitored), &embedding_weights(&manual)) < tol);
        assert!(max_diff(&output_weights(&monitored), &output_weights(&manual)) < tol);
        assert!(max_diff(&output_bias(&monitored), &output_bias(&manual)) < tol);
    }

    #[test]
    fn apply_temperature_keeps_extremely_low_temperature_sampling_stable() {
        let probs = arr2(&[[0.7_f32, 0.2_f32, 0.1_f32]]);
        let adjusted = LLM::apply_temperature(&probs, 1e-6);

        let sum: f32 = adjusted.row(0).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "temperature-adjusted probabilities should stay normalized"
        );
        assert!(
            adjusted.iter().all(|value| value.is_finite()),
            "temperature-adjusted probabilities should stay finite"
        );
        assert!(
            adjusted[[0, 0]] > 0.999_999,
            "very low temperature should collapse toward the argmax token instead of random fallback"
        );
        assert!(adjusted[[0, 1]] < 1e-6);
        assert!(adjusted[[0, 2]] < 1e-6);
    }

    #[test]
    fn sample_token_from_probs_uses_greedy_path_for_non_positive_temperature() {
        let vocab = Vocab::build_from_texts(&["a".to_string(), "b".to_string(), "c".to_string()]);
        let mut model = make_two_layer_training_model(vocab);
        let probs = arr2(&[[0.1_f32, 0.7_f32, 0.2_f32]]);

        let next = model.sample_token_from_probs(&probs, 0.0, 0.9, 0);
        assert_eq!(next, 1);
    }

    #[test]
    fn sample_from_probs_zero_sum_falls_back_to_argmax_instead_of_random() {
        let vocab = Vocab::build_from_texts(&["a".to_string(), "b".to_string(), "c".to_string()]);
        let model = make_two_layer_training_model(vocab);
        let choice = model.sample_from_probs(&[0.0_f32, 0.0_f32, 0.0_f32]);
        assert_eq!(choice, 0);
    }

    #[test]
    fn train_monitored_returns_zero_when_no_valid_samples_exist() {
        let texts = vec!["示例".to_string()];
        let vocab = Vocab::build_from_texts(&texts);
        let mut model = make_two_layer_training_model(vocab);

        let epochs = model.train_monitored(Vec::new(), 3, 0.01, 2, 1);
        assert_eq!(epochs, 0);
    }

    #[test]
    fn train_bucketed_sequential_returns_zero_when_no_valid_samples_exist() {
        let texts = vec!["示例".to_string()];
        let vocab = Vocab::build_from_texts(&texts);
        let mut model = make_two_layer_training_model(vocab);

        let epochs = model.train_bucketed_sequential(Vec::new(), 3, 0.01, 2, 2);
        assert_eq!(epochs, 0);
    }

    #[test]
    #[should_panic(expected = "LLM::new 收到不受支持的网络拓扑")]
    fn llm_new_rejects_non_teaching_topology() {
        let vocab = Vocab::build_from_texts(&["a".to_string(), "b".to_string()]);
        let embeddings = Embeddings::new(vocab.clone());
        let output = OutputProjection::new(EMBEDDING_DIM, vocab.len());

        let _ = LLM::new(vocab, vec![Box::new(output), Box::new(embeddings)]);
    }

    #[test]
    fn epoch_accumulator_uses_token_weighted_loss() {
        let mut tracker = EpochAccumulator::default();
        tracker.total_nll = 3.0;
        tracker.total_tokens = 2;
        tracker.total_grad_norm = 4.0;
        tracker.sample_count = 2;

        assert_eq!(tracker.avg_loss(), Some(1.5));
        assert_eq!(tracker.avg_grad_norm(), Some(2.0));
        assert!(tracker.has_valid_samples());
    }
}
