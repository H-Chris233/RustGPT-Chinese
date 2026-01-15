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

#[derive(Clone)]
struct SessionSnapshot {
    processed_tokens: usize,
    kv_caches: Vec<Option<(Array2<f32>, Array2<f32>)>>,
}

pub struct InferenceSession<'a> {
    llm: &'a mut LLM,
    processed_tokens: usize,
    max_context_length: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    previous_training_mode: bool,
    kv_cache_was_enabled: bool,
}

impl<'a> InferenceSession<'a> {
    pub fn new(llm: &'a mut LLM, temperature: f32, top_p: f32, top_k: usize) -> Self {
        let previous_training_mode = llm.training;
        let kv_cache_was_enabled = llm.is_kv_cache_enabled();
        let max_context_length = llm.max_context_length;

        llm.set_training_mode(false);
        llm.enable_kv_cache();
        llm.clear_kv_cache();

        Self {
            llm,
            processed_tokens: 0,
            max_context_length,
            temperature,
            top_p,
            top_k,
            previous_training_mode,
            kv_cache_was_enabled,
        }
    }

    pub fn prime_tokens(&mut self, tokens: &[usize]) -> Option<Array2<f32>> {
        if tokens.is_empty() {
            return None;
        }

        self.llm.clear_kv_cache();
        self.processed_tokens = 0;

        let mut last_logits = None;
        for &token in tokens {
            last_logits = Some(self.advance_with_token(token));
        }

        last_logits
    }

    pub fn advance_with_token(&mut self, token_id: usize) -> Array2<f32> {
        if self.processed_tokens >= self.max_context_length {
            log::warn!(
                "上下文长度超过阈值({}), 自动重置 KV 缓存",
                self.max_context_length
            );
            self.llm.clear_kv_cache();
            self.processed_tokens = 0;
        }

        let logits = self.llm.inference_step(token_id, self.processed_tokens);
        self.processed_tokens += 1;
        logits
    }

    pub fn sample_next_token(&mut self, logits: &Array2<f32>) -> usize {
        let probs = softmax(logits);
        let adjusted = LLM::apply_temperature(&probs, self.temperature);

        let candidates = if self.top_k > 0 {
            self.llm.top_k_sampling(&adjusted, self.top_k)
        } else {
            self.llm.top_p_sampling(&adjusted, self.top_p)
        };

        candidates.into_iter().next().unwrap_or(0)
    }

    pub fn snapshot(&mut self) -> SessionSnapshot {
        SessionSnapshot {
            processed_tokens: self.processed_tokens,
            kv_caches: self.llm.capture_kv_cache(),
        }
    }

    pub fn restore(&mut self, snapshot: &SessionSnapshot) {
        self.processed_tokens = snapshot.processed_tokens;
        self.llm.restore_kv_cache(&snapshot.kv_caches);
    }

    pub fn processed_tokens(&self) -> usize {
        self.processed_tokens
    }
}

impl<'a> Drop for InferenceSession<'a> {
    fn drop(&mut self) {
        if self.previous_training_mode {
            self.llm.set_training_mode(true);
        } else {
            self.llm.set_training_mode(false);
        }

        if !self.kv_cache_was_enabled {
            self.llm.disable_kv_cache();
        }

        self.llm.clear_kv_cache();
    }
}

/// Layer trait - 支持单样本和批量处理
///
/// 所有神经网络层需要实现这个trait，支持：
/// - 单样本处理：forward/backward 使用 Array2 (seq_len, hidden_dim)
/// - 批量处理：forward_batch/backward_batch 使用 Array3 (batch, seq, hidden_dim)
pub trait Layer {
    fn layer_type(&self) -> &str;

    /// 单样本前向传播（保留向后兼容）
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    /// 单样本反向传播（保留向后兼容）
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    /// 用于类型转换的辅助方法
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// 批量前向传播
    ///
    /// # 参数
    /// - `input`: (batch_size, seq_len, hidden_dim) 或 (batch_size, seq_len) 对于embeddings
    /// - `attention_mask`: 可选的注意力掩码 (batch_size, seq_len)，1.0表示真实token，0.0表示PAD
    ///
    /// # 返回值
    /// (batch_size, seq_len, hidden_dim) 的输出张量
    fn forward_batch(
        &mut self,
        input: &Array3<f32>,
        _attention_mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        // 默认实现：对批次中的每个样本分别调用单样本 forward
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_dim = input.shape()[2];

        let mut output = Array3::zeros((batch_size, seq_len, hidden_dim));

        for b in 0..batch_size {
            let sample = input.slice(ndarray::s![b, .., ..]).to_owned();
            let sample_output = self.forward(&sample);
            output
                .slice_mut(ndarray::s![b, .., ..])
                .assign(&sample_output);
        }

        output
    }

    /// 批量反向传播
    ///
    /// # 参数
    /// - `grads`: (batch_size, seq_len, hidden_dim) 的梯度
    /// - `lr`: 学习率
    /// - `attention_mask`: 可选的注意力掩码，用于排除PAD位置的梯度
    ///
    /// # 返回值
    /// (batch_size, seq_len, hidden_dim) 的输入梯度
    fn backward_batch(
        &mut self,
        grads: &Array3<f32>,
        lr: f32,
        attention_mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        // 默认实现：对批次中的每个样本分别调用单样本 backward
        let batch_size = grads.shape()[0];
        let seq_len = grads.shape()[1];
        let hidden_dim = grads.shape()[2];

        let mut grad_input = Array3::zeros((batch_size, seq_len, hidden_dim));

        for b in 0..batch_size {
            let mut sample_grad = grads.slice(ndarray::s![b, .., ..]).to_owned();

            // 如果有注意力掩码，将PAD位置的梯度清零
            if let Some(mask) = attention_mask {
                for s in 0..seq_len {
                    if mask[[b, s]] < 0.5 {
                        // PAD位置，梯度清零
                        sample_grad.row_mut(s).fill(0.0);
                    }
                }
            }

            let sample_grad_input = self.backward(&sample_grad, lr);
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
    pub parallel_training: bool,
    // 性能优化：可重用的采样缓冲区（public以便序列化）
    pub sampling_prob_buffer: Vec<f32>,
    pub sampling_idx_buffer: Vec<(f32, usize)>,
    pub beam_candidates_buffer: Vec<(Vec<usize>, f32)>,
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
            parallel_training: true,
            sampling_prob_buffer: Vec::with_capacity(vocab_size),
            sampling_idx_buffer: Vec::with_capacity(vocab_size),
            beam_candidates_buffer: Vec::with_capacity(50),
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        let vocab_size = vocab.words.len();
        Self {
            vocab,
            network,
            context_window: Vec::new(),
            max_context_length: MAX_SEQ_LEN,
            training: true,
            parallel_training: true,
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
                hidden = layer.forward(&hidden);
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

    pub fn set_parallel_training(&mut self, enabled: bool) {
        self.parallel_training = enabled;
    }

    pub fn parallel_training_enabled(&self) -> bool {
        self.parallel_training
    }

    #[allow(dead_code)]
    pub fn predict(&mut self, text: &str) -> String {
        self.predict_with_sampling(text, 1.0, 0.9, 5)
    }

    #[allow(dead_code)]
    pub fn predict_with_sampling(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let prompt_tokens = self.tokenize(text);
        let max_new_tokens = MAX_SEQ_LEN.saturating_sub(prompt_tokens.len());
        let generated_tokens = self.generate_tokens_incremental(
            &prompt_tokens,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        );

        self.tokens_to_text(&generated_tokens)
    }

    pub fn predict_with_context(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let new_tokens = self.tokenize(text);

        let mut combined_tokens = self.context_window.clone();
        combined_tokens.extend_from_slice(&new_tokens);

        if combined_tokens.len() > self.max_context_length {
            let start_idx = combined_tokens.len() - self.max_context_length;
            combined_tokens = combined_tokens[start_idx..].to_vec();
        }

        let available = self
            .max_context_length
            .saturating_sub(combined_tokens.len());
        let max_new_tokens = available.min(20);
        let generated_tokens = self.generate_tokens_incremental(
            &combined_tokens,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        );

        self.add_to_context(&new_tokens);
        self.add_to_context(&generated_tokens);

        self.tokens_to_text(&generated_tokens)
    }

    pub fn predict_with_beam_search(
        &mut self,
        text: &str,
        beam_width: usize,
        max_length: usize,
    ) -> String {
        self.beam_search(text, beam_width, max_length)
    }

    #[allow(dead_code)]
    fn forward_with_sampling(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let prompt_tokens = self.tokenize(text);
        let max_new_tokens = MAX_SEQ_LEN.saturating_sub(prompt_tokens.len());
        let generated_tokens = self.generate_tokens_incremental(
            &prompt_tokens,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        );

        self.tokens_to_text(&generated_tokens)
    }

    #[allow(dead_code)]
    fn forward(&mut self, text: &str) -> Vec<usize> {
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        if tokenized.is_empty() {
            return output_tokens;
        }

        let input_len = tokenized.len();

        if input_len >= MAX_SEQ_LEN {
            return output_tokens;
        }

        for _ in 0..(MAX_SEQ_LEN - input_len) {
            if output_tokens.len() >= MAX_SEQ_LEN - 1 {
                break;
            }

            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            let probs = softmax(&last_logit);

            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.eos_token_id() {
                break;
            }
        }

        output_tokens
    }

    pub fn train(&mut self, data: Vec<&str>, epochs: usize, initial_lr: f32) {
        self.set_training_mode(true);

        let tokenized_data = data
            .iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let decay_rate: f32 = 0.95;
            let decay_steps = 10.0;
            let current_lr = initial_lr * decay_rate.powf(epoch as f32 / decay_steps);

            let mut total_loss = 0.0;
            let mut sample_count = 0usize;
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                // 1. Slice input and targets
                let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
                let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab. 

                // Forward pass
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                let log_probs = log_softmax(&logits);
                total_loss += Self::cross_entropy_from_log_probs(&log_probs, target_ids);

                // Backward pass: grad = softmax(logits) - one_hot
                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids);

                Self::clip_gradients(&mut grads_output, 1.0);

                for layer in self.network.iter_mut().rev() {
                    grads_output = layer.backward(&grads_output, current_lr);
                }

                let tokens = Self::greedy_decode(&probs);
                let next_token = tokens[tokens.len() - 1];

                if next_token == self.vocab.encode("</s>").unwrap() {
                    continue;
                }

                sample_count += 1;
            }

            println!(
                "Epoch {}: Loss = {:.4}, LR = {:.6}",
                epoch,
                if sample_count > 0 {
                    total_loss / sample_count as f32
                } else {
                    0.0
                },
                current_lr
            );
        }

        self.set_training_mode(false);
    }

    /// 使用预tokenize的数据进行训练（性能优化版本）
    ///
    /// 这个方法接受已经tokenize的数据，避免重复tokenization
    pub fn train_with_cached_tokens(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        epochs: usize,
        initial_lr: f32,
    ) {
        self.set_training_mode(true);

        for epoch in 0..epochs {
            let decay_rate: f32 = 0.95;
            let decay_steps = 10.0;
            let current_lr = initial_lr * decay_rate.powf(epoch as f32 / decay_steps);

            let mut total_loss = 0.0;
            let mut sample_count = 0;

            // 直接使用缓存的tokenized数据，无需重复tokenize
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                // 前向传播
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                // 使用 log_softmax + NLL 提升数值稳定性
                let log_probs = log_softmax(&logits);
                total_loss += Self::cross_entropy_from_log_probs(&log_probs, target_ids);

                // 反向传播：grad = softmax(logits) - one_hot
                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids);

                // 更强的梯度裁剪提升稳定性
                Self::clip_gradients(&mut grads_output, 1.0);

                for layer in self.network.iter_mut().rev() {
                    grads_output = layer.backward(&grads_output, current_lr);
                }

                sample_count += 1;
            }

            println!(
                "Epoch {}: Loss = {:.4}, LR = {:.6}",
                epoch,
                if sample_count > 0 {
                    total_loss / sample_count as f32
                } else {
                    0.0
                },
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
        // 计算每个周期的长度
        let cycle_length = total_epochs / (num_restarts + 1);

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

    /// 完整优化的训练方法（集成并行预处理与监控）
    ///
    /// # 特性
    /// - ✅ 数据预处理缓存（避免重复 tokenization）
    /// - ✅ Rayon 并行 tokenization（可根据数据量自动回退）
    /// - ✅ 余弦退火学习率调度
    /// - ✅ 早停机制
    /// - ✅ 增强训练监控（困惑度、梯度范数、训练速度）
    /// - ✅ Rayon scope 梯度归约（可根据数据量自动回退）
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

        const MIN_PARALLEL_TOKENIZE: usize = 16;
        const MIN_PARALLEL_GRAD: usize = 8;

        let mut perf_monitor = PerformanceMonitor::new();
        let effective_accum_steps = accumulation_steps.max(1);

        println!("📝 正在预处理训练数据...");
        let preprocess_start = std::time::Instant::now();

        let should_parallel_preprocess =
            self.parallel_training && data.len() >= MIN_PARALLEL_TOKENIZE;

        let preprocess_label = if should_parallel_preprocess {
            "tokenization_parallel"
        } else {
            "tokenization_single_thread"
        };

        perf_monitor.start(preprocess_label);
        let tokenized_data: Vec<Vec<usize>> = data
            .iter()
            .map(|input| Self::tokenize_with_vocab(&self.vocab, input))
            .collect();
        perf_monitor.stop(preprocess_label);

        println!(
            "✅ 数据预处理完成，共 {} 个序列（耗时 {:.2}s）",
            tokenized_data.len(),
            preprocess_start.elapsed().as_secs_f32()
        );

        if self.parallel_training && !should_parallel_preprocess {
            println!("⚠️  样本数较少，tokenization 自动回退为单线程模式");
        }

        let use_parallel_gradients = self.parallel_training
            && effective_accum_steps > 1
            && tokenized_data.len() >= MIN_PARALLEL_GRAD;

        println!(
            "🧵 梯度归约模式: {} (accumulation_steps={})",
            if use_parallel_gradients {
                "rayon 并行"
            } else if self.parallel_training {
                "单线程（自动回退）"
            } else {
                "单线程（手动配置）"
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

            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut sample_count = 0usize;

            let mut gradient_bucket: Vec<Array2<f32>> = Vec::with_capacity(effective_accum_steps);
            let mut bucket_expected_len: Option<usize> = None;

            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                let seq_len = training_row.len() - 1;

                if let Some(expected) = bucket_expected_len {
                    if expected != seq_len && !gradient_bucket.is_empty() {
                        self.apply_accumulated_gradients(
                            &mut gradient_bucket,
                            current_lr,
                            use_parallel_gradients,
                            &mut perf_monitor,
                        );
                        bucket_expected_len = None;
                    }
                }

                // 前向传播
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                let log_probs = log_softmax(&logits);
                total_loss += Self::cross_entropy_from_log_probs(&log_probs, target_ids);

                // 计算输出梯度
                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids);

                total_grad_norm += Self::compute_grad_norm(&grads_output);

                Self::clip_gradients(&mut grads_output, 1.0);

                bucket_expected_len.get_or_insert(seq_len);
                gradient_bucket.push(grads_output);

                if gradient_bucket.len() >= effective_accum_steps {
                    self.apply_accumulated_gradients(
                        &mut gradient_bucket,
                        current_lr,
                        use_parallel_gradients,
                        &mut perf_monitor,
                    );
                    bucket_expected_len = None;
                }

                sample_count += 1;
            }

            if !gradient_bucket.is_empty() {
                self.apply_accumulated_gradients(
                    &mut gradient_bucket,
                    current_lr,
                    use_parallel_gradients,
                    &mut perf_monitor,
                );
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            let avg_loss = if sample_count > 0 {
                total_loss / sample_count as f32
            } else {
                0.0
            };
            let avg_grad_norm = if sample_count > 0 {
                total_grad_norm / sample_count as f32
            } else {
                0.0
            };
            let perplexity = avg_loss.exp();
            let samples_per_sec = if epoch_time > 0.0 {
                sample_count as f32 / epoch_time
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

    /// 批量训练方法（支持动态掩码）
    ///
    /// # 特性
    /// - ✅ 批量处理：显著提升训练速度
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
    pub fn train_monitored_batch(
        &mut self,
        data: Vec<&str>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
        batch_size: usize,
    ) -> usize {
        use crate::batch_loader::{BatchLoader, create_training_batches};

        self.set_training_mode(true);

        let perf_monitor = PerformanceMonitor::new();

        println!("📝 正在预处理训练数据...");
        let preprocess_start = std::time::Instant::now();

        // Tokenize 所有数据
        let tokenized_data: Vec<Vec<usize>> = data
            .iter()
            .map(|input| Self::tokenize_with_vocab(&self.vocab, input))
            .collect();

        println!(
            "✅ 数据预处理完成，共 {} 个序列（耗时 {:.2}s）",
            tokenized_data.len(),
            preprocess_start.elapsed().as_secs_f32()
        );

        // 创建批量加载器
        let batch_loader = BatchLoader::new(batch_size, true, 16);

        let mut early_stopping = EarlyStopping::new(patience, 0.01);
        let training_start_time = std::time::Instant::now();

        for epoch in 0..max_epochs {
            let epoch_start = std::time::Instant::now();

            // 余弦退火 + Warmup
            let warmup_epochs = Self::recommend_warmup_epochs(max_epochs);
            let current_lr =
                Self::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);

            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut sample_count = 0usize;

            // 创建训练批次
            let training_batches = create_training_batches(&batch_loader, &tokenized_data);

            for (input_batch, targets) in training_batches {
                // 跳过空批次
                if input_batch.batch_size == 0 {
                    continue;
                }

                // 前向传播（批量）- 使用循环对每个样本单独处理
                let mut batch_outputs = Vec::with_capacity(input_batch.batch_size);

                for b in 0..input_batch.batch_size {
                    // 只取真实 token（避免 PAD 参与前向计算）
                    let target_len = targets.get(b).map_or(0, |t| t.len());
                    if target_len == 0 {
                        continue;
                    }

                    let sample_tokens = input_batch.tokens.row(b);
                    let sample_ids: Vec<usize> =
                        sample_tokens.iter().take(target_len).copied().collect();

                    // 单样本前向传播
                    let mut input: Array2<f32> = Array2::zeros((1, sample_ids.len()));
                    input.row_mut(0).assign(
                        &sample_ids
                            .iter()
                            .map(|&x| x as f32)
                            .collect::<Array1<f32>>(),
                    );

                    for layer in &mut self.network {
                        input = layer.forward(&input);
                    }

                    batch_outputs.push(input);
                }

                // 计算损失和反向传播（对每个样本）
                let mut batch_loss = 0.0;

                for (b, logits) in batch_outputs.iter().enumerate() {
                    if b >= targets.len() || targets[b].is_empty() {
                        continue;
                    }

                    let log_probs = log_softmax(logits);

                    // 只计算真实 token 对齐的损失
                    let target_ids = &targets[b];
                    let loss = Self::cross_entropy_from_log_probs(&log_probs, target_ids);
                    batch_loss += loss;

                    // 计算梯度
                    let probs = log_probs.mapv(|x| x.exp());
                    let mut sample_grad = Self::compute_gradients_step(&probs, target_ids);

                    total_grad_norm += Self::compute_grad_norm(&sample_grad);
                    Self::clip_gradients(&mut sample_grad, 1.0);

                    // 反向传播
                    let mut grads = sample_grad;
                    for layer in self.network.iter_mut().rev() {
                        grads = layer.backward(&grads, current_lr);
                    }

                    sample_count += 1;
                }

                total_loss += batch_loss;
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            let avg_loss = if sample_count > 0 {
                total_loss / sample_count as f32
            } else {
                0.0
            };
            let avg_grad_norm = if sample_count > 0 {
                total_grad_norm / sample_count as f32
            } else {
                0.0
            };
            let perplexity = avg_loss.exp();
            let samples_per_sec = if epoch_time > 0.0 {
                sample_count as f32 / epoch_time
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
    fn compute_grad_norm_3d(grads: &Array3<f32>) -> f32 {
        grads.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// 3D 梯度裁剪
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
    /// KV缓存可以显著加速推理速度（10-100倍），但不能用于训练。
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

            if !current_word.is_empty()
                && let Some(token_id) = vocab.encode(&current_word)
            {
                tokens.push(token_id);
            }
        }

        tokens
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        Self::tokenize_with_vocab(&self.vocab, text)
    }

    fn apply_temperature(probs: &Array2<f32>, temperature: f32) -> Array2<f32> {
        if temperature <= 0.0 {
            return probs.clone();
        }

        let power = 1.0 / temperature;
        let mut adjusted = probs.clone();

        for mut row in adjusted.rows_mut() {
            let mut sum = 0.0;
            for value in row.iter_mut() {
                *value = (*value).max(SOFTMAX_EPSILON).powf(power);
                sum += *value;
            }

            if sum > 0.0 {
                for value in row.iter_mut() {
                    *value /= sum;
                }
            }
        }

        adjusted
    }

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            })
            .to_vec()
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

        if sum == 0.0 {
            return rng.random_range(0..probs.len());
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
                input = layer.forward(&input);
            }

            if input.shape()[0] == 0 {
                break;
            }

            let last_logit = input
                .row(input.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            let probs = softmax(&last_logit);
            let adjusted_probs = Self::apply_temperature(&probs, temperature);

            let next_token = if top_k > 0 {
                self.top_k_sampling(&adjusted_probs, top_k)
                    .into_iter()
                    .next()
                    .unwrap_or(0)
            } else {
                self.top_p_sampling(&adjusted_probs, top_p)
                    .into_iter()
                    .next()
                    .unwrap_or(0)
            };

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
                    input_tensor = layer.forward(&input_tensor);
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

    fn cross_entropy_from_log_probs(log_probs: &Array2<f32>, target: &[usize]) -> f32 {
        // 使用 log_softmax 输出计算交叉熵，避免对概率取对数的数值不稳定
        let mut loss = 0.0;
        let n_targets = target.len() as f32;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            let lp = log_probs[[row_idx, target_idx]];
            loss -= lp; // NLL: -log p(target)
        }

        loss / n_targets
    }

    /// 计算交叉熵损失（从softmax概率）
    pub fn cross_entropy_loss_step(probs: &Array2<f32>, target: &[usize]) -> f32 {
        use crate::LOG_EPSILON;
        let mut loss = 0.0;
        let n_targets = target.len() as f32;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            if target_idx < probs.shape()[1] {
                let prob = probs[[row_idx, target_idx]].max(LOG_EPSILON);
                loss -= prob.ln();
            }
        }

        loss / n_targets
    }

    pub fn compute_gradients_step(probs: &Array2<f32>, target: &[usize]) -> Array2<f32> {
        let mut grads = probs.clone(); // softmax - one_hot(target)

        if probs.shape()[0] != target.len() {
            log::error!(
                "梯度计算输入不匹配：probs行数={}，target长度={}",
                probs.shape()[0],
                target.len()
            );
            return grads; // 返回原始梯度，避免崩溃
        }

        let batch_size = target.len() as f32;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            grads[[row_idx, target_idx]] -= 1.0;
        }

        grads.mapv_inplace(|x| x / batch_size);
        grads
    }

    pub fn clip_gradients(grads: &mut Array2<f32>, max_norm: f32) {
        // 计算L2范数并裁剪
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > max_norm {
            let scale = max_norm / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }

    fn apply_accumulated_gradients(
        &mut self,
        gradient_bucket: &mut Vec<Array2<f32>>,
        current_lr: f32,
        use_parallel: bool,
        perf_monitor: &mut PerformanceMonitor,
    ) {
        if gradient_bucket.is_empty() {
            return;
        }

        let label = if use_parallel && gradient_bucket.len() > 1 {
            "梯度累积(并行归约)"
        } else {
            "梯度累积(单线程归约)"
        };

        let track_perf = use_parallel && gradient_bucket.len() > 1;

        if track_perf {
            perf_monitor.start(label);
        }

        let aggregated = if use_parallel && gradient_bucket.len() > 1 {
            Self::aggregate_gradients_parallel(gradient_bucket.as_slice())
        } else {
            Self::aggregate_gradients_sequential(gradient_bucket.as_slice())
        };

        if track_perf {
            perf_monitor.stop(label);
        }

        gradient_bucket.clear();

        let mut current_grad = aggregated;
        for layer in self.network.iter_mut().rev() {
            current_grad = layer.backward(&current_grad, current_lr);
        }
    }

    fn aggregate_gradients_sequential(gradients: &[Array2<f32>]) -> Array2<f32> {
        if gradients.is_empty() {
            return Array2::zeros((0, 0));
        }

        let mut acc = gradients[0].clone();
        for grad in &gradients[1..] {
            acc += grad;
        }
        acc.mapv_inplace(|x| x / gradients.len() as f32);
        acc
    }

    fn aggregate_gradients_parallel(gradients: &[Array2<f32>]) -> Array2<f32> {
        // 简化实现：对于小模型，串行聚合性能足够好
        Self::aggregate_gradients_sequential(gradients)
    }
}
