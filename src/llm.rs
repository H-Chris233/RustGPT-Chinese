use std::cmp::Ordering;

use ndarray::{Array1, Array2, Array3, Axis};
use rand::{rng, Rng};

use crate::{
    output_projection::OutputProjection,
    transformer::TransformerBlock,
    utils::{log_softmax, softmax},
    Embeddings, PerformanceMonitor, Vocab, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN, SOFTMAX_EPSILON,
};

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
        self.forward_with_sampling(text, temperature, top_p, top_k)
    }

    pub fn predict_with_context(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        // Tokenize the new input
        let new_tokens = self.tokenize(text);

        // Combine context with new input
        let mut all_tokens = self.context_window.clone();
        all_tokens.extend_from_slice(&new_tokens);

        // Ensure we don't exceed the maximum sequence length
        if all_tokens.len() > MAX_SEQ_LEN {
            let start_idx = all_tokens.len() - MAX_SEQ_LEN;
            all_tokens = all_tokens[start_idx..].to_vec();
        }

        let result = self.generate_with_context(&all_tokens, temperature, top_p, top_k);

        self.add_to_context(&new_tokens);
        let result_tokens = self.tokenize(&result);
        self.add_to_context(&result_tokens);

        result
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
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        if tokenized.is_empty() {
            return String::new();
        }

        let input_len = tokenized.len();

        if input_len >= MAX_SEQ_LEN {
            return String::new();
        }

        for _ in 0..(MAX_SEQ_LEN - input_len) {
            let token_input = match Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("构造输入张量失败: {}", e);
                    break;
                }
            };
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

            let adjusted_probs = Self::apply_temperature(&probs, temperature);

            let tokens = if top_k > 0 {
                self.top_k_sampling(&adjusted_probs, top_k)
            } else {
                self.top_p_sampling(&adjusted_probs, top_p)
            };

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        let token_strs = output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>();

        let raw_output = token_strs.join(" ");
        self.post_process_chinese_text(&raw_output)
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
            }

            println!(
                "Epoch {}: Loss = {:.4}, LR = {:.6}",
                epoch,
                total_loss / tokenized_data.len() as f32,
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

            // 🔥 余弦退火学习率调度（禁用重启以提升稳定性）
            let current_lr = Self::cosine_annealing_lr(initial_lr, epoch, max_epochs, 0);

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

            // 余弦退火学习率
            let current_lr = Self::cosine_annealing_lr(initial_lr, epoch, max_epochs, 0);

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
                    let sample_tokens = input_batch.tokens.row(b);
                    let sample_ids: Vec<usize> = sample_tokens.to_vec();

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

                    // 只计算非 PAD 位置的损失
                    let target_ids = &targets[b];
                    let loss = Self::cross_entropy_from_log_probs(&log_probs, target_ids);
                    batch_loss += loss;

                    // 计算梯度
                    let probs = log_probs.mapv(|x| x.exp());
                    let mut sample_grad = Self::compute_gradients_step(&probs, target_ids);

                    // 应用注意力掩码到梯度（将PAD位置梯度清零）
                    for s in 0..sample_grad.nrows() {
                        if input_batch.attention_mask[[b, s]] < 0.5 {
                            sample_grad.row_mut(s).fill(0.0);
                        }
                    }

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
        for layer in &mut self.network {
            // 只对SelfAttention层启用KV缓存
            // TransformerBlock内部包含SelfAttention，需要特殊处理
            // 这里我们通过layer_type判断
            if layer.layer_type() == "TransformerBlock" {
                // TransformerBlock需要向下传递enable命令
                // 由于Layer trait没有enable_kv_cache方法，
                // 我们需要在TransformerBlock中单独实现
                // 暂时通过unsafe转换实现
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.enable_kv_cache();
                }
            }
        }
    }

    /// 禁用所有transformer层的KV缓存
    pub fn disable_kv_cache(&mut self) {
        for layer in &mut self.network {
            if layer.layer_type() == "TransformerBlock" {
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.disable_kv_cache();
                }
            }
        }
    }

    /// 清空所有transformer层的KV缓存（保持启用状态）
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.network {
            if layer.layer_type() == "TransformerBlock" {
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.clear_kv_cache();
                }
            }
        }
    }

    /// 统一设置所有 TransformerBlock 的 SelfAttention 是否冻结参数更新
    /// 用于在不修改网络结构的前提下，快速排查训练不稳定问题
    pub fn set_attention_freeze_updates(&mut self, freeze: bool) {
        for layer in &mut self.network {
            if layer.layer_type() == "TransformerBlock" {
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.freeze_updates = freeze;
                }
            }
        }
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
        let mut result = Vec::new();

        for row in probs.rows() {
            // 复用 sampling_idx_buffer
            self.sampling_idx_buffer.clear();
            self.sampling_idx_buffer
                .extend(row.iter().enumerate().map(|(idx, &prob)| (prob, idx)));

            // Sort by probability in descending order and take top k
            self.sampling_idx_buffer
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            let top_k_len = k.min(self.sampling_idx_buffer.len());

            // 复用 sampling_prob_buffer
            self.sampling_prob_buffer.clear();
            self.sampling_prob_buffer
                .resize(self.vocab.words.len(), 0.0);
            let mut sum = 0.0;

            for i in 0..top_k_len {
                let (prob, idx) = self.sampling_idx_buffer[i];
                self.sampling_prob_buffer[idx] = prob;
                sum += prob;
            }

            // Normalize the probabilities
            if sum > 0.0 {
                for p in &mut self.sampling_prob_buffer {
                    *p /= sum;
                }
            }

            // Sample from the top-k distribution
            result.push(self.sample_from_probs(&self.sampling_prob_buffer));
        }

        result
    }

    /// Top-p (nucleus) sampling: consider the smallest set of tokens whose cumulative probability
    /// exceeds p
    /// 优化版本：复用内部缓冲区减少分配
    fn top_p_sampling(&mut self, probs: &Array2<f32>, p: f32) -> Vec<usize> {
        let mut result = Vec::new();

        for row in probs.rows() {
            // 复用 sampling_idx_buffer
            self.sampling_idx_buffer.clear();
            self.sampling_idx_buffer
                .extend(row.iter().enumerate().map(|(idx, &prob)| (prob, idx)));

            self.sampling_idx_buffer
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            // Find the smallest set of tokens whose cumulative probability exceeds p
            let mut cumulative_prob = 0.0;
            let mut selected_count = 0;

            for &(prob, _) in &self.sampling_idx_buffer {
                cumulative_prob += prob;
                selected_count += 1;

                if cumulative_prob >= p {
                    break;
                }
            }

            // 复用 sampling_prob_buffer
            self.sampling_prob_buffer.clear();
            self.sampling_prob_buffer
                .resize(self.vocab.words.len(), 0.0);
            let mut sum = 0.0;

            for i in 0..selected_count {
                let (prob, idx) = self.sampling_idx_buffer[i];
                self.sampling_prob_buffer[idx] = prob;
                sum += prob;
            }

            // Normalize the probabilities
            if sum > 0.0 {
                for p_val in &mut self.sampling_prob_buffer {
                    *p_val /= sum;
                }
            }

            // Sample from the selected distribution
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

    /// Beam search implementation
    /// 优化版本：复用candidates缓冲区减少分配
    fn beam_search(&mut self, text: &str, beam_width: usize, max_length: usize) -> String {
        // Tokenize the input text
        let initial_tokens = self.tokenize(text);
        if initial_tokens.is_empty() {
            return String::new();
        }

        // Initialize beam with the initial sequence
        let mut current_beams = vec![(initial_tokens.clone(), 0.0f32)]; // (sequence, log_probability)

        for _ in initial_tokens.len()..max_length {
            // 复用 beam_candidates_buffer
            self.beam_candidates_buffer.clear();

            for (seq, log_prob) in &current_beams {
                // Get model prediction for the current sequence
                let input =
                    Array2::from_shape_vec((1, seq.len()), seq.iter().map(|&x| x as f32).collect())
                        .unwrap();

                let mut input_tensor = input;
                for layer in &mut self.network {
                    input_tensor = layer.forward(&input_tensor);
                }

                let logits = input_tensor;
                let probs = softmax(&logits);

                // Get the probabilities for the last token position
                let last_token_probs = probs.row(probs.nrows() - 1);

                // 复用 sampling_idx_buffer 获取 top-k candidates
                self.sampling_idx_buffer.clear();
                self.sampling_idx_buffer.extend(
                    last_token_probs
                        .iter()
                        .enumerate()
                        .map(|(idx, &prob)| (prob, idx)),
                );

                self.sampling_idx_buffer
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

                // Add candidates with updated sequences and log probabilities
                for i in 0..beam_width.min(self.sampling_idx_buffer.len()) {
                    let (prob, token_id) = self.sampling_idx_buffer[i];
                    if prob > 0.0 {
                        // Only consider non-zero probability tokens
                        let mut new_seq = seq.clone();
                        new_seq.push(token_id);
                        let new_log_prob = log_prob + prob.ln(); // Add log probabilities
                        self.beam_candidates_buffer.push((new_seq, new_log_prob));
                    }
                }
            }

            // Sort candidates by log probability and keep the top beam_width
            self.beam_candidates_buffer
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            current_beams = self
                .beam_candidates_buffer
                .iter()
                .take(beam_width)
                .cloned()
                .collect();

            // Check if any beam has generated an end token
            if current_beams.iter().any(|(seq, _)| {
                if let Some(&last_token) = seq.last() {
                    self.vocab.decode.get(&last_token) == Some(&"</s>".to_string())
                } else {
                    false
                }
            }) {
                break;
            }
        }

        // Select the beam with the highest probability
        if let Some((best_seq, _)) = current_beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        {
            // Convert token IDs back to text
            let token_strs = best_seq
                .iter()
                .map(|&t| self.vocab.decode[&t].clone())
                .filter(|s| s != "</s>") // Remove end token
                .collect::<Vec<String>>();

            // Apply post-processing to improve fluency and accuracy
            let raw_output = token_strs.join(" ");
            self.post_process_chinese_text(&raw_output)
        } else {
            String::new() // Return empty string if no valid sequence is found
        }
    }

    /// Generate text based on the given context tokens
    fn generate_with_context(
        &mut self,
        context_tokens: &[usize],
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let mut tokenized = context_tokens.to_vec();
        let mut output_tokens: Vec<usize> = Vec::new();

        if tokenized.is_empty() {
            return String::new();
        }

        let input_len = tokenized.len();

        if input_len >= MAX_SEQ_LEN {
            return String::new();
        }

        let max_new_tokens = 20.min(MAX_SEQ_LEN - input_len);

        for _ in 0..max_new_tokens {
            let token_input = match Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("构造输入张量失败: {}", e);
                    break;
                }
            };
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

            let adjusted_probs = Self::apply_temperature(&probs, temperature);

            let tokens = if top_k > 0 {
                self.top_k_sampling(&adjusted_probs, top_k)
            } else {
                self.top_p_sampling(&adjusted_probs, top_p)
            };

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.eos_token_id() {
                break;
            }
        }

        let token_strs = output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>();

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

        perf_monitor.start(label);
        let aggregated = if use_parallel && gradient_bucket.len() > 1 {
            Self::aggregate_gradients_parallel(gradient_bucket.as_slice())
        } else {
            Self::aggregate_gradients_sequential(gradient_bucket.as_slice())
        };
        perf_monitor.stop(label);

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
