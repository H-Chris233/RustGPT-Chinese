//!  训练性能优化模块
//!
//!  包含阶段1的快速优化:
//!  1.  数据预处理缓存
//!  2.  余弦退火学习率调度
//!  3.  早停机制
//!  4.  训练监控增强
//!  5.  检查点管理集成

use ndarray::{Array1, Array2};
use crate::utils::log_softmax;
use crate::llm::LLM;

impl LLM {
    ///  使用预tokenize的数据进行训练（性能优化版本，简单版）
    ///
    ///  这个方法接受已经tokenize的数据，避免重复tokenization
    ///  相比train方法,在500个epoch的训练中可以节省99.8%的tokenization时间
    ///  注意：这是简化版本，不带早停和检查点，仅用于快速测试
    pub fn train_with_cached_tokens_simple(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        epochs: usize,
        initial_lr: f32,
    ) {
        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        for epoch in 0..epochs {
            let decay_rate: f32 = 0.95;
            let decay_steps = 10.0;
            let current_lr = initial_lr * decay_rate.powf(epoch as f32 / decay_steps);

            let mut total_nll = 0.0;
            let mut total_tokens = 0usize;

            //  直接使用缓存的tokenized数据，无需重复tokenize
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                //  1.  Slice  input  and  targets
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                //  Forward  pass
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                // 前向传播：收集每层 ctx，供反向传播使用（避免缓存覆盖）
                let mut layer_ctxs: Vec<crate::llm::LayerContext> =
                    Vec::with_capacity(self.network.len());
                for layer in &mut self.network {
                    let (out, ctx) = layer.forward(&input);
                    layer_ctxs.push(ctx);
                    input = out;
                }

                let logits = input;
                let log_probs = log_softmax(&logits);

                //  Backward  pass
                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output =
                    match Self::compute_gradients_step(&probs, target_ids, pad_token_id) {
                        Ok(Some(grads)) => grads,
                        Ok(None) => continue,
                        Err(err) => {
                            log::error!("训练信号错误({err:?})，已跳过 optimizer step");
                            continue;
                        }
                    };

                let n_targets = target_ids
                    .iter()
                    .filter(|&&t| t != pad_token_id)
                    .count();
                if n_targets == 0 {
                    continue;
                }
                let loss_mean =
                    Self::cross_entropy_from_log_probs(&log_probs, target_ids, pad_token_id);
                total_nll += loss_mean * (n_targets as f32);
                total_tokens += n_targets;
                Self::clip_gradients(&mut grads_output, 5.0);

                for (layer, ctx) in self
                    .network
                    .iter_mut()
                    .rev()
                    .zip(layer_ctxs.iter().rev())
                {
                    grads_output = layer.backward(ctx, &grads_output, current_lr);
                }

            }

            println!(
                "Epoch  {}:  Loss  =  {:.4},  LR  =  {:.6}",
                epoch,
                if total_tokens > 0 {
                    total_nll / total_tokens as f32
                } else {
                    0.0
                },
                current_lr
            );
        }

        self.set_training_mode(false);
    }

    ///  改进的训练方法：使用余弦退火学习率 + 线性 Warmup
    pub fn train_with_cosine_lr(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        epochs: usize,
        initial_lr: f32,
        num_restarts: usize, //  推荐值:  2-3
    ) {
        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        for epoch in 0..epochs {
            //  🔥  使用余弦退火学习率 + Warmup
            let warmup_epochs = Self::recommend_warmup_epochs(epochs);
            let current_lr =
                Self::cosine_with_warmup_lr(initial_lr, epoch, epochs, num_restarts, warmup_epochs);

            let mut total_nll = 0.0;
            let mut total_tokens = 0usize;
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                let mut layer_ctxs: Vec<crate::llm::LayerContext> =
                    Vec::with_capacity(self.network.len());
                for layer in &mut self.network {
                    let (out, ctx) = layer.forward(&input);
                    layer_ctxs.push(ctx);
                    input = out;
                }

                let logits = input;
                let log_probs = log_softmax(&logits);

                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output =
                    match Self::compute_gradients_step(&probs, target_ids, pad_token_id) {
                        Ok(Some(grads)) => grads,
                        Ok(None) => continue,
                        Err(err) => {
                            log::error!("训练信号错误({err:?})，已跳过 optimizer step");
                            continue;
                        }
                    };

                let n_targets = target_ids
                    .iter()
                    .filter(|&&t| t != pad_token_id)
                    .count();
                if n_targets == 0 {
                    continue;
                }
                let loss_mean =
                    Self::cross_entropy_from_log_probs(&log_probs, target_ids, pad_token_id);
                total_nll += loss_mean * (n_targets as f32);
                total_tokens += n_targets;
                Self::clip_gradients(&mut grads_output, 5.0);

                for (layer, ctx) in self
                    .network
                    .iter_mut()
                    .rev()
                    .zip(layer_ctxs.iter().rev())
                {
                    grads_output = layer.backward(ctx, &grads_output, current_lr);
                }

            }

            //  每10个epoch打印一次，减少输出
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!(
                    "Epoch  {}:  Loss  =  {:.4},  LR  =  {:.6}",
                    epoch,
                    if total_tokens > 0 {
                        total_nll / total_tokens as f32
                    } else {
                        0.0
                    },
                    current_lr
                );
            }
        }

        self.set_training_mode(false);
    }

    ///  带早停的训练方法
    ///
    ///  #  参数
    ///  -  `patience`:  容忍多少个epoch  loss不改善（推荐30-50）
    ///
    ///  #  返回值
    ///  返回实际训练的epoch数
    pub fn train_with_early_stopping(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
    ) -> usize {
        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        let mut best_loss = f32::INFINITY;
        let mut counter = 0;
        let min_delta = 0.001f32;
        let mut best_epoch = 0;

        for epoch in 0..max_epochs {
            let warmup_epochs = Self::recommend_warmup_epochs(max_epochs);
            let current_lr =
                Self::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 2, warmup_epochs);

            let mut total_nll = 0.0;
            let mut total_tokens = 0usize;
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                let mut layer_ctxs: Vec<crate::llm::LayerContext> =
                    Vec::with_capacity(self.network.len());
                for layer in &mut self.network {
                    let (out, ctx) = layer.forward(&input);
                    layer_ctxs.push(ctx);
                    input = out;
                }

                let logits = input;
                let log_probs = log_softmax(&logits);

                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output =
                    match Self::compute_gradients_step(&probs, target_ids, pad_token_id) {
                        Ok(Some(grads)) => grads,
                        Ok(None) => continue,
                        Err(err) => {
                            log::error!("训练信号错误({err:?})，已跳过 optimizer step");
                            continue;
                        }
                    };

                let n_targets = target_ids
                    .iter()
                    .filter(|&&t| t != pad_token_id)
                    .count();
                if n_targets == 0 {
                    continue;
                }
                let loss_mean =
                    Self::cross_entropy_from_log_probs(&log_probs, target_ids, pad_token_id);
                total_nll += loss_mean * (n_targets as f32);
                total_tokens += n_targets;
                Self::clip_gradients(&mut grads_output, 5.0);

                for (layer, ctx) in self
                    .network
                    .iter_mut()
                    .rev()
                    .zip(layer_ctxs.iter().rev())
                {
                    grads_output = layer.backward(ctx, &grads_output, current_lr);
                }

            }

            let avg_loss = if total_tokens > 0 {
                total_nll / total_tokens as f32
            } else {
                0.0
            };

            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                println!(
                    "Epoch  {}:  Loss  =  {:.4},  LR  =  {:.6}",
                    epoch, avg_loss, current_lr
                );
            }

            //  🔥  检查早停条件
            if avg_loss < best_loss - min_delta {
                best_loss = avg_loss;
                best_epoch = epoch;
                counter = 0;
            } else {
                counter += 1;
                if counter >= patience {
                    println!("\n🛑  早停触发:");
                    println!("        •  最佳epoch:  {}", best_epoch);
                    println!("        •  最佳loss:  {:.4}", best_loss);
                    println!("        •  停止epoch:  {}", epoch);
                    println!("        •  节省时间:  {}  epochs\n", max_epochs - epoch);

                    self.set_training_mode(false);
                    return epoch + 1;
                }
            }
        }

        self.set_training_mode(false);
        max_epochs
    }

    ///  带完整监控的训练方法（结合早停、余弦学习率、详细统计）
    ///
    ///  这是完整的训练方法，使用预tokenized数据
    ///  注意：这个方法与llm.rs中的train_monitored不同，使用预tokenized数据避免重复tokenization
    pub fn train_monitored_tokenized(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
    ) -> usize {
        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        let mut best_loss = f32::INFINITY;
        let mut counter = 0;
        let min_delta = 0.001f32;
        let mut best_epoch = 0;
        let start_time = std::time::Instant::now();

        for epoch in 0..max_epochs {
            let epoch_start = std::time::Instant::now();
            let current_lr = Self::cosine_annealing_lr(initial_lr, epoch, max_epochs, 2);

            let mut total_nll = 0.0;
            let mut total_tokens = 0usize;
            let mut total_grad_norm = 0.0;
            let mut sample_count = 0;

            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                let mut layer_ctxs: Vec<crate::llm::LayerContext> =
                    Vec::with_capacity(self.network.len());
                for layer in &mut self.network {
                    let (out, ctx) = layer.forward(&input);
                    layer_ctxs.push(ctx);
                    input = out;
                }

                let logits = input;
                let log_probs = log_softmax(&logits);

                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output =
                    match Self::compute_gradients_step(&probs, target_ids, pad_token_id) {
                        Ok(Some(grads)) => grads,
                        Ok(None) => continue,
                        Err(err) => {
                            log::error!("训练信号错误({err:?})，已跳过 optimizer step");
                            continue;
                        }
                    };

                let n_targets = target_ids
                    .iter()
                    .filter(|&&t| t != pad_token_id)
                    .count();
                if n_targets == 0 {
                    continue;
                }
                let loss_mean =
                    Self::cross_entropy_from_log_probs(&log_probs, target_ids, pad_token_id);
                total_nll += loss_mean * (n_targets as f32);
                total_tokens += n_targets;

                //  记录梯度范数
                total_grad_norm += Self::compute_grad_norm(&grads_output);

                Self::clip_gradients(&mut grads_output, 5.0);

                for (layer, ctx) in self
                    .network
                    .iter_mut()
                    .rev()
                    .zip(layer_ctxs.iter().rev())
                {
                    grads_output = layer.backward(ctx, &grads_output, current_lr);
                }

                sample_count += 1;
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            if sample_count == 0 {
                log::error!(
                    "train_with_early_stopping: 没有有效训练样本（所有序列长度 < 2），无法继续训练。epoch={}",
                    epoch
                );
                self.set_training_mode(false);
                return epoch;
            }

            let avg_loss = if total_tokens > 0 {
                total_nll / total_tokens as f32
            } else {
                0.0
            };
            let avg_grad_norm = total_grad_norm / sample_count as f32;
            let perplexity = avg_loss.exp();
            let samples_per_sec = if epoch_time > 0.0 {
                sample_count as f32 / epoch_time
            } else {
                0.0
            };

            //  📊  丰富的训练信息
            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                let progress = (epoch + 1) as f32 / max_epochs as f32 * 100.0;
                let elapsed = start_time.elapsed().as_secs();
                let eta =
                    (elapsed as f32 / (epoch + 1) as f32 * (max_epochs - epoch - 1) as f32) as u64;

                println!(
                    "[{:3}/{:3}]  ({:.1}%)  Loss:  {:.4}  |  PPL:  {:.2}  |  LR:  {:.6}  |  Grad:  {:.4}  |  Speed:  {:.1}  samples/s  |  ETA:  {}s",
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

            //  🔥  检查早停条件
            if avg_loss < best_loss - min_delta {
                best_loss = avg_loss;
                best_epoch = epoch;
                counter = 0;
            } else {
                counter += 1;
                if counter >= patience {
                    println!("\n🛑  早停触发:");
                    println!("        •  最佳epoch:  {}", best_epoch);
                    println!("        •  最佳loss:  {:.4}", best_loss);
                    println!("        •  停止epoch:  {}", epoch);
                    println!("        •  节省时间:  {}  epochs\n", max_epochs - epoch);

                    self.set_training_mode(false);
                    return epoch + 1;
                }
            }
        }

        self.set_training_mode(false);
        max_epochs
    }

    ///  带检查点管理的训练方法
    ///
    ///  这是最完整的训练方法，集成了早停、余弦学习率、检查点管理
    ///
    ///  #  参数
    ///  -  `checkpoint_manager`:  检查点管理器（可选）
    ///  -  `phase`:  训练阶段标识（如"pretraining", "instruction_tuning"）
    ///  -  `resume_epoch`:  从哪个epoch开始（用于resume训练）
    ///
    ///  #  返回值
    ///  返回实际训练的epoch数
    pub fn train_with_checkpointing(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
        mut checkpoint_manager: Option<&mut crate::checkpoint_manager::CheckpointManager>,
        phase: &str,
        resume_epoch: usize,
    ) -> usize {
        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        let mut best_loss = if let Some(ref manager) = checkpoint_manager {
            manager.get_best_loss()
        } else {
            f32::INFINITY
        };
        let mut counter = 0;
        // 与 `train_monitored()` 保持一致：避免 resume 训练与主训练的早停判据不一致。
        let min_delta = 0.01f32;
        let mut best_epoch = resume_epoch;
        let start_time = std::time::Instant::now();

        for epoch in resume_epoch..max_epochs {
            let epoch_start = std::time::Instant::now();
            // 避免同一轮中“最佳保存 + 周期保存”重复写盘（BestAndLast 策略下尤为明显）。
            let mut saved_checkpoint_this_epoch = false;
            // 与 `train_monitored()` 保持一致：余弦退火 + warmup（禁用重启）。
            let warmup_epochs = Self::recommend_warmup_epochs(max_epochs);
            let current_lr =
                Self::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);

            let mut total_nll = 0.0;
            let mut total_tokens = 0usize;
            let mut total_grad_norm = 0.0;
            let mut sample_count = 0;

            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                let mut layer_ctxs: Vec<crate::llm::LayerContext> =
                    Vec::with_capacity(self.network.len());
                for layer in &mut self.network {
                    let (out, ctx) = layer.forward(&input);
                    layer_ctxs.push(ctx);
                    input = out;
                }

                let logits = input;
                let log_probs = log_softmax(&logits);

                let probs = log_probs.mapv(|x| x.exp());
                let mut grads_output =
                    match Self::compute_gradients_step(&probs, target_ids, pad_token_id) {
                        Ok(Some(grads)) => grads,
                        Ok(None) => continue,
                        Err(err) => {
                            log::error!("训练信号错误({err:?})，已跳过 optimizer step");
                            continue;
                        }
                    };

                let n_targets = target_ids
                    .iter()
                    .filter(|&&t| t != pad_token_id)
                    .count();
                if n_targets == 0 {
                    continue;
                }
                let loss_mean =
                    Self::cross_entropy_from_log_probs(&log_probs, target_ids, pad_token_id);
                total_nll += loss_mean * (n_targets as f32);
                total_tokens += n_targets;

                //  记录梯度范数
                total_grad_norm += Self::compute_grad_norm(&grads_output);

                // 与 `train_monitored()` 保持一致：使用更严格的裁剪阈值提升稳定性。
                Self::clip_gradients(&mut grads_output, 1.0);

                for (layer, ctx) in self
                    .network
                    .iter_mut()
                    .rev()
                    .zip(layer_ctxs.iter().rev())
                {
                    grads_output = layer.backward(ctx, &grads_output, current_lr);
                }

                sample_count += 1;
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            if sample_count == 0 {
                log::error!(
                    "train_with_checkpointing: 没有有效训练样本（所有序列长度 < 2），无法继续训练。epoch={}",
                    epoch
                );
                self.set_training_mode(false);
                return epoch;
            }

            let avg_loss = if total_tokens > 0 {
                total_nll / total_tokens as f32
            } else {
                0.0
            };
            let avg_grad_norm = total_grad_norm / sample_count as f32;
            let perplexity = avg_loss.exp();
            let samples_per_sec = if epoch_time > 0.0 {
                sample_count as f32 / epoch_time
            } else {
                0.0
            };

            //  📊  丰富的训练信息
            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                let progress = (epoch + 1) as f32 / max_epochs as f32 * 100.0;
                let elapsed = start_time.elapsed().as_secs();
                let eta = (elapsed as f32 / (epoch - resume_epoch + 1) as f32
                    * (max_epochs - epoch - 1) as f32) as u64;

                println!(
                    "[{:3}/{:3}]  ({:.1}%)  Loss:  {:.4}  |  PPL:  {:.2}  |  LR:  {:.6}  |  Grad:  {:.4}  |  Speed:  {:.1}  samples/s  |  ETA:  {}s",
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

            //  🔥  检查早停条件和保存检查点
            if avg_loss < best_loss - min_delta {
                best_loss = avg_loss;
                best_epoch = epoch;
                counter = 0;

                //  保存最佳检查点
                if let Some(ref mut manager) = checkpoint_manager {
                    let metadata = crate::checkpoint_manager::CheckpointMetadata {
                        epoch,
                        loss: avg_loss,
                        learning_rate: current_lr,
                        timestamp: chrono::Local::now()
                            .format("%Y-%m-%d  %H:%M:%S")
                            .to_string(),
                        phase: phase.to_string(),
                    };

                    match manager.save_checkpoint(self, metadata) {
                        Ok(path) => {
                            // `save_checkpoint()` 在“本轮不需要保存”时会返回空 PathBuf。
                            // 因此这里要用路径是否为空来判断是否真的发生了写盘。
                            if !path.as_os_str().is_empty() {
                                saved_checkpoint_this_epoch = true;
                            }
                        }
                        Err(e) => {
                            log::warn!("保存检查点失败:  {}", e);
                        }
                    }
                }
            } else {
                counter += 1;
                if counter >= patience {
                    println!("\n🛑  早停触发:");
                    println!("        •  最佳epoch:  {}", best_epoch);
                    println!("        •  最佳loss:  {:.4}", best_loss);
                    println!("        •  停止epoch:  {}", epoch);
                    println!("        •  节省时间:  {}  epochs\n", max_epochs - epoch);

                    //  尝试加载最佳检查点
                    if let Some(ref manager) = checkpoint_manager {
                        if let Some(best_checkpoint_path) = manager.get_best_checkpoint() {
                            println!("🔄  加载最佳检查点:  {}", best_checkpoint_path.display());
                            match crate::checkpoint_manager::CheckpointManager::load_checkpoint(
                                best_checkpoint_path,
                            ) {
                                Ok((best_llm, _metadata)) => {
                                    //  复制最佳模型的参数到当前模型
                                    self.network = best_llm.network;
                                    println!("✅  已回滚到最佳epoch的模型参数");
                                }
                                Err(e) => {
                                    log::warn!("加载最佳检查点失败:  {}", e);
                                }
                            }
                        }
                    }

                    self.set_training_mode(false);
                    return epoch + 1;
                }
            }

            //  周期性保存检查点（如果配置了）
            //  注意：对于BestAndLast策略，即使loss不是best也要保存last checkpoint
            if let Some(ref mut manager) = checkpoint_manager {
                if !saved_checkpoint_this_epoch && manager.should_save(epoch, avg_loss) {
                    let metadata = crate::checkpoint_manager::CheckpointMetadata {
                        epoch,
                        loss: avg_loss,
                        learning_rate: current_lr,
                        timestamp: chrono::Local::now()
                            .format("%Y-%m-%d  %H:%M:%S")
                            .to_string(),
                        phase: phase.to_string(),
                    };

                    if let Err(e) = manager.save_checkpoint(self, metadata) {
                        log::warn!("保存检查点失败:  {}", e);
                    }
                }
            }
        }

        self.set_training_mode(false);
        max_epochs
    }
}
