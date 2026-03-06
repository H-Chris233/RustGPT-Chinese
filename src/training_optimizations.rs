//!  训练性能优化模块
//!
//!  保留训练阶段的进阶入口。
//!
//!  当前对外仅保留：
//!  1.  带检查点管理的训练接口
//!
//!  其余历史训练变体已删除，以减少教学项目的 API 表面积。

use crate::llm::LLM;

impl LLM {
    ///  带检查点管理的训练方法
    ///
    ///  该接口组合了早停、学习率调度以及检查点保存/恢复能力。
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

                let Some(mut step) = self.prepare_training_step(input_ids, target_ids, pad_token_id)
                else {
                    continue;
                };

                total_nll += step.loss_mean * (step.n_targets as f32);
                total_tokens += step.n_targets;

                //  记录梯度范数
                total_grad_norm += Self::compute_grad_norm(&step.grads_output);

                // 与 `train_monitored()` 保持一致：使用更严格的裁剪阈值提升稳定性。
                Self::clip_gradients(&mut step.grads_output, 1.0);
                self.backward_with_ctx(&step.layer_ctxs, &step.grads_output, current_lr);

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
