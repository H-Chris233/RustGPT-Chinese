//!  训练性能优化模块
//!
//!  保留训练阶段的进阶入口。
//!
//!  注意：这里描述的是 **本模块** 当前保留的训练入口，
//!  不是整个 `LLM` 类型对外公开的全部训练 API。
//!
//!  当前本模块仅保留：
//!  1.  带检查点管理的训练接口
//!
//!  其余公开训练入口（如 `train(...)`、`train_monitored(...)`、
//!  `train_bucketed_sequential(...)`）定义在 `src/llm.rs` 中。
//!
//!  推荐阅读顺序：
//!  `train_with_checkpointing()`
//!    -> `run_checkpoint_epoch()`
//!    -> `prepare_training_step()`
//!    -> `backward_with_ctx()`
//!
//!  其余历史训练变体已删除，以减少教学项目的 API 表面积。

use crate::checkpoint_manager::{CheckpointManager, CheckpointMetadata};
use crate::llm::LLM;

/// 单个 checkpoint 训练 epoch 的汇总指标。
///
/// 这是 `train_with_checkpointing()` 的内部数据载体：
/// - `run_checkpoint_epoch()` 只负责跑完一轮训练并汇总指标；
/// - 外层函数再决定打印、早停、保存检查点等控制流。
struct CheckpointEpochMetrics {
    avg_loss: f32,
    avg_grad_norm: f32,
    perplexity: f32,
    current_lr: f32,
    sample_count: usize,
    epoch_time_secs: f32,
}

impl LLM {
    /// 判定当前 loss 在“真实最佳模型”与“早停阈值”两个维度上的进展情况。
    ///
    /// 返回值：
    /// - 第 1 项：是否刷新了历史真实 best（只要 loss 更低就算）
    /// - 第 2 项：是否达到 early stopping 所需的“显著改善”（需要超过 `min_delta`）
    fn classify_checkpoint_progress(
        true_best_loss: f32,
        early_stop_best_loss: f32,
        current_loss: f32,
        min_delta: f32,
    ) -> (bool, bool) {
        let is_true_best = current_loss < true_best_loss;
        let is_significant_improvement = current_loss < early_stop_best_loss - min_delta;
        (is_true_best, is_significant_improvement)
    }

    /// 执行一个 epoch 的 checkpoint 训练主循环，并返回该轮聚合指标。
    ///
    /// 教学说明：
    /// - 这里故意只做“单轮训练 + 指标汇总”，不掺杂早停/检查点策略；
    /// - loss 口径保持为 token-weighted mean；
    /// - 若整轮都没有有效样本，会返回 `None`，由外层决定如何提前结束。
    fn run_checkpoint_epoch(
        &mut self,
        tokenized_data: &[Vec<usize>],
        pad_token_id: usize,
        epoch: usize,
        max_epochs: usize,
        initial_lr: f32,
    ) -> Option<CheckpointEpochMetrics> {
        let epoch_start = std::time::Instant::now();
        // 与 `train_monitored()` 保持一致：余弦退火 + warmup（禁用重启）。
        let warmup_epochs = Self::recommend_warmup_epochs(max_epochs);
        let current_lr =
            Self::cosine_with_warmup_lr(initial_lr, epoch, max_epochs, 0, warmup_epochs);

        let mut total_nll = 0.0;
        let mut total_tokens = 0usize;
        let mut total_grad_norm = 0.0;
        let mut sample_count = 0usize;

        for training_row in tokenized_data {
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

        if sample_count == 0 {
            return None;
        }

        let avg_loss = if total_tokens > 0 {
            total_nll / total_tokens as f32
        } else {
            0.0
        };

        Some(CheckpointEpochMetrics {
            avg_loss,
            avg_grad_norm: total_grad_norm / sample_count as f32,
            perplexity: avg_loss.exp(),
            current_lr,
            sample_count,
            epoch_time_secs: epoch_start.elapsed().as_secs_f32(),
        })
    }

    /// 按当前状态尝试保存检查点。
    ///
    /// 返回值表示“本轮是否真的发生了写盘”：
    /// - `true`：至少写入了一个检查点文件；
    /// - `false`：没有管理器、策略判定无需保存，或保存失败。
    ///
    /// 说明：`CheckpointManager::save_checkpoint()` 在“不需要保存”时会返回空 `PathBuf`，
    /// 因此这里统一把“空路径”解释为“未写盘”。
    fn maybe_save_checkpoint(
        &self,
        checkpoint_manager: Option<&mut CheckpointManager>,
        epoch: usize,
        avg_loss: f32,
        current_lr: f32,
        phase: &str,
    ) -> bool {
        let Some(manager) = checkpoint_manager else {
            return false;
        };

        let metadata = CheckpointMetadata {
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
                !path.as_os_str().is_empty()
            }
            Err(e) => {
                log::warn!("保存检查点失败:  {}", e);
                false
            }
        }
    }

    /// 在早停触发后尝试回滚到最佳检查点。
    ///
    /// 这里刻意只恢复 `network` 参数，不整体替换整个 `LLM`：
    /// - 这样可以保留当前实例上的词表、运行期 buffer 与外围状态；
    /// - 也更符合“只回滚模型权重”的教学表达。
    fn maybe_restore_best_checkpoint(&mut self, checkpoint_manager: Option<&CheckpointManager>) {
        let Some(manager) = checkpoint_manager else {
            return;
        };
        let Some(best_checkpoint_path) = manager.get_best_checkpoint() else {
            return;
        };

        println!("🔄  加载最佳检查点:  {}", best_checkpoint_path.display());
        match CheckpointManager::load_checkpoint(best_checkpoint_path) {
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

    ///  带检查点管理的训练方法
    ///
    ///  该接口组合了早停、学习率调度以及检查点保存/恢复能力。
    ///
    ///  教学主线：
    ///  1. `run_checkpoint_epoch()` 负责单轮训练；
    ///  2. 外层函数负责打印、早停判定与 checkpoint 编排；
    ///  3. 指标口径统一使用 token-weighted mean loss。
    ///
    ///  #  参数
    ///  -  `checkpoint_manager`:  检查点管理器（可选）
    ///  -  `phase`:  训练阶段标识（如"pretraining", "instruction_tuning"）
    ///  -  `resume_epoch`:  从哪个epoch开始（用于resume训练）
    ///
    ///  #  返回值
    ///  返回当前训练停止时的绝对 epoch 坐标：
    ///  - 正常完成时返回 `max_epochs`；
    ///  - 早停时返回 `epoch + 1`；
    ///  - 若在 resume 训练中立即遇到零样本，则返回传入的 `resume_epoch`。
    pub fn train_with_checkpointing(
        &mut self,
        tokenized_data: Vec<Vec<usize>>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
        mut checkpoint_manager: Option<&mut CheckpointManager>,
        phase: &str,
        resume_epoch: usize,
    ) -> usize {
        self.set_training_mode(true);
        let pad_token_id = self.vocab.pad_token_id();

        let mut true_best_loss = if let Some(ref manager) = checkpoint_manager {
            manager.get_best_loss()
        } else {
            f32::INFINITY
        };
        let mut early_stop_best_loss = true_best_loss;
        let mut counter = 0;
        // 与 `train_monitored()` 保持一致：避免 resume 训练与主训练的早停判据不一致。
        let min_delta = 0.01f32;
        let mut best_epoch = if let Some(ref manager) = checkpoint_manager {
            manager.get_best_epoch()
        } else {
            resume_epoch
        };
        let start_time = std::time::Instant::now();

        for epoch in resume_epoch..max_epochs {
            let metrics = match self.run_checkpoint_epoch(
                &tokenized_data,
                pad_token_id,
                epoch,
                max_epochs,
                initial_lr,
            ) {
                Some(metrics) => metrics,
                None => {
                    log::error!(
                        "train_with_checkpointing: 没有有效训练样本（所有序列长度 < 2），无法继续训练。epoch={}",
                        epoch
                    );
                    self.set_training_mode(false);
                    return epoch;
                }
            };
            // 避免同一轮中“最佳保存 + 周期保存”重复写盘（BestAndLast 策略下尤为明显）。
            let mut saved_checkpoint_this_epoch = false;

            //  📊  丰富的训练信息
            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                let progress = (epoch + 1) as f32 / max_epochs as f32 * 100.0;
                let elapsed = start_time.elapsed().as_secs();
                let eta = (elapsed as f32 / (epoch - resume_epoch + 1) as f32
                    * (max_epochs - epoch - 1) as f32) as u64;
                let samples_per_sec = if metrics.epoch_time_secs > 0.0 {
                    metrics.sample_count as f32 / metrics.epoch_time_secs
                } else {
                    0.0
                };

                println!(
                    "[{:3}/{:3}]  ({:.1}%)  Loss:  {:.4}  |  PPL:  {:.2}  |  LR:  {:.6}  |  Grad:  {:.4}  |  Speed:  {:.1}  samples/s  |  ETA:  {}s",
                    epoch + 1,
                    max_epochs,
                    progress,
                    metrics.avg_loss,
                    metrics.perplexity,
                    metrics.current_lr,
                    metrics.avg_grad_norm,
                    samples_per_sec,
                    eta
                );
            }

            //  🔥  检查早停条件和保存检查点
            let (is_true_best, is_significant_improvement) = Self::classify_checkpoint_progress(
                true_best_loss,
                early_stop_best_loss,
                metrics.avg_loss,
                min_delta,
            );

            // 真实 best：只要 loss 更低，就更新最佳模型与 checkpoint。
            if is_true_best {
                true_best_loss = metrics.avg_loss;
                best_epoch = epoch;
                saved_checkpoint_this_epoch = self.maybe_save_checkpoint(
                    checkpoint_manager.as_mut().map(|manager| &mut **manager),
                    epoch,
                    metrics.avg_loss,
                    metrics.current_lr,
                    phase,
                );
            }

            // 早停判据：只有“超过 min_delta 的显著改善”才重置 patience。
            if is_significant_improvement {
                early_stop_best_loss = metrics.avg_loss;
                counter = 0;
            } else {
                counter += 1;
                if counter >= patience {
                    println!("\n🛑  早停触发:");
                    println!("        •  最佳epoch:  {}", best_epoch);
                    println!("        •  最佳loss:  {:.4}", true_best_loss);
                    println!("        •  停止epoch:  {}", epoch);
                    println!("        •  节省时间:  {}  epochs\n", max_epochs - epoch);

                    self.maybe_restore_best_checkpoint(
                        checkpoint_manager.as_ref().map(|manager| &**manager),
                    );

                    self.set_training_mode(false);
                    return epoch + 1;
                }
            }

            //  周期性保存检查点（如果配置了）
            //  注意：对于BestAndLast策略，即使loss不是best也要保存last checkpoint
            let should_save_periodic = checkpoint_manager
                .as_ref()
                .map(|manager| manager.should_save(epoch, metrics.avg_loss))
                .unwrap_or(false);
            if !saved_checkpoint_this_epoch && should_save_periodic {
                self.maybe_save_checkpoint(
                    checkpoint_manager.as_mut().map(|manager| &mut **manager),
                    epoch,
                    metrics.avg_loss,
                    metrics.current_lr,
                    phase,
                );
            }
        }

        self.set_training_mode(false);
        max_epochs
    }
}

#[cfg(test)]
mod tests {
    use super::LLM;

    #[test]
    fn checkpoint_progress_decouples_true_best_from_min_delta() {
        let (is_true_best, is_significant_improvement) =
            LLM::classify_checkpoint_progress(0.500, 0.500, 0.495, 0.01);

        assert!(is_true_best, "loss 只要更低，就应视为真实 best");
        assert!(
            !is_significant_improvement,
            "小于 min_delta 的改善不应重置 early stopping patience"
        );
    }

    #[test]
    fn checkpoint_progress_marks_large_improvement_for_both_dimensions() {
        let (is_true_best, is_significant_improvement) =
            LLM::classify_checkpoint_progress(0.500, 0.500, 0.480, 0.01);

        assert!(is_true_best);
        assert!(is_significant_improvement);
    }
}
