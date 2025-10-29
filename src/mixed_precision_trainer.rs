//! # 混合精度训练器
//!
//! 为 LLM 提供混合精度训练支持，集成损失缩放、精度转换和自动回退。
//!
//! ## 核心功能
//!
//! 1. **训练循环包装**：在标准训练流程中注入混合精度逻辑
//! 2. **损失缩放**：自动管理动态损失缩放
//! 3. **梯度处理**：Unscale、溢出检测、裁剪
//! 4. **自动回退**：检测持续不稳定时切换到 FP32
//! 5. **性能监控**：记录精度、缩放因子、溢出率
//!
//! ## 使用示例
//!
//! ```rust
//! use llm::{LLM, MixedPrecisionConfig, MixedPrecisionTrainer};
//!
//! let mut llm = LLM::default();
//! let config = MixedPrecisionConfig::fp16();
//! let mut trainer = MixedPrecisionTrainer::new(config);
//!
//! trainer.train_monitored(
//!     &mut llm,
//!     tokenized_data,
//!     100,  // epochs
//!     0.001,  // initial_lr
//!     30,  // patience
//! );
//! ```

use log::{info, warn};
use ndarray::Array2;

use crate::{
    llm::LLM,
    loss_scaler::LossScaler,
    mixed_precision::{MixedPrecisionConfig, PrecisionType},
    precision_convert::{has_invalid_values, round_trip_inplace},
    utils::softmax,
};

/// **混合精度训练器**
///
/// 封装 LLM 的训练逻辑，添加混合精度支持。
pub struct MixedPrecisionTrainer {
    /// 混合精度配置
    config: MixedPrecisionConfig,

    /// 损失缩放器
    loss_scaler: LossScaler,

    /// 连续溢出计数（用于触发自动回退）
    consecutive_overflows: usize,

    /// 是否已回退到 FP32
    fallback_triggered: bool,

    /// 原始精度类型（用于记录）
    original_precision: PrecisionType,
}

impl MixedPrecisionTrainer {
    /// **创建新的混合精度训练器**
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let loss_scaler = LossScaler::from_config(&config);
        let original_precision = config.precision_type;

        Self {
            config,
            loss_scaler,
            consecutive_overflows: 0,
            fallback_triggered: false,
            original_precision,
        }
    }

    /// **创建禁用混合精度的训练器**
    pub fn disabled() -> Self {
        Self::new(MixedPrecisionConfig::disabled())
    }

    /// **训练一个epoch**
    ///
    /// 返回 (平均损失, 是否发生溢出)
    fn train_epoch(
        &mut self,
        llm: &mut LLM,
        tokenized_data: &[Vec<usize>],
        lr: f32,
    ) -> (f32, bool) {
        let mut total_loss = 0.0;
        let mut valid_samples = 0;
        let mut epoch_had_overflow = false;

        for training_row in tokenized_data {
            if training_row.len() < 2 {
                continue;
            }

            let input_ids = &training_row[..training_row.len() - 1];
            let target_ids = &training_row[1..];

            // === 前向传播 ===
            let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
            input.row_mut(0).assign(
                &input_ids
                    .iter()
                    .map(|&x| x as f32)
                    .collect::<ndarray::Array1<f32>>(),
            );

            // 如果启用混合精度，在前向传播前模拟低精度计算
            if self.config.is_low_precision() {
                round_trip_inplace(&mut input, self.config.precision_type);
            }

            for layer in &mut llm.network {
                input = layer.forward(&input);

                // 模拟低精度计算
                if self.config.is_low_precision() {
                    round_trip_inplace(&mut input, self.config.precision_type);
                }
            }

            let logits = input;

            // 检查前向传播是否产生异常值
            if has_invalid_values(&logits) {
                warn!("[FORWARD] Invalid values detected in logits, skipping sample");
                epoch_had_overflow = true;
                continue;
            }

            let probs = softmax(&logits);

            // 计算损失
            let loss = LLM::cross_entropy_loss_step(&probs, target_ids);

            // 缩放损失（用于记录，梯度缩放在后续步骤）
            let _scaled_loss = self.loss_scaler.scale_loss(loss);
            total_loss += loss; // 记录原始损失，不是缩放后的

            // === 反向传播 ===
            let mut grads_output = LLM::compute_gradients_step(&probs, target_ids);

            // 缩放梯度（在反向传播计算中，梯度会自动被缩放）
            grads_output *= self.loss_scaler.get_scale();

            // 检查梯度是否有异常
            if has_invalid_values(&grads_output) {
                warn!("[BACKWARD] Invalid values detected in gradients before unscale");
                epoch_had_overflow = true;
                continue;
            }

            // Unscale 梯度并检查溢出
            let should_update = self.loss_scaler.unscale_gradients(&mut grads_output);

            if !should_update {
                // 检测到溢出，跳过本次更新
                epoch_had_overflow = true;
                self.consecutive_overflows += 1;

                // 检查是否需要触发自动回退
                if self.config.auto_fallback
                    && !self.fallback_triggered
                    && self.consecutive_overflows >= self.config.fallback_threshold
                {
                    self.trigger_fallback();
                }

                continue;
            }

            // 无溢出，重置连续溢出计数
            if self.consecutive_overflows > 0 {
                self.consecutive_overflows = 0;
            }

            // 梯度裁剪
            LLM::clip_gradients(&mut grads_output, 5.0);

            // 模拟低精度梯度
            if self.config.is_low_precision() {
                round_trip_inplace(&mut grads_output, self.config.precision_type);
            }

            // 反向传播（注意：master 权重在 layer 内部始终是 FP32）
            for layer in llm.network.iter_mut().rev() {
                grads_output = layer.backward(&grads_output, lr);

                // 模拟低精度梯度传播
                if self.config.is_low_precision() {
                    round_trip_inplace(&mut grads_output, self.config.precision_type);
                }
            }

            valid_samples += 1;
        }

        let avg_loss = if valid_samples > 0 {
            total_loss / valid_samples as f32
        } else {
            f32::INFINITY
        };

        (avg_loss, epoch_had_overflow)
    }

    /// **触发自动回退到 FP32**
    fn trigger_fallback(&mut self) {
        warn!(
            "[FALLBACK] Switching from {} to FP32 due to persistent instability ({} consecutive overflows)",
            self.original_precision, self.consecutive_overflows
        );

        self.config.precision_type = PrecisionType::F32;
        self.config.enabled = false;
        self.fallback_triggered = true;

        // 重新初始化 loss scaler（禁用）
        self.loss_scaler = LossScaler::disabled();
    }

    /// **完整的监控训练循环**
    ///
    /// # 参数
    /// - `llm`: 语言模型
    /// - `tokenized_data`: 预tokenize的训练数据
    /// - `max_epochs`: 最大epoch数
    /// - `initial_lr`: 初始学习率
    /// - `patience`: 早停容忍epoch数
    ///
    /// # 返回值
    /// 实际训练的epoch数
    pub fn train_monitored(
        &mut self,
        llm: &mut LLM,
        tokenized_data: Vec<Vec<usize>>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
    ) -> usize {
        llm.set_training_mode(true);

        let mut best_loss = f32::INFINITY;
        let mut counter = 0;
        let min_delta = 0.001f32;
        let mut best_epoch = 0;

        info!(
            "[MIXED PRECISION] Training with precision: {}, loss scale: {:.0}",
            self.config.precision_type,
            self.loss_scaler.get_scale()
        );

        for epoch in 0..max_epochs {
            // 余弦退火学习率
            let current_lr = LLM::cosine_annealing_lr(initial_lr, epoch, max_epochs, 2);

            // 训练一个epoch
            let (avg_loss, had_overflow) = self.train_epoch(llm, &tokenized_data, current_lr);

            // 获取溢出统计
            let (total_overflows, total_steps, overflow_rate) =
                self.loss_scaler.get_overflow_stats();

            // 打印训练信息
            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                info!(
                    "Epoch {}: Loss = {:.4}, LR = {:.6}, Precision = {}, Scale = {:.1}, Overflows = {}/{} ({:.2}%)",
                    epoch,
                    avg_loss,
                    current_lr,
                    self.config.precision_type,
                    self.loss_scaler.get_scale(),
                    total_overflows,
                    total_steps,
                    overflow_rate * 100.0
                );
            }

            // 早停检查
            if avg_loss < best_loss - min_delta {
                best_loss = avg_loss;
                best_epoch = epoch;
                counter = 0;
            } else {
                counter += 1;
                if counter >= patience {
                    info!(
                        "[EARLY STOP] No improvement for {} epochs. Best loss: {:.4} at epoch {}",
                        patience, best_loss, best_epoch
                    );
                    llm.set_training_mode(false);
                    return epoch + 1;
                }
            }
        }

        llm.set_training_mode(false);

        info!(
            "[TRAINING COMPLETE] Final stats - Loss: {:.4}, Precision: {}, Overflow rate: {:.2}%",
            best_loss,
            self.config.precision_type,
            self.loss_scaler.get_overflow_stats().2 * 100.0
        );

        max_epochs
    }

    /// **获取当前配置**
    pub fn config(&self) -> &MixedPrecisionConfig {
        &self.config
    }

    /// **获取损失缩放器统计**
    pub fn scaler_stats(&self) -> (usize, usize, f32) {
        self.loss_scaler.get_overflow_stats()
    }

    /// **检查是否已触发回退**
    pub fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    /// **获取当前精度类型**
    pub fn current_precision(&self) -> PrecisionType {
        self.config.precision_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MixedPrecisionConfig, LLM};

    #[test]
    fn test_trainer_creation() {
        let config = MixedPrecisionConfig::fp16();
        let trainer = MixedPrecisionTrainer::new(config);
        assert_eq!(trainer.current_precision(), PrecisionType::F16);
        assert!(!trainer.is_fallback_triggered());
    }

    #[test]
    fn test_disabled_trainer() {
        let trainer = MixedPrecisionTrainer::disabled();
        assert_eq!(trainer.current_precision(), PrecisionType::F32);
        assert!(!trainer.config.is_low_precision());
    }

    #[test]
    fn test_fallback_trigger() {
        let mut config = MixedPrecisionConfig::fp16();
        config.fallback_threshold = 3;
        let mut trainer = MixedPrecisionTrainer::new(config);

        // 模拟连续溢出
        trainer.consecutive_overflows = 3;
        trainer.trigger_fallback();

        assert!(trainer.is_fallback_triggered());
        assert_eq!(trainer.current_precision(), PrecisionType::F32);
    }
}
