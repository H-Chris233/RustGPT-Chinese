//! # 动态损失缩放器
//!
//! 实现动态损失缩放（Dynamic Loss Scaling）机制，用于混合精度训练中防止梯度下溢。
//!
//! ## 工作原理
//!
//! 1. **缩放损失**：在计算损失时乘以缩放因子
//! 2. **累积梯度**：反向传播得到缩放后的梯度
//! 3. **检测溢出**：检查梯度中是否有 NaN/Inf
//! 4. **Unscale 梯度**：将梯度除以缩放因子恢复原始大小
//! 5. **动态调整**：
//!    - 溢出时：减小缩放因子，跳过本次更新
//!    - 稳定时：逐步增大缩放因子
//!
//! ## 使用示例
//!
//! ```rust
//! use llm::loss_scaler::LossScaler;
//! use llm::mixed_precision::MixedPrecisionConfig;
//!
//! let config = MixedPrecisionConfig::fp16();
//! let mut scaler = LossScaler::from_config(&config);
//!
//! // 在训练循环中
//! let scaled_loss = scaler.scale_loss(loss);
//! // ... 反向传播 ...
//! let (gradients, should_update) = scaler.unscale_and_check(&scaled_gradients);
//! if should_update {
//!     // 执行优化器更新
//! }
//! ```

use log::{info, warn};
use ndarray::Array2;

use crate::mixed_precision::MixedPrecisionConfig;

/// **动态损失缩放器**
///
/// 管理混合精度训练中的损失缩放因子，自动检测梯度溢出并调整缩放策略。
#[derive(Debug, Clone)]
pub struct LossScaler {
    /// **当前损失缩放因子**
    current_scale: f32,

    /// **增长因子**（无溢出时缩放因子的增长倍数）
    growth_factor: f32,

    /// **回退因子**（溢出时缩放因子的缩小倍数）
    backoff_factor: f32,

    /// **增长间隔**（连续多少步无溢出才增长）
    growth_interval: usize,

    /// **自上次溢出以来的步数**
    steps_since_overflow: usize,

    /// **最大缩放因子**
    max_scale: f32,

    /// **最小缩放因子**
    min_scale: f32,

    /// **总溢出次数统计**
    total_overflows: usize,

    /// **总步数统计**
    total_steps: usize,

    /// **是否启用**（如果禁用，缩放因子始终为1.0）
    enabled: bool,
}

impl LossScaler {
    /// **从混合精度配置创建缩放器**
    pub fn from_config(config: &MixedPrecisionConfig) -> Self {
        if config.is_low_precision() {
            Self::new(
                config.loss_scale,
                config.scale_growth_factor,
                config.scale_backoff_factor,
                config.scale_growth_interval,
                config.max_loss_scale,
                config.min_loss_scale,
            )
        } else {
            Self::disabled()
        }
    }

    /// **创建新的损失缩放器**
    ///
    /// # 参数
    /// - `initial_scale`: 初始缩放因子（推荐 2^15 到 2^16）
    /// - `growth_factor`: 增长因子（推荐 2.0）
    /// - `backoff_factor`: 回退因子（推荐 0.5）
    /// - `growth_interval`: 增长间隔（推荐 1000-2000）
    /// - `max_scale`: 最大缩放因子（推荐 2^24）
    /// - `min_scale`: 最小缩放因子（推荐 1.0）
    pub fn new(
        initial_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
        max_scale: f32,
        min_scale: f32,
    ) -> Self {
        Self {
            current_scale: initial_scale.clamp(min_scale, max_scale),
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_overflow: 0,
            max_scale,
            min_scale,
            total_overflows: 0,
            total_steps: 0,
            enabled: true,
        }
    }

    /// **创建禁用的缩放器**（缩放因子始终为1.0）
    pub fn disabled() -> Self {
        Self {
            current_scale: 1.0,
            growth_factor: 1.0,
            backoff_factor: 1.0,
            growth_interval: usize::MAX,
            steps_since_overflow: 0,
            max_scale: 1.0,
            min_scale: 1.0,
            total_overflows: 0,
            total_steps: 0,
            enabled: false,
        }
    }

    /// **缩放损失**
    ///
    /// 在反向传播前将损失乘以缩放因子，以防止梯度下溢。
    ///
    /// # 参数
    /// - `loss`: 原始损失值
    ///
    /// # 返回值
    /// 缩放后的损失值
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if !self.enabled {
            return loss;
        }
        loss * self.current_scale
    }

    /// **检查梯度是否溢出**
    ///
    /// 检测梯度张量中是否存在 NaN 或 Inf 值。
    ///
    /// # 参数
    /// - `gradients`: 梯度张量
    ///
    /// # 返回值
    /// - `true`: 检测到溢出
    /// - `false`: 梯度正常
    fn has_overflow(gradients: &Array2<f32>) -> bool {
        gradients.iter().any(|&g| !g.is_finite())
    }

    /// **Unscale 梯度并检查溢出**
    ///
    /// 将缩放后的梯度除以缩放因子，并检测是否有溢出。
    /// 如果检测到溢出，会自动调整缩放因子并返回 `false`。
    ///
    /// # 参数
    /// - `gradients`: 缩放后的梯度（会被原地修改）
    ///
    /// # 返回值
    /// - `true`: 梯度正常，可以继续优化器更新
    /// - `false`: 检测到溢出，应跳过本次更新
    pub fn unscale_gradients(&mut self, gradients: &mut Array2<f32>) -> bool {
        self.total_steps += 1;

        if !self.enabled {
            return true;
        }

        // 检查是否有溢出
        if Self::has_overflow(gradients) {
            self.handle_overflow();
            return false;
        }

        // Unscale 梯度
        *gradients /= self.current_scale;

        // 再次检查 unscale 后是否溢出
        if Self::has_overflow(gradients) {
            self.handle_overflow();
            return false;
        }

        // 无溢出，增加稳定步数
        self.steps_since_overflow += 1;

        // 检查是否应该增长缩放因子
        if self.steps_since_overflow >= self.growth_interval {
            self.grow_scale();
        }

        true
    }

    /// **处理梯度溢出**
    ///
    /// 当检测到溢出时：
    /// 1. 减小缩放因子
    /// 2. 重置稳定计数器
    /// 3. 记录溢出事件
    fn handle_overflow(&mut self) {
        self.total_overflows += 1;
        let old_scale = self.current_scale;
        self.current_scale = (self.current_scale * self.backoff_factor).max(self.min_scale);
        self.steps_since_overflow = 0;

        warn!(
            "[OVERFLOW] Step {}: Loss scale reduced from {:.1} to {:.1} (overflow #{}/{})",
            self.total_steps, old_scale, self.current_scale, self.total_overflows, self.total_steps
        );
    }

    /// **增长缩放因子**
    ///
    /// 当连续多步无溢出时，增大缩放因子以提高梯度精度。
    fn grow_scale(&mut self) {
        let old_scale = self.current_scale;
        self.current_scale = (self.current_scale * self.growth_factor).min(self.max_scale);
        self.steps_since_overflow = 0;

        if (self.current_scale - old_scale).abs() > 1e-3 {
            info!(
                "[SCALE] Loss scale increased from {:.1} to {:.1} after {} stable steps",
                old_scale, self.current_scale, self.growth_interval
            );
        }
    }

    /// **获取当前缩放因子**
    pub fn get_scale(&self) -> f32 {
        self.current_scale
    }

    /// **获取溢出统计信息**
    ///
    /// # 返回值
    /// (总溢出次数, 总步数, 溢出率)
    pub fn get_overflow_stats(&self) -> (usize, usize, f32) {
        let overflow_rate = if self.total_steps > 0 {
            self.total_overflows as f32 / self.total_steps as f32
        } else {
            0.0
        };
        (self.total_overflows, self.total_steps, overflow_rate)
    }

    /// **重置统计信息**
    pub fn reset_stats(&mut self) {
        self.total_overflows = 0;
        self.total_steps = 0;
        self.steps_since_overflow = 0;
    }

    /// **检查是否频繁溢出**
    ///
    /// 用于判断是否应该触发自动回退到 FP32。
    ///
    /// # 参数
    /// - `threshold`: 连续溢出次数阈值
    ///
    /// # 返回值
    /// - `true`: 最近连续溢出次数超过阈值
    /// - `false`: 溢出在可控范围内
    pub fn is_frequently_overflowing(&self, threshold: usize) -> bool {
        self.steps_since_overflow == 0 && self.total_overflows >= threshold
    }

    /// **手动设置缩放因子**
    ///
    /// 用于从检查点恢复或手动调整。
    pub fn set_scale(&mut self, scale: f32) {
        self.current_scale = scale.clamp(self.min_scale, self.max_scale);
    }

    /// **是否启用**
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_scale_loss() {
        let scaler = LossScaler::new(1000.0, 2.0, 0.5, 100, 1e6, 1.0);
        let loss = 0.5;
        let scaled_loss = scaler.scale_loss(loss);
        assert_eq!(scaled_loss, 500.0);
    }

    #[test]
    fn test_disabled_scaler() {
        let scaler = LossScaler::disabled();
        let loss = 0.5;
        let scaled_loss = scaler.scale_loss(loss);
        assert_eq!(scaled_loss, 0.5);
        assert!(!scaler.is_enabled());
    }

    #[test]
    fn test_unscale_normal_gradients() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 100, 1e6, 1.0);
        let mut gradients = Array2::from_elem((2, 3), 1000.0);
        let should_update = scaler.unscale_gradients(&mut gradients);
        assert!(should_update);
        assert!((gradients[[0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_overflow_detection() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 100, 1e6, 1.0);
        let mut gradients = Array2::from_elem((2, 3), f32::NAN);
        let should_update = scaler.unscale_gradients(&mut gradients);
        assert!(!should_update);
        let (overflows, _, _) = scaler.get_overflow_stats();
        assert_eq!(overflows, 1);
    }

    #[test]
    fn test_scale_backoff() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 100, 1e6, 1.0);
        let initial_scale = scaler.get_scale();
        let mut gradients = Array2::from_elem((2, 3), f32::INFINITY);
        scaler.unscale_gradients(&mut gradients);
        let new_scale = scaler.get_scale();
        assert!(new_scale < initial_scale);
        assert_eq!(new_scale, initial_scale * 0.5);
    }

    #[test]
    fn test_scale_growth() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 5, 1e6, 1.0);
        let initial_scale = scaler.get_scale();

        for _ in 0..5 {
            let mut gradients = Array2::from_elem((2, 3), 1000.0);
            scaler.unscale_gradients(&mut gradients);
        }

        let new_scale = scaler.get_scale();
        assert!(new_scale > initial_scale);
        assert_eq!(new_scale, initial_scale * 2.0);
    }

    #[test]
    fn test_overflow_stats() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 100, 1e6, 1.0);

        let mut gradients = Array2::from_elem((2, 3), f32::NAN);
        scaler.unscale_gradients(&mut gradients);

        let mut gradients = Array2::from_elem((2, 3), 1000.0);
        scaler.unscale_gradients(&mut gradients);

        let (overflows, steps, rate) = scaler.get_overflow_stats();
        assert_eq!(overflows, 1);
        assert_eq!(steps, 2);
        assert!((rate - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_frequent_overflow_detection() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 100, 1e6, 1.0);

        for _ in 0..5 {
            let mut gradients = Array2::from_elem((2, 3), f32::NAN);
            scaler.unscale_gradients(&mut gradients);
        }

        assert!(scaler.is_frequently_overflowing(3));
        assert!(scaler.is_frequently_overflowing(5));
        assert!(!scaler.is_frequently_overflowing(6));
    }

    #[test]
    fn test_min_max_scale_clamping() {
        let mut scaler = LossScaler::new(1000.0, 2.0, 0.5, 1, 2000.0, 500.0);

        let mut gradients = Array2::from_elem((2, 3), 1000.0);
        scaler.unscale_gradients(&mut gradients);
        assert!(scaler.get_scale() <= 2000.0);

        for _ in 0..10 {
            let mut gradients = Array2::from_elem((2, 3), f32::NAN);
            scaler.unscale_gradients(&mut gradients);
        }
        assert!(scaler.get_scale() >= 500.0);
    }
}
