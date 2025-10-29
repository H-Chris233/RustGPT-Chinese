//! # 混合精度训练配置
//!
//! 提供 FP16/BF16 混合精度训练支持，通过降低计算精度来加速训练并减少内存占用，
//! 同时使用动态损失缩放（Dynamic Loss Scaling）确保训练稳定性。
//!
//! ## 核心概念
//!
//! 1. **双权重系统**：Master 权重（FP32）+ Working 副本（FP16/BF16）
//! 2. **动态损失缩放**：自动调整缩放因子以防止梯度下溢
//! 3. **自动回退**：检测到数值不稳定时自动切换回 FP32
//!
//! ## 使用示例
//!
//! ```rust
//! use llm::mixed_precision::{MixedPrecisionConfig, PrecisionType};
//!
//! let config = MixedPrecisionConfig::new(true, PrecisionType::F16);
//! // 在训练循环中使用配置...
//! ```

use serde::{Deserialize, Serialize};

/// **精度类型**
///
/// 定义模型前向传播和梯度计算时使用的数值精度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionType {
    /// **半精度浮点数 (FP16)**
    ///
    /// - 16位浮点数：1位符号 + 5位指数 + 10位尾数
    /// - 表示范围：±6.55e4
    /// - 精度：约3-4位有效数字
    /// - 适用场景：大部分深度学习任务
    /// - 优势：广泛支持，内存占用减半
    F16,

    /// **Brain Float 16 (BF16)**
    ///
    /// - 16位浮点数：1位符号 + 8位指数 + 7位尾数
    /// - 表示范围：与 FP32 相同（±3.4e38）
    /// - 精度：约2-3位有效数字
    /// - 适用场景：需要更大动态范围的任务
    /// - 优势：更好的数值稳定性，避免溢出
    BF16,

    /// **单精度浮点数 (FP32)**
    ///
    /// - 32位浮点数：1位符号 + 8位指数 + 23位尾数
    /// - 表示范围：±3.4e38
    /// - 精度：约7位有效数字
    /// - 适用场景：需要高精度或作为回退选项
    F32,
}

impl Default for PrecisionType {
    fn default() -> Self {
        Self::F32
    }
}

impl std::fmt::Display for PrecisionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => write!(f, "FP16"),
            Self::BF16 => write!(f, "BF16"),
            Self::F32 => write!(f, "FP32"),
        }
    }
}

impl std::str::FromStr for PrecisionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f16" | "fp16" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "f32" | "fp32" | "off" => Ok(Self::F32),
            _ => Err(format!(
                "Invalid precision type: '{}'. Expected 'f16', 'bf16', or 'off'",
                s
            )),
        }
    }
}

/// **混合精度训练配置**
///
/// 控制混合精度训练的所有参数，包括精度类型、损失缩放策略和回退机制。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// **是否启用混合精度训练**
    pub enabled: bool,

    /// **计算精度类型**（F16, BF16, F32）
    pub precision_type: PrecisionType,

    /// **初始损失缩放因子**
    ///
    /// 在计算损失时将其乘以此因子，以防止低精度下的梯度下溢。
    /// 典型值：2^15 (32768) 或 2^16 (65536)
    pub loss_scale: f32,

    /// **缩放增长因子**
    ///
    /// 当连续多步训练无溢出时，将 loss_scale 乘以此因子。
    /// 典型值：2.0
    pub scale_growth_factor: f32,

    /// **缩放回退因子**
    ///
    /// 当检测到梯度溢出时，将 loss_scale 乘以此因子。
    /// 典型值：0.5
    pub scale_backoff_factor: f32,

    /// **缩放增长间隔**
    ///
    /// 需要连续多少步无溢出才能增长 loss_scale。
    /// 典型值：1000-2000
    pub scale_growth_interval: usize,

    /// **最大损失缩放因子**
    ///
    /// 防止缩放因子无限增长。
    /// 典型值：2^24 (16777216)
    pub max_loss_scale: f32,

    /// **最小损失缩放因子**
    ///
    /// 防止缩放因子缩小到无效值。
    /// 典型值：1.0
    pub min_loss_scale: f32,

    /// **是否启用自动回退到 FP32**
    ///
    /// 当检测到持续的数值不稳定性时，自动切换回 FP32 精度。
    pub auto_fallback: bool,

    /// **回退阈值**
    ///
    /// 在多少次连续溢出后触发自动回退。
    /// 典型值：3-5
    pub fallback_threshold: usize,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            precision_type: PrecisionType::F32,
            loss_scale: 65536.0, // 2^16
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
            max_loss_scale: 16777216.0, // 2^24
            min_loss_scale: 1.0,
            auto_fallback: true,
            fallback_threshold: 5,
        }
    }
}

impl MixedPrecisionConfig {
    /// **创建新的混合精度配置**
    ///
    /// # 参数
    /// - `enabled`: 是否启用混合精度
    /// - `precision_type`: 精度类型（F16/BF16/F32）
    ///
    /// # 返回值
    /// 使用默认参数的配置实例
    pub fn new(enabled: bool, precision_type: PrecisionType) -> Self {
        Self {
            enabled,
            precision_type,
            ..Default::default()
        }
    }

    /// **创建 FP16 混合精度配置**
    pub fn fp16() -> Self {
        Self::new(true, PrecisionType::F16)
    }

    /// **创建 BF16 混合精度配置**
    pub fn bf16() -> Self {
        Self::new(true, PrecisionType::BF16)
    }

    /// **创建禁用混合精度的配置（纯 FP32）**
    pub fn disabled() -> Self {
        Self::default()
    }

    /// **检查是否实际使用低精度计算**
    pub fn is_low_precision(&self) -> bool {
        self.enabled && self.precision_type != PrecisionType::F32
    }

    /// **设置损失缩放参数**
    pub fn with_loss_scale(mut self, initial_scale: f32) -> Self {
        self.loss_scale = initial_scale;
        self
    }

    /// **设置缩放增长参数**
    pub fn with_growth_params(mut self, factor: f32, interval: usize) -> Self {
        self.scale_growth_factor = factor;
        self.scale_growth_interval = interval;
        self
    }

    /// **设置缩放回退参数**
    pub fn with_backoff_factor(mut self, factor: f32) -> Self {
        self.scale_backoff_factor = factor;
        self
    }

    /// **设置缩放范围**
    pub fn with_scale_range(mut self, min: f32, max: f32) -> Self {
        self.min_loss_scale = min;
        self.max_loss_scale = max;
        self
    }

    /// **设置自动回退**
    pub fn with_auto_fallback(mut self, enabled: bool, threshold: usize) -> Self {
        self.auto_fallback = enabled;
        self.fallback_threshold = threshold;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_type_from_str() {
        assert_eq!("f16".parse::<PrecisionType>().unwrap(), PrecisionType::F16);
        assert_eq!("fp16".parse::<PrecisionType>().unwrap(), PrecisionType::F16);
        assert_eq!(
            "bf16".parse::<PrecisionType>().unwrap(),
            PrecisionType::BF16
        );
        assert_eq!("f32".parse::<PrecisionType>().unwrap(), PrecisionType::F32);
        assert_eq!("off".parse::<PrecisionType>().unwrap(), PrecisionType::F32);
        assert!("invalid".parse::<PrecisionType>().is_err());
    }

    #[test]
    fn test_precision_type_display() {
        assert_eq!(PrecisionType::F16.to_string(), "FP16");
        assert_eq!(PrecisionType::BF16.to_string(), "BF16");
        assert_eq!(PrecisionType::F32.to_string(), "FP32");
    }

    #[test]
    fn test_default_config() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.precision_type, PrecisionType::F32);
        assert_eq!(config.loss_scale, 65536.0);
    }

    #[test]
    fn test_fp16_config() {
        let config = MixedPrecisionConfig::fp16();
        assert!(config.enabled);
        assert_eq!(config.precision_type, PrecisionType::F16);
        assert!(config.is_low_precision());
    }

    #[test]
    fn test_bf16_config() {
        let config = MixedPrecisionConfig::bf16();
        assert!(config.enabled);
        assert_eq!(config.precision_type, PrecisionType::BF16);
        assert!(config.is_low_precision());
    }

    #[test]
    fn test_disabled_config() {
        let config = MixedPrecisionConfig::disabled();
        assert!(!config.enabled);
        assert!(!config.is_low_precision());
    }

    #[test]
    fn test_config_builder() {
        let config = MixedPrecisionConfig::fp16()
            .with_loss_scale(32768.0)
            .with_growth_params(1.5, 1000)
            .with_backoff_factor(0.25)
            .with_scale_range(1.0, 1_000_000.0)
            .with_auto_fallback(false, 3);

        assert_eq!(config.loss_scale, 32768.0);
        assert_eq!(config.scale_growth_factor, 1.5);
        assert_eq!(config.scale_growth_interval, 1000);
        assert_eq!(config.scale_backoff_factor, 0.25);
        assert_eq!(config.min_loss_scale, 1.0);
        assert_eq!(config.max_loss_scale, 1_000_000.0);
        assert!(!config.auto_fallback);
        assert_eq!(config.fallback_threshold, 3);
    }
}
