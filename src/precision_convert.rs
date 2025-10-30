//! # 精度转换工具
//!
//! 提供 FP32、FP16、BF16 之间的高效转换函数，用于混合精度训练。
//!
//! ## 核心功能
//!
//! 1. **批量转换**：Array2<f32> ↔ Vec<u16>（F16/BF16 的位表示）
//! 2. **原地转换**：避免不必要的内存分配
//! 3. **SIMD 友好**：设计上便于编译器优化
//!
//! ## 使用示例
//!
//! ```rust
//! use llm::precision_convert::{to_fp16, from_fp16};
//! use ndarray::Array2;
//!
//! let weights_fp32 = Array2::from_elem((2, 3), 1.5f32);
//! let weights_fp16 = to_fp16(&weights_fp32);
//! let restored = from_fp16(&weights_fp16, (2, 3));
//! ```

use half::{bf16, f16};
use ndarray::Array2;

use crate::mixed_precision::PrecisionType;

/// **将 FP32 数组转换为 FP16**
///
/// # 参数
/// - `arr`: FP32 精度的二维数组
///
/// # 返回值
/// FP16 的位表示（u16 向量）
pub fn to_fp16(arr: &Array2<f32>) -> Vec<u16> {
    arr.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

/// **从 FP16 恢复到 FP32**
///
/// # 参数
/// - `data`: FP16 的位表示（u16 向量）
/// - `shape`: 目标数组形状 (rows, cols)
///
/// # 返回值
/// FP32 精度的二维数组
pub fn from_fp16(data: &[u16], shape: (usize, usize)) -> Array2<f32> {
    let values: Vec<f32> = data
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect();
    Array2::from_shape_vec(shape, values).expect("Shape mismatch in from_fp16")
}

/// **将 FP32 数组转换为 BF16**
///
/// # 参数
/// - `arr`: FP32 精度的二维数组
///
/// # 返回值
/// BF16 的位表示（u16 向量）
pub fn to_bf16(arr: &Array2<f32>) -> Vec<u16> {
    arr.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}

/// **从 BF16 恢复到 FP32**
///
/// # 参数
/// - `data`: BF16 的位表示（u16 向量）
/// - `shape`: 目标数组形状 (rows, cols)
///
/// # 返回值
/// FP32 精度的二维数组
pub fn from_bf16(data: &[u16], shape: (usize, usize)) -> Array2<f32> {
    let values: Vec<f32> = data
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect();
    Array2::from_shape_vec(shape, values).expect("Shape mismatch in from_bf16")
}

/// **原地转换：FP32 -> 低精度 -> FP32**
///
/// 用于模拟低精度计算对数值的影响。
///
/// # 参数
/// - `arr`: FP32 数组（会被原地修改）
/// - `precision`: 目标精度类型
pub fn round_trip_inplace(arr: &mut Array2<f32>, precision: PrecisionType) {
    match precision {
        PrecisionType::F16 => {
            for val in arr.iter_mut() {
                *val = f16::from_f32(*val).to_f32();
            }
        }
        PrecisionType::BF16 => {
            for val in arr.iter_mut() {
                *val = bf16::from_f32(*val).to_f32();
            }
        }
        PrecisionType::F32 => {
            // 不做任何转换
        }
    }
}

/// **检查数组是否包含异常值**
///
/// 检测 NaN 或 Inf，用于溢出检测。
///
/// # 参数
/// - `arr`: 待检查的数组
///
/// # 返回值
/// - `true`: 包含 NaN 或 Inf
/// - `false`: 所有值都正常
pub fn has_invalid_values(arr: &Array2<f32>) -> bool {
    arr.iter().any(|&x| !x.is_finite())
}

/// **将数组值裁剪到安全范围**
///
/// 防止低精度下的溢出。
///
/// # 参数
/// - `arr`: 待裁剪的数组（会被原地修改）
/// - `precision`: 精度类型（决定裁剪范围）
pub fn clip_to_safe_range(arr: &mut Array2<f32>, precision: PrecisionType) {
    let (min_val, max_val) = match precision {
        PrecisionType::F16 => (-65504.0f32, 65504.0f32), // FP16 的最大值
        PrecisionType::BF16 => (-3.38e38f32, 3.38e38f32), // BF16 与 FP32 范围相同
        PrecisionType::F32 => (f32::MIN, f32::MAX),
    };

    for val in arr.iter_mut() {
        if !val.is_finite() {
            *val = 0.0;
        } else {
            *val = val.clamp(min_val, max_val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_fp16_round_trip() {
        let original = Array2::from_elem((2, 3), 1.5f32);
        let fp16_data = to_fp16(&original);
        let restored = from_fp16(&fp16_data, (2, 3));

        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 1e-3); // FP16 精度约 1e-3
        }
    }

    #[test]
    fn test_bf16_round_trip() {
        let original = Array2::from_elem((2, 3), 1.5f32);
        let bf16_data = to_bf16(&original);
        let restored = from_bf16(&bf16_data, (2, 3));

        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 1e-2); // BF16 精度约 1e-2
        }
    }

    #[test]
    fn test_round_trip_inplace() {
        let mut arr = Array2::from_elem((2, 3), 1.234567f32);
        let original_val = arr[[0, 0]];

        round_trip_inplace(&mut arr, PrecisionType::F16);
        let fp16_val = arr[[0, 0]];
        assert!((original_val - fp16_val).abs() < 1e-2);

        arr = Array2::from_elem((2, 3), 1.234567f32);
        round_trip_inplace(&mut arr, PrecisionType::BF16);
        let bf16_val = arr[[0, 0]];
        assert!((original_val - bf16_val).abs() < 1e-2);

        arr = Array2::from_elem((2, 3), 1.234567f32);
        round_trip_inplace(&mut arr, PrecisionType::F32);
        assert_eq!(arr[[0, 0]], original_val);
    }

    #[test]
    fn test_has_invalid_values() {
        let normal = Array2::from_elem((2, 3), 1.5f32);
        assert!(!has_invalid_values(&normal));

        let with_nan = Array2::from_elem((2, 3), f32::NAN);
        assert!(has_invalid_values(&with_nan));

        let with_inf = Array2::from_elem((2, 3), f32::INFINITY);
        assert!(has_invalid_values(&with_inf));
    }

    #[test]
    fn test_clip_to_safe_range() {
        let mut arr = Array2::from_elem((2, 3), 70000.0f32);
        clip_to_safe_range(&mut arr, PrecisionType::F16);
        assert!(arr[[0, 0]] <= 65504.0);

        let mut arr_with_nan = Array2::from_elem((2, 3), f32::NAN);
        clip_to_safe_range(&mut arr_with_nan, PrecisionType::F16);
        assert_eq!(arr_with_nan[[0, 0]], 0.0);
    }

    #[test]
    fn test_precision_loss() {
        // 测试低精度转换的精度损失
        let original = Array2::from_shape_vec((1, 3), vec![0.1f32, 0.01f32, 0.001f32]).unwrap();

        // FP16 转换
        let fp16_data = to_fp16(&original);
        let restored_fp16 = from_fp16(&fp16_data, (1, 3));

        // FP16 在小数上有明显精度损失
        assert!((original[[0, 0]] - restored_fp16[[0, 0]]).abs() < 1e-3);
        assert!((original[[0, 1]] - restored_fp16[[0, 1]]).abs() < 1e-4);
        assert!((original[[0, 2]] - restored_fp16[[0, 2]]).abs() < 1e-4);

        // BF16 转换
        let bf16_data = to_bf16(&original);
        let restored_bf16 = from_bf16(&bf16_data, (1, 3));

        // BF16 精度损失较大但动态范围更广
        assert!((original[[0, 0]] - restored_bf16[[0, 0]]).abs() < 1e-2);
    }
}
