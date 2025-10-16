/// 工具函数模块
///
/// 包含数学运算、激活函数等通用工具
use ndarray::Array2;

// Softmax专用的epsilon常量（避免除零）
const SOFTMAX_EPS: f32 = 1e-12;

/// Softmax激活函数
///
/// 对输入张量的每一行应用softmax，将数值转换为概率分布。
/// 使用数值稳定的实现（减去最大值避免溢出）。
///
/// # 参数
/// - `logits`: 输入张量，形状为 (batch_size, num_classes)
///
/// # 返回
/// Softmax输出，形状与输入相同，每行元素和为1
///
/// # 数值稳定性
/// 1. 每行减去最大值，避免exp溢出
/// 2. 除以总和时添加epsilon，避免除零错误
pub fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = logits.clone();

    for mut row in result.rows_mut() {
        // 找到该行的最大值（用于数值稳定）
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // 计算exp(x - max)
        row.mapv_inplace(|x| (x - max_val).exp());

        // 归一化
        let sum_exp: f32 = row.sum();
        row.mapv_inplace(|x| x / sum_exp.max(SOFTMAX_EPS));
    }

    result
}

/// 稳定的 log_softmax 实现
///
/// 对每一行计算：log_softmax(x_i) = x_i - log(sum_j exp(x_j))，
/// 使用减去行最大值的方式避免数值溢出。
pub fn log_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = logits.clone();

    for mut row in result.rows_mut() {
        // 数值稳定：减去该行最大值
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // 先计算 shifted = x - max
        let mut sum_exp = 0.0f32;
        for v in row.iter_mut() {
            *v = *v - max_val;
            sum_exp += v.exp();
        }
        // logsumexp = log(sum(exp(shifted)))
        let log_sum_exp = sum_exp.max(SOFTMAX_EPS).ln();
        // log_softmax = shifted - logsumexp
        for v in row.iter_mut() {
            *v = *v - log_sum_exp;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_softmax_basic() {
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = softmax(&input);

        // 检查每行和为1
        for row in output.rows() {
            let sum: f32 = row.sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Row sum should be 1.0, got {}",
                sum
            );
        }

        // 检查所有值在[0, 1]区间
        for &val in output.iter() {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value should be in [0, 1], got {}",
                val
            );
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 测试大数值的稳定性
        let input = Array2::from_shape_vec((1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        let output = softmax(&input);

        // 应该不会产生NaN或Inf
        for &val in output.iter() {
            assert!(val.is_finite(), "Value should be finite, got {}", val);
        }
    }

    #[test]
    fn test_log_softmax_properties() {
        let input = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let ls = log_softmax(&input);
        let s = softmax(&input);
        for i in 0..3 {
            assert!((ls[[0, i]].exp() - s[[0, i]]).abs() < 1e-6);
        }
        // log_softmax 的指数应当归一化为1
        let sum_exp: f32 = (0..3).map(|i| ls[[0, i]].exp()).sum();
        assert!((sum_exp - 1.0).abs() < 1e-6);
    }
}
