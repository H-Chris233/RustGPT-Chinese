/// 测试自注意力矩阵运算与稳定性优化
///
/// 验证：
/// 1. 因果掩码缓存机制
/// 2. 矩阵乘法优化
/// 3. 稳定的softmax实现
/// 4. 梯度数值稳定性
use llm::{EMBEDDING_DIM, Layer, self_attention::SelfAttention};
use ndarray::{Array2, s};

#[test]
fn test_causal_mask_caching() {
    // 创建自注意力层
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 首次前向传播，应该创建并缓存掩码
    let seq_len1 = 5;
    let input1 = Array2::ones((seq_len1, EMBEDDING_DIM));
    let output1 = self_attention.forward(&input1);

    // 验证掩码已被缓存
    assert!(self_attention.causal_mask_cache.contains_key(&seq_len1));

    // 验证掩码的正确性：下三角为0，上三角为NEG_INFINITY
    let mask = &self_attention.causal_mask_cache[&seq_len1];
    for i in 0..seq_len1 {
        for j in 0..seq_len1 {
            if j > i {
                assert_eq!(
                    mask[[i, j]],
                    f32::NEG_INFINITY,
                    "位置[{},{}]应该是NEG_INFINITY",
                    i,
                    j
                );
            } else {
                assert_eq!(mask[[i, j]], 0.0, "位置[{},{}]应该是0", i, j);
            }
        }
    }

    // 第二次使用相同序列长度，应该复用缓存
    let input2 = Array2::ones((seq_len1, EMBEDDING_DIM)) * 2.0;
    let output2 = self_attention.forward(&input2);

    // 输出形状应该正确
    assert_eq!(output1.shape(), [seq_len1, EMBEDDING_DIM]);
    assert_eq!(output2.shape(), [seq_len1, EMBEDDING_DIM]);

    // 不同序列长度应该创建新的缓存条目
    let seq_len2 = 10;
    let input3 = Array2::ones((seq_len2, EMBEDDING_DIM));
    let _output3 = self_attention.forward(&input3);

    // 验证两个不同长度的掩码都被缓存
    assert!(self_attention.causal_mask_cache.contains_key(&seq_len1));
    assert!(self_attention.causal_mask_cache.contains_key(&seq_len2));
    assert_eq!(self_attention.causal_mask_cache.len(), 2);
}

#[test]
fn test_multiple_sequence_lengths() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 测试多个不同的序列长度
    for seq_len in [1, 3, 5, 8, 16, 32, 64] {
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        let output = self_attention.forward(&input);

        // 验证输出形状
        assert_eq!(
            output.shape(),
            [seq_len, EMBEDDING_DIM],
            "序列长度{}的输出形状不正确",
            seq_len
        );

        // 验证掩码被正确缓存
        assert!(
            self_attention.causal_mask_cache.contains_key(&seq_len),
            "序列长度{}的掩码未被缓存",
            seq_len
        );
    }
}

#[test]
fn test_numerical_stability_with_large_values() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 测试大数值输入的数值稳定性
    let seq_len = 10;
    let mut input = Array2::zeros((seq_len, EMBEDDING_DIM));

    // 填充大数值
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            input[[i, j]] = 100.0 + (i as f32) * 10.0;
        }
    }

    let output = self_attention.forward(&input);

    // 验证输出不包含NaN或Inf
    for &val in output.iter() {
        assert!(val.is_finite(), "输出包含非有限值: {}", val);
    }

    // 验证输出形状
    assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
}

#[test]
fn test_numerical_stability_with_small_values() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 测试小数值输入的数值稳定性
    let seq_len = 10;
    let mut input = Array2::zeros((seq_len, EMBEDDING_DIM));

    // 填充小数值
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            input[[i, j]] = 1e-6 * ((i + j) as f32);
        }
    }

    let output = self_attention.forward(&input);

    // 验证输出不包含NaN或Inf
    for &val in output.iter() {
        assert!(val.is_finite(), "输出包含非有限值: {}", val);
    }

    // 验证输出形状
    assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
}

#[test]
fn test_gradient_flow() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 前向传播
    let seq_len = 5;
    let input = Array2::ones((seq_len, EMBEDDING_DIM));
    let _output = self_attention.forward(&input);

    // 反向传播
    let grad_output = Array2::ones((seq_len, EMBEDDING_DIM));
    let grad_input = self_attention.backward(&grad_output, 0.001);

    // 验证梯度不包含NaN或Inf
    for &val in grad_input.iter() {
        assert!(val.is_finite(), "梯度包含非有限值: {}", val);
    }

    // 验证梯度形状
    assert_eq!(grad_input.shape(), input.shape());
}

#[test]
fn test_gradient_stability_with_extreme_values() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 测试极端值情况下的梯度稳定性
    let seq_len = 5;
    let mut input = Array2::zeros((seq_len, EMBEDDING_DIM));

    // 混合大小数值
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            if i % 2 == 0 {
                input[[i, j]] = 100.0;
            } else {
                input[[i, j]] = -100.0;
            }
        }
    }

    let _output = self_attention.forward(&input);

    // 使用极端梯度值
    let mut grad_output = Array2::zeros((seq_len, EMBEDDING_DIM));
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            grad_output[[i, j]] = if (i + j) % 2 == 0 { 10.0 } else { -10.0 };
        }
    }

    let grad_input = self_attention.backward(&grad_output, 0.001);

    // 验证梯度稳定性
    for &val in grad_input.iter() {
        assert!(val.is_finite(), "梯度包含非有限值: {}", val);
        assert!(val.abs() < 1e6, "梯度值过大: {}", val);
    }
}

#[test]
fn test_forward_backward_consistency() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 多次前向-反向传播，验证一致性
    let seq_len = 5;
    let input = Array2::ones((seq_len, EMBEDDING_DIM));

    for iter in 0..10 {
        let output = self_attention.forward(&input);

        // 验证输出有效
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "迭代{}：输出包含无效值",
            iter
        );

        let grad_output = Array2::ones((seq_len, EMBEDDING_DIM));
        let grad_input = self_attention.backward(&grad_output, 0.001);

        // 验证梯度有效
        assert!(
            grad_input.iter().all(|&v| v.is_finite()),
            "迭代{}：梯度包含无效值",
            iter
        );
    }
}

#[test]
fn test_mask_cache_performance() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    let seq_len = 20;
    let input = Array2::ones((seq_len, EMBEDDING_DIM));

    // 第一次调用会创建缓存
    let start = std::time::Instant::now();
    let _output1 = self_attention.forward(&input);
    let first_duration = start.elapsed();

    // 后续调用应该复用缓存（理论上更快，但由于其他计算占主导，差异可能不明显）
    let start = std::time::Instant::now();
    let _output2 = self_attention.forward(&input);
    let second_duration = start.elapsed();

    println!("第一次调用: {:?}", first_duration);
    println!("第二次调用: {:?}", second_duration);

    // 验证缓存存在
    assert!(self_attention.causal_mask_cache.contains_key(&seq_len));
}

#[test]
fn test_output_causality() {
    // 验证因果掩码确实起作用：位置i的输出不应该依赖于位置j (j > i)
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    let seq_len = 5;

    // 创建一个输入，其中后面的位置有不同的值
    let input1 = Array2::ones((seq_len, EMBEDDING_DIM));
    let output1 = self_attention.forward(&input1);

    // 修改最后一个位置的值（未来 token）
    let mut input2 = input1.clone();
    input2.row_mut(seq_len - 1).fill(999.0);
    let output2 = self_attention.forward(&input2);

    // 两个输出的形状应该相同
    assert_eq!(output1.shape(), output2.shape());

    // 输出应该都是有限的
    assert!(output1.iter().all(|&v| v.is_finite()));
    assert!(output2.iter().all(|&v| v.is_finite()));

    // 核心断言：
    // - 由于因果掩码，位置 < last 的输出不应受到 last token 变化影响。
    let prefix1 = output1.slice(s![0..seq_len - 1, ..]);
    let prefix2 = output2.slice(s![0..seq_len - 1, ..]);
    let max_abs_diff = prefix1
        .iter()
        .zip(prefix2.iter())
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(
        max_abs_diff < 1e-6,
        "因果性违反：修改未来 token 影响了过去输出，max_abs_diff={}",
        max_abs_diff
    );
}

#[test]
fn test_different_batch_sizes() {
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // 测试不同的"批次大小"（实际上是序列长度）
    for seq_len in [1, 2, 4, 8, 16] {
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        let output = self_attention.forward(&input);

        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
        assert!(output.iter().all(|&v| v.is_finite()));
    }
}
