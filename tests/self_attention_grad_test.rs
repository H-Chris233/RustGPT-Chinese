/// 自注意力梯度正确性与数值稳定性测试
///
/// 该测试文件聚焦三个核心场景：
/// 1. 与数值梯度对比验证反向传播实现
/// 2. 大数值输入下的梯度稳定性
/// 3. 小数值输入下的梯度稳定性
use llm::{self_attention::SelfAttention, Layer};
use ndarray::Array2;

/// 数值梯度检查使用的扰动大小
const EPSILON: f32 = 1e-3;
/// 相对误差允许范围
const REL_TOLERANCE: f32 = 0.3;

/// 基于均方误差的简易损失函数
fn compute_mse_loss(output: &Array2<f32>, target: &Array2<f32>) -> f32 {
    let diff = output - target;
    diff.mapv(|x| x * x).sum() / (output.len() as f32)
}

/// 有限差分法估计某个输入位置的梯度
fn numerical_gradient(
    attention: &mut SelfAttention,
    input: &Array2<f32>,
    target: &Array2<f32>,
    i: usize,
    j: usize,
) -> f32 {
    let mut input_plus = input.clone();
    input_plus[[i, j]] += EPSILON;
    let (output_plus, _ctx) = attention.forward(&input_plus);
    let loss_plus = compute_mse_loss(&output_plus, target);

    let mut input_minus = input.clone();
    input_minus[[i, j]] -= EPSILON;
    let (output_minus, _ctx) = attention.forward(&input_minus);
    let loss_minus = compute_mse_loss(&output_minus, target);

    (loss_plus - loss_minus) / (2.0 * EPSILON)
}

#[test]
fn test_gradient_matches_numerical_estimate() {
    let embedding_dim = 64; // 使用较小的维度以加快测试
    let seq_len = 3;

    let mut attention = SelfAttention::new(embedding_dim);
    let mut input = Array2::zeros((seq_len, embedding_dim));
    for i in 0..seq_len {
        for j in 0..embedding_dim {
            input[[i, j]] = ((i + j) as f32) * 0.05;
        }
    }

    let target = Array2::ones((seq_len, embedding_dim)) * 0.5;

    let (output, ctx) = attention.forward(&input);
    let grad_output = (&output - &target) * (2.0 / output.len() as f32);
    let grad_input = attention.backward(&ctx, &grad_output, 0.0);

    let sample_positions = vec![
        (0, 0),
        (1, embedding_dim / 2),
        (seq_len - 1, embedding_dim - 1),
    ];
    for (i, j) in sample_positions {
        let numerical = numerical_gradient(&mut attention, &input, &target, i, j);
        let analytical = grad_input[[i, j]];

        let abs_err = (numerical - analytical).abs();
        let rel_err = if numerical.abs() > 1e-5 {
            abs_err / numerical.abs()
        } else {
            abs_err
        };

        assert!(
            rel_err <= REL_TOLERANCE || abs_err < 2e-2,
            "梯度验证失败: idx=({},{}) 数值梯度={:.6} 解析梯度={:.6} 绝对误差={:.6} 相对误差={:.6}",
            i,
            j,
            numerical,
            analytical,
            abs_err,
            rel_err
        );
    }
}

#[test]
fn test_gradient_stability_large_values() {
    let embedding_dim = 128;
    let seq_len = 4;

    let mut attention = SelfAttention::new(embedding_dim);
    let mut input = Array2::zeros((seq_len, embedding_dim));
    for i in 0..seq_len {
        for j in 0..embedding_dim {
            input[[i, j]] = 200.0 + (i as f32) * 10.0 + (j as f32) * 0.1;
        }
    }

    let (output, ctx) = attention.forward(&input);
    let grad_output = Array2::ones(output.dim());
    let grad_input = attention.backward(&ctx, &grad_output, 0.0);

    assert!(
        output.iter().all(|&v| v.is_finite()),
        "前向输出出现非有限值"
    );
    assert!(
        grad_input.iter().all(|&v| v.is_finite()),
        "梯度出现非有限值"
    );
    assert!(
        grad_input.iter().all(|&v| v.abs() < 1e8),
        "梯度绝对值过大，可能存在数值不稳定"
    );
}

#[test]
fn test_gradient_stability_small_values() {
    let embedding_dim = 128;
    let seq_len = 4;

    let mut attention = SelfAttention::new(embedding_dim);
    let mut input = Array2::zeros((seq_len, embedding_dim));
    for i in 0..seq_len {
        for j in 0..embedding_dim {
            input[[i, j]] = 1e-6 * ((i + j) as f32);
        }
    }

    let (output, ctx) = attention.forward(&input);
    let grad_output = Array2::ones(output.dim()) * 1e-3;
    let grad_input = attention.backward(&ctx, &grad_output, 0.0);

    assert!(
        output.iter().all(|&v| v.is_finite()),
        "前向输出出现非有限值"
    );
    assert!(
        grad_input.iter().all(|&v| v.is_finite()),
        "梯度出现非有限值"
    );
}
