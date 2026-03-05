//! `LLM::compute_gradients_step` 的单元测试
//!
//! 目标：锁定关键行为，防止回归：
//! - shape 不匹配时返回全 0（而不是把 probs 当作梯度）
//! - PAD/非法 target 不产生训练信号（对应行梯度全 0）

use llm::LLM;
use ndarray::Array2;

#[test]
fn compute_gradients_step_returns_err_on_shape_mismatch() {
    let probs = Array2::from_shape_vec((2, 3), vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3]).unwrap();
    let target = vec![1usize]; // len=1，与 probs 行数=2 不匹配

    let err = LLM::compute_gradients_step(&probs, &target, 0).unwrap_err();
    assert!(
        matches!(err, llm::llm::TrainingSignalError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn compute_gradients_step_zeros_pad_rows() {
    let probs = Array2::from_shape_vec((2, 3), vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3]).unwrap();
    let target = vec![0usize, 1usize]; // row0=PAD

    let grads = LLM::compute_gradients_step(&probs, &target, 0)
        .unwrap()
        .expect("expected Some(grads)");

    // PAD 行梯度应全 0
    assert!(grads.row(0).iter().all(|&x| x == 0.0));

    // 有效行：softmax - one_hot（n_targets=1，不会额外缩放）
    let expected_row1 = vec![0.1, -0.4, 0.3];
    for (a, b) in grads.row(1).iter().zip(expected_row1) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }
}

#[test]
fn compute_gradients_step_returns_err_on_out_of_range_target() {
    let probs = Array2::from_shape_vec((2, 3), vec![0.25, 0.25, 0.5, 0.2, 0.5, 0.3]).unwrap();
    let target = vec![999usize, 2usize]; // row0 越界

    let err = LLM::compute_gradients_step(&probs, &target, 0).unwrap_err();
    assert!(
        matches!(err, llm::llm::TrainingSignalError::TargetOutOfRange { .. }),
        "expected TargetOutOfRange, got {err:?}"
    );
}

#[test]
fn compute_gradients_step_all_pad_returns_none_and_all_invalid_returns_err() {
    let probs = Array2::from_shape_vec((2, 3), vec![0.3, 0.3, 0.4, 0.1, 0.1, 0.8]).unwrap();

    let grads_all_pad = LLM::compute_gradients_step(&probs, &[0, 0], 0).unwrap();
    assert!(grads_all_pad.is_none());

    assert!(LLM::compute_gradients_step(&probs, &[999, 999], 0).is_err());
}

#[test]
fn compute_gradients_step_scales_by_number_of_valid_targets() {
    // 两个有效 target：期望整体按 n_targets=2 做平均
    let probs = Array2::from_shape_vec((2, 3), vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3]).unwrap();
    let target = vec![1usize, 2usize];

    let grads = LLM::compute_gradients_step(&probs, &target, 0)
        .unwrap()
        .expect("expected Some(grads)");

    let expected_row0 = vec![0.1, -0.35, 0.25];
    let expected_row1 = vec![0.05, 0.3, -0.35];

    for (a, b) in grads.row(0).iter().zip(expected_row0) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }
    for (a, b) in grads.row(1).iter().zip(expected_row1) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }
}
