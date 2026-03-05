//! 回归测试：`cosine_annealing_lr` 在极端参数下不应 panic（cycle_length=0）。

use llm::LLM;

#[test]
fn cosine_annealing_lr_does_not_panic_when_cycle_length_zero() {
    // 旧实现：
    // total_epochs / (num_restarts+1) == 0 会导致 `epoch % cycle_length` 取模 0 panic。
    //
    // 新实现应当做鲁棒性保护（cycle_length>=1），返回一个有限的学习率。
    let lr = LLM::cosine_annealing_lr(0.001, 0, 1, 10);
    assert!(lr.is_finite(), "lr should be finite, got {}", lr);
    assert!(lr > 0.0, "lr should be >0, got {}", lr);
}

