//! # Adam 优化器
//!
//! Adam (Adaptive Moment Estimation) 是一种自适应学习率的优化算法，
//! 结合了 RMSprop 和 Momentum 的优点。
//!
//! ## 核心特性
//!
//! 1. **自适应学习率**：每个参数都有独立的学习率
//! 2. **动量机制**：使用梯度的一阶矩（均值）和二阶矩（方差）
//! 3. **偏差校正**：修正初始时刻的偏差
//!
//! ## 算法原理
//!
//! ```text
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        // 一阶矩估计（动量）
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       // 二阶矩估计（RMSprop）
//! m̂_t = m_t / (1 - β₁^t)                     // 偏差校正
//! v̂_t = v_t / (1 - β₂^t)                     // 偏差校正
//! θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)      // 参数更新
//! ```
//!
//! 其中：
//! - g_t: 当前梯度
//! - m_t: 梯度的指数移动平均（动量）
//! - v_t: 梯度平方的指数移动平均（自适应学习率）
//! - β₁ = 0.9: 一阶矩的衰减率
//! - β₂ = 0.999: 二阶矩的衰减率
//! - ε = 1e-8: 数值稳定性常量，防止除零
//! - α: 学习率

use ndarray::Array2;

use crate::EPSILON;

/// **Adam 优化器结构体**
///
/// 维护梯度的一阶矩（m）和二阶矩（v），用于自适应调整每个参数的学习率。
pub struct Adam {
    /// **一阶矩衰减率 (β₁)**，默认 0.9
    pub beta1: f32,
    
    /// **二阶矩衰减率 (β₂)**，默认 0.999
    pub beta2: f32,
    
    /// **数值稳定性常量 (ε)**，防止除零
    pub epsilon: f32,
    
    /// **时间步数**，用于偏差校正
    pub timestep: usize,
    
    /// **一阶矩（梯度的指数移动平均）**
    pub m: Array2<f32>,
    
    /// **二阶矩（梯度平方的指数移动平均）**
    pub v: Array2<f32>,
}

impl Adam {
    /// **创建新的 Adam 优化器**
    ///
    /// # 参数
    /// - `shape`: 参数矩阵的形状 (行数, 列数)
    ///
    /// # 返回值
    /// 初始化的 Adam 优化器实例
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: EPSILON, // 使用统一的EPSILON常量
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
        }
    }

    /// **执行一步优化**
    ///
    /// 根据当前梯度更新参数，使用 Adam 算法。
    ///
    /// # 参数
    /// - `params`: 待更新的参数矩阵（会被原地修改）
    /// - `grads`: 当前梯度
    /// - `lr`: 学习率 (α)
    ///
    /// # 算法步骤
    /// 1. 更新时间步
    /// 2. 更新一阶矩 m (动量)
    /// 3. 更新二阶矩 v (RMSprop)
    /// 4. 偏差校正 (修正初始时刻的偏差)
    /// 5. 计算更新量并应用到参数
    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        // 1. 增加时间步
        self.timestep += 1;
        
        // 2. 更新一阶矩（动量）：m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        self.m = &self.m * self.beta1 + &(grads * (1.0 - self.beta1));
        
        // 3. 更新二阶矩（RMSprop）：v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        self.v = &self.v * self.beta2 + &(grads.mapv(|x| x * x) * (1.0 - self.beta2));

        // 4. 偏差校正
        // m̂_t = m_t / (1 - β₁^t) - 修正初始时刻的低估
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.timestep as i32));
        
        // v̂_t = v_t / (1 - β₂^t) - 修正初始时刻的低估
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.timestep as i32));

        // 5. 计算更新量：Δθ = α * m̂_t / (√v̂_t + ε)
        let update = m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon);

        // 6. 更新参数：θ_t = θ_{t-1} - Δθ
        *params -= &(update * lr);
    }
}
