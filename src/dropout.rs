//! # Dropout 正则化层
//!
//! Dropout 是一种简单但有效的正则化技术，通过随机"丢弃"神经元来防止过拟合。
//!
//! ## 核心思想
//!
//! **训练时**：随机将一部分神经元的输出设为0
//! **推理时**：使用所有神经元，但按比例缩放输出
//!
//! ## 为什么 Dropout 有效？
//!
//! 1. **防止神经元共适应**：
//!    - 神经元不能依赖特定的其他神经元
//!    - 必须学习更鲁棒的特征
//!
//! 2. **集成效果**：
//!    - 每次训练相当于训练不同的子网络
//!    - 最终模型是这些子网络的"平均"
//!
//! 3. **减少过拟合**：
//!    - 限制模型复杂度
//!    - 提高泛化能力
//!
//! ## Inverted Dropout（反向 Dropout）
//!
//! 本实现使用 **Inverted Dropout**，训练时进行缩放：
//!
//! ```text
//! 训练时: output = input * mask / (1 - p)
//! 推理时: output = input（不需要缩放）
//! ```
//!
//! **优势**：推理时不需要额外计算，性能更好。
//!
//! ## 示例
//!
//! ```text
//! 输入: [1.0, 2.0, 3.0, 4.0, 5.0]
//! Dropout 率: 0.2 (20% 的神经元被丢弃)
//!
//! 生成掩码: [1, 0, 1, 1, 1]  (随机丢弃第2个)
//!
//! 训练时输出:
//!   = [1.0, 2.0, 3.0, 4.0, 5.0] * [1, 0, 1, 1, 1] / 0.8
//!   = [1.25, 0.0, 3.75, 5.0, 6.25]
//!
//! 推理时输出:
//!   = [1.0, 2.0, 3.0, 4.0, 5.0]  (保持不变)
//! ```

use ndarray::Array2;
use rand::{Rng, rng};

use crate::llm::{Layer, LayerContext};

#[derive(Clone)]
struct DropoutContext {
    /// Dropout 掩码（0/1 矩阵）。
    ///
    /// 教学说明：
    /// - mask 是“一次 forward 的随机上下文”，必须随 ctx 回传给 backward；
    /// - 它不应缓存在 `self`（否则 batch/并发时会被覆盖，导致梯度与激活错配）。
    mask: Option<Array2<f32>>,

    /// Inverted Dropout 的缩放因子：`1 / (1 - p)`。
    ///
    /// 说明：
    /// - forward 与 backward 必须使用同一份 scale_factor；
    /// - 因此它也属于 ctx（而不是在 backward 时重新用 `self.dropout_rate` 计算）。
    scale_factor: f32,
}

/// **Dropout 正则化层**
pub struct Dropout {
    /// **丢弃率**: 0.0-1.0，表示神经元被丢弃的概率
    /// 常见值：0.1 (10%), 0.2 (20%), 0.5 (50%)
    pub dropout_rate: f32,

    /// **训练模式标志**
    /// - true: 训练模式，应用 dropout
    /// - false: 推理模式，不应用 dropout
    training: bool,
}

impl Dropout {
    /// **创建新的 Dropout 层**
    ///
    /// # 参数
    /// - `dropout_rate`: 丢弃率（0.0-1.0）
    ///
    /// # 常见配置
    /// - **0.1**: 轻度正则化，适用于小模型或数据充足
    /// - **0.2**: 更强的正则化，可用于更容易过拟合的设置
    /// - **0.5**: 强正则化，适用于容易过拟合的大模型
    pub fn new(dropout_rate: f32) -> Self {
        // Inverted Dropout 要求：0.0 <= p < 1.0
        //
        // p==1.0 时缩放因子是 `1/(1-p)=∞`，会立刻产生 Inf/NaN。
        assert!(
            (0.0..1.0).contains(&dropout_rate),
            "dropout_rate 必须满足 0.0 <= p < 1.0，当前={}",
            dropout_rate
        );
        Self {
            dropout_rate,
            training: true, // 默认训练模式
        }
    }

    /// **设置训练/推理模式**
    ///
    /// # 参数
    /// - `training`: true=训练模式（应用dropout），false=推理模式（不应用）
    pub fn set_training_mode(&mut self, training: bool) {
        self.training = training;
    }

    /// **生成随机掩码**
    ///
    /// 创建一个与输入形状相同的 0/1 矩阵：
    /// - 1: 保留该神经元（概率 = 1 - dropout_rate）
    /// - 0: 丢弃该神经元（概率 = dropout_rate）
    ///
    /// # 参数
    /// - `shape`: 输入张量的形状 (rows, cols)
    ///
    /// # 返回值
    /// 随机生成的掩码矩阵
    ///
    /// # 示例
    /// ```text
    /// dropout_rate = 0.2 (20%)
    /// shape = (2, 3)
    ///
    /// 可能的掩码:
    /// [[1.0, 0.0, 1.0],   // 第2列被丢弃
    ///  [1.0, 1.0, 1.0]]   // 全部保留
    /// ```
    fn create_mask(&self, shape: (usize, usize)) -> Array2<f32> {
        let mut rng = rng();
        let (rows, cols) = shape;
        let mut mask = Array2::zeros((rows, cols));

        for mut row in mask.rows_mut() {
            for element in row.iter_mut() {
                // 生成 0-1 之间的随机数
                let random_val: f32 = rng.random();

                // 如果随机数 > dropout_rate，保留该神经元
                // 例如：dropout_rate=0.2，则 80% 概率保留（random_val > 0.2）
                *element = if random_val > self.dropout_rate {
                    1.0
                } else {
                    0.0
                };
            }
        }

        mask
    }

    // 注意：历史版本曾提供 `backward_cached()` 并把 mask 缓存在 `self` 里。
    // 本轮重构已经移除该路径：mask 必须通过 ctx 显式传递（见 `Layer::forward/backward`）。
}

impl Layer for Dropout {
    fn layer_type(&self) -> &str {
        "Dropout"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        if self.training && self.dropout_rate > 0.0 {
            let mask = self.create_mask(input.dim());
            let scale_factor = 1.0 / (1.0 - self.dropout_rate);
            let mut result = input.clone();
            result *= &mask;
            result *= scale_factor;
            (
                result,
                Box::new(DropoutContext {
                    mask: Some(mask),
                    scale_factor,
                }),
            )
        } else {
            (
                input.clone(),
                Box::new(DropoutContext {
                    mask: None,
                    scale_factor: 1.0,
                }),
            )
        }
    }

    fn backward(&mut self, ctx: &LayerContext, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // 数学一致性原则：
        // - 是否应用 dropout，取决于 forward 是否生成了 mask；
        // - backward 不应依赖 `self.training`（forward/backward 之间切换 training flag 会导致错梯度）。
        let Some(ctx) = ctx.downcast_ref::<DropoutContext>() else {
            log::warn!("Dropout.backward 收到未知 ctx，直接传递梯度");
            return grads.clone();
        };

        let Some(mask) = ctx.mask.as_ref() else {
            return grads.clone();
        };

        let mut result = grads.clone();
        result *= mask;
        result *= ctx.scale_factor;
        result
    }

    fn parameters(&self) -> usize {
        0
    }

    fn set_training_mode(&mut self, training: bool) {
        self.training = training;
    }
}
