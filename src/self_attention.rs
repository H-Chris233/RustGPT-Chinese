//! # 多头自注意力机制（Multi-Head Self-Attention）
//!
//! 这是 Transformer 架构的核心创新，让模型能够捕捉序列中的长距离依赖关系。
//!
//! ## 性能优化（v0.3.2）
//!
//! ### 1. 因果掩码缓存
//! - **问题**: 每次前向传播都需要逐元素填充 NEG_INFINITY 创建掩码矩阵
//! - **解决**: 使用 HashMap 缓存不同序列长度的掩码，避免重复创建
//! - **收益**: 减少 O(seq_len²) 的掩码创建开销
//!
//! ### 2. 优化矩阵乘法
//! - **策略**: 使用 ndarray 的优化 dot() 方法（基于 BLAS）
//! - **掩码应用**: 使用矩阵加法替代逐元素设置
//! - **并行处理**: 多头计算使用 rayon 并行化
//!
//! ### 3. 稳定的 Softmax 实现
//! - **数值稳定性**: 使用 log-sum-exp 技巧（减去最大值）
//! - **避免溢出**: 处理极大/极小值时保持数值稳定
//! - **梯度计算**: 简化但稳定的反向传播（注：完整梯度计算较复杂，当前使用近似）
//!
//! ## 核心思想：注意力即"权重分配"
//!
//! 在自然语言中，不是所有词都同等重要。注意力机制让模型学习：
//! - 哪些词在理解当前词时更"相关"
//! - 如何动态调整对不同词的关注程度
//!
//! **直观理解**：
//! ```text
//! "我昨天在书店买了一本《机器学习》"
//!
//! 在理解"买"这个动作时，模型应该：
//! - 更多关注"我"（谁买？）→ 权重 0.5
//! - 适中关注"书店"（在哪里买？）→ 权重 0.3
//! - 少量关注"昨天"（什么时候？）→ 权重 0.2
//! ```
//!
//! ## 数学原理：缩放点积注意力（Scaled Dot-Product Attention）
//!
//! ### 公式
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d_k) · V
//! ```
//!
//! **变量说明**：
//! - **Q（Query）**: "我在寻找什么" - 当前位置的查询向量
//! - **K（Key）**: "我能提供什么" - 所有位置的键向量（用于匹配）
//! - **V（Value）**: "我具体是什么" - 所有位置的值向量（用于加权）
//! - **d_k**: Key/Query 的维度（64），用于缩放防止梯度消失
//!
//! ### 步骤详解
//! 1. **计算相似度**: `Q·K^T` - 查询与每个键的点积（相似度越高，点积越大）
//! 2. **缩放**: `/ √d_k` - 防止点积过大导致 softmax 梯度消失
//! 3. **归一化**: `softmax` - 转换为概率分布（总和为1）
//! 4. **加权求和**: `·V` - 根据注意力权重加权各个值
//!
//! ## 多头机制
//!
//! **为什么需要多头？** 不同头学习不同类型的 attention：
//!
//! - **Head 1**: 语法关系（主谓宾结构）
//! - **Head 2**: 语义关系（同义词、反义词）
//! - **Head 3**: 位置关系（远近、顺序）
//! - **Head 4-8**: 其他抽象模式
//!
//! **实现方式**：
//! - 将 512 维分成 8 个头，每个头 64 维
//! - 并行计算 8 个注意力
//! - 最后拼接并投影回 512 维
//!
//! ## Causal Mask（因果掩码）
//!
//! **问题**：在生成文本时，模型不应该看到未来的词。
//!
//! **解决方案**：使用下三角矩阵将未来位置的注意力设为 -∞
//!
//! ```text
//! 注意力掩码矩阵：
//!    位置0 位置1 位置2 位置3
//! 0  [  √    ×    ×    ×  ]  ✓ 只能看自己
//! 1  [  √    √    ×    ×  ]  ✓ 可以看到位置0和1
//! 2  [  √    √    √    ×  ]  ✓ 可以看到位置0-2
//! 3  [  √    √    √    √  ]  ✓ 可以看到所有位置
//!
//! （× 表示设为 -∞，softmax 后概率为 0）
//! ```

use std::collections::HashMap;
use std::f32;

use ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis};
use ndarray::linalg::general_mat_mul;

use crate::{
    EMBEDDING_DIM,
    adam::Adam,
    llm::{Layer, LayerContext},
    utils::sample_normal,
};

/// SelfAttention 层的 forward/backward 上下文。
///
/// 教学说明：
/// - 注意力层的反向传播需要大量中间量（Q/K/V、softmax 权重、输出投影前的 attention_output 等）。
/// - 旧实现把这些值存进 self.cached_*，在 batch（同一层实例连续 forward 多个样本）时会被覆盖。
/// - 新版将中间量放到 ctx 中：每个样本一份 ctx，batch 时不会错配。
#[derive(Clone)]
struct SelfAttentionContext {
    input: Array2<f32>,
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    attention_output: Array2<f32>,
    attention_weights: Option<Vec<Array2<f32>>>,
}

/// **稳定的 Softmax 实现（使用 log-sum-exp 技巧）**
///
/// 通过减去最大值来避免数值溢出，确保在处理大数值时的稳定性。
///
/// # 参数
/// - `logits`: 输入矩阵 (seq_len, seq_len)
///
/// # 返回值
/// Softmax 输出，形状与输入相同：
/// - **正常行**：每行元素和为 1
/// - **全被 mask 的行**（例如整行都是 `-∞`）：返回全 0（行和为 0）
fn stable_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(logits.dim());

    for (i, row) in logits.rows().into_iter().enumerate() {
        // 找到该行的最大值（数值稳定性）
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // 极端情况：
        // - 如果一整行都是 -∞（例如：padding mask 把所有 key 都屏蔽掉），那么 softmax 在数学上
        //   是未定义的（0/0）。此时我们返回“全 0 权重”，并让上层输出自然变为 0，避免把
        //   PAD/value 当成有效信息做均匀平均。
        if max_val.is_nan() {
            log::warn!("stable_softmax: logits contains NaN; returning zero row (row={})", i);
            continue;
        }
        if max_val == f32::NEG_INFINITY {
            // 全被 mask（或全是 -∞）
            continue;
        }
        if max_val.is_infinite() {
            log::warn!(
                "stable_softmax: logits contains infinite value; returning zero row (row={})",
                i
            );
            // 保持 result 的该行全零
            continue;
        }

        // 计算 exp(x - max)
        let exp_vals = row.mapv(|x| (x - max_val).exp());

        // 计算归一化因子
        let sum_exp: f32 = exp_vals.sum();

        // 归一化（添加epsilon避免除零）
        let normalized = if sum_exp.is_finite() && sum_exp > 1e-15 {
            exp_vals.mapv(|x| x / sum_exp)
        } else {
            if !sum_exp.is_finite() {
                log::warn!(
                    "stable_softmax: sum_exp is not finite; returning zero row (row={}, sum_exp={})",
                    i,
                    sum_exp
                );
            }
            // 如果所有值都极小/非有限，返回全 0 分布（与 “全部被 mask” 的语义一致）
            Array1::zeros(exp_vals.len())
        };

        result.row_mut(i).assign(&normalized);
    }

    result
}

/// **稳定的 Softmax 梯度计算**
///
/// 给定 softmax 的输出和上游梯度，计算对 softmax 输入的梯度。
///
/// # 数学原理
/// 对于 softmax: y_i = exp(x_i) / sum(exp(x_j))
/// 梯度公式: ∂L/∂x_i = y_i * (∂L/∂y_i - sum_j(y_j * ∂L/∂y_j))
///
/// 这个公式确保了数值稳定性，并且正确处理了 softmax 的 Jacobian 矩阵。
///
/// # 参数
/// - `softmax_output`: Softmax 的输出 (seq_len, seq_len)
/// - `grad_output`: 上游梯度 (seq_len, seq_len)
///
/// # 返回值
/// 对 softmax 输入的梯度，形状与输入相同
fn stable_softmax_gradient(softmax_output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    let mut grad_input = Array2::zeros(softmax_output.dim());

    for (i, (sm_row, grad_row)) in softmax_output
        .rows()
        .into_iter()
        .zip(grad_output.rows())
        .enumerate()
    {
        // 计算 sum_j(y_j * ∂L/∂y_j)
        let dot_product: f32 = sm_row
            .iter()
            .zip(grad_row.iter())
            .map(|(&y, &g)| y * g)
            .sum();

        // 计算梯度: y_i * (∂L/∂y_i - dot_product)
        for (j, (&y_val, &g_val)) in sm_row.iter().zip(grad_row.iter()).enumerate() {
            grad_input[[i, j]] = y_val * (g_val - dot_product);
        }
    }

    grad_input
}

/// **多头自注意力机制结构体**
pub struct SelfAttention {
    /// **嵌入维度**: 512（输入/输出的向量维度）
    pub embedding_dim: usize,

    /// **注意力头数**: 8（并行计算的注意力头数量）
    pub num_heads: usize,

    /// **每个头的维度**: 64（512 / 8 = 64）
    pub head_dim: usize,

    // ========== 核心权重矩阵 ==========
    /// **Query 投影矩阵** W_Q: (512, 512)
    /// 将输入转换为查询向量："我在寻找什么信息？"
    pub w_q: Array2<f32>,

    /// **Key 投影矩阵** W_K: (512, 512)
    /// 将输入转换为键向量："我能提供什么信息？"
    pub w_k: Array2<f32>,

    /// **Value 投影矩阵** W_V: (512, 512)
    /// 将输入转换为值向量："我具体包含什么内容？"
    pub w_v: Array2<f32>,

    /// **输出投影矩阵** W_O: (512, 512)
    /// 将多头拼接后的结果投影回原始维度
    pub w_o: Array2<f32>,

    // ========== KV缓存优化（推理加速） ==========
    /// **KV缓存**: (K_cache, V_cache)
    ///
    /// 存储历史 token 的 K 和 V 矩阵，避免重复计算。
    ///
    /// **性能提升示例**：
    /// - 不使用缓存：生成100个token需要 O(100²) = 10,000 次计算
    /// - 使用缓存：生成100个token需要 O(100) = 100 次计算
    /// - **加速比**: 100倍！
    pub kv_cache: Option<(Array2<f32>, Array2<f32>)>,

    /// **是否启用KV缓存**
    /// - true: 推理模式（快速生成）
    /// - false: 训练模式（需要完整梯度）
    pub use_kv_cache: bool,

    /// **是否冻结注意力层参数更新**（用于稳定训练排障）
    pub freeze_updates: bool,

    // ========== 因果掩码缓存（性能优化） ==========
    /// **缓存不同序列长度的因果掩码**
    /// Key: 序列长度, Value: 下三角掩码矩阵
    pub causal_mask_cache: HashMap<usize, Array2<f32>>,

    // ========== Adam 优化器（每个权重矩阵一个） ==========
    pub optimizer_w_q: Adam,
    pub optimizer_w_k: Adam,
    pub optimizer_w_v: Adam,
    pub optimizer_w_o: Adam,

    // =====================================================================
    // 梯度累积支持（Gradient Accumulation）
    // =====================================================================
    //
    // 注意力层的参数比较多（Q/K/V/O 四个矩阵）。为了支持“正确的梯度累积”，我们需要：
    // - micro-batch 级别立即 backward（由 ctx 携带中间量，避免缓存覆盖错误）；
    // - 但把 grad_W_* 累加到 buffer；累积结束再统一做一次 Adam step。
    pub grad_w_q_accum: Array2<f32>,
    pub grad_w_k_accum: Array2<f32>,
    pub grad_w_v_accum: Array2<f32>,
    pub grad_w_o_accum: Array2<f32>,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(EMBEDDING_DIM)
    }
}

impl SelfAttention {
    /// **创建新的多头自注意力层**
    ///
    /// # 参数
    /// - `embedding_dim`: 嵌入维度（通常为512）
    ///
    /// # 架构配置
    /// - **头数**: 8个（Transformer 论文的标准配置）
    /// - **每头维度**: embedding_dim / num_heads = 64
    /// - **总参数量**: 4 × 512² = 1,048,576 参数（Q、K、V、O 四个矩阵）
    ///
    /// # 权重初始化
    /// 使用 He 初始化：std = sqrt(2 / embedding_dim)
    ///
    /// **为什么是 sqrt(2/512) ≈ 0.0625？**
    /// - 保持激活值的方差在层与层之间稳定
    /// - 防止梯度爆炸或消失
    ///
    /// # 示例
    /// ```rust
    /// use llm::self_attention::SelfAttention;
    /// let attention = SelfAttention::new(512);
    /// // 创建 8 头注意力，每头 64 维
    /// assert_eq!(attention.num_heads, 8);
    /// assert_eq!(attention.head_dim, 64);
    /// ```
    pub fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let num_heads = 8; // Transformer 标准：8个注意力头
        let head_dim = embedding_dim / num_heads;

        // 确保维度可以被头数整除
        let (num_heads, head_dim) = if embedding_dim % num_heads != 0 {
            log::warn!(
                "embedding_dim={} 不能被 num_heads={} 整除，回退为单头注意力",
                embedding_dim,
                num_heads
            );
            (1, embedding_dim)
        } else {
            (num_heads, head_dim)
        };

        // He 初始化：std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();

        let w_q = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        let w_k = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        let w_v = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        let w_o = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        SelfAttention {
            embedding_dim,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            kv_cache: None,      // 默认不使用 KV 缓存
            use_kv_cache: false, // 默认训练模式
            freeze_updates: false,
            causal_mask_cache: HashMap::new(), // 初始化掩码缓存
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_o: Adam::new((embedding_dim, embedding_dim)),
            grad_w_q_accum: Array2::zeros((embedding_dim, embedding_dim)),
            grad_w_k_accum: Array2::zeros((embedding_dim, embedding_dim)),
            grad_w_v_accum: Array2::zeros((embedding_dim, embedding_dim)),
            grad_w_o_accum: Array2::zeros((embedding_dim, embedding_dim)),
        }
    }

    pub fn zero_grad_accum(&mut self) {
        self.grad_w_q_accum.fill(0.0);
        self.grad_w_k_accum.fill(0.0);
        self.grad_w_v_accum.fill(0.0);
        self.grad_w_o_accum.fill(0.0);
    }

    /// 计算反向传播所需的梯度（不更新参数）。
    ///
    /// 返回：(grad_input, grad_w_o, grad_w_q, grad_w_k, grad_w_v)
    fn compute_grads_from_ctx(
        &self,
        ctx: &SelfAttentionContext,
        grads: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
        let input = &ctx.input;
        let attention_output = &ctx.attention_output;

        // ========== 步骤1: 输出投影层梯度 ==========
        let grad_w_o = attention_output.t().dot(grads);
        let grad_attention_output = grads.dot(&self.w_o.t());

        // ========== 步骤2: 注意力反传 ==========
        let weights_per_head = ctx.attention_weights.as_ref()?;

        let seq_len = input.nrows();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let sqrt_dk = (head_dim as f32).sqrt();

        // 这些中间 view 必须绑定到局部变量，否则会出现“临时值被释放但仍被借用”的生命周期问题。
        let grad_attention_view = grad_attention_output.view();
        let grad_attention_heads = grad_attention_view
            .to_shape((seq_len, num_heads, head_dim))
            .ok()?
            .permuted_axes([1, 0, 2]);

        let q_view = ctx.q.view();
        let q_heads = q_view
            .to_shape((seq_len, num_heads, head_dim))
            .ok()?
            .permuted_axes([1, 0, 2]);

        let k_view = ctx.k.view();
        let k_heads = k_view
            .to_shape((seq_len, num_heads, head_dim))
            .ok()?
            .permuted_axes([1, 0, 2]);

        let v_view = ctx.v.view();
        let v_heads = v_view
            .to_shape((seq_len, num_heads, head_dim))
            .ok()?
            .permuted_axes([1, 0, 2]);

        let mut grad_q_total = Array2::zeros((seq_len, self.embedding_dim));
        let mut grad_k_total = Array2::zeros((seq_len, self.embedding_dim));
        let mut grad_v_total = Array2::zeros((seq_len, self.embedding_dim));

        for head_idx in 0..num_heads {
            let q_head = q_heads.slice(s![head_idx, .., ..]);
            let k_head = k_heads.slice(s![head_idx, .., ..]);
            let v_head = v_heads.slice(s![head_idx, .., ..]);
            let grad_out_head = grad_attention_heads.slice(s![head_idx, .., ..]);
            let weights = &weights_per_head[head_idx];

            // 梯度 w.r.t. V
            let mut grad_v_head = Array2::zeros((v_head.nrows(), v_head.ncols()));
            general_mat_mul(1.0, &weights.t(), &grad_out_head, 0.0, &mut grad_v_head);

            // 梯度 w.r.t. softmax(weights)
            let mut grad_weights = Array2::zeros((grad_out_head.nrows(), v_head.nrows()));
            general_mat_mul(1.0, &grad_out_head, &v_head.t(), 0.0, &mut grad_weights);
            let grad_scores = stable_softmax_gradient(weights, &grad_weights);

            // 梯度 w.r.t. Q 和 K
            let mut grad_q_head = Array2::zeros((q_head.nrows(), q_head.ncols()));
            general_mat_mul(1.0 / sqrt_dk, &grad_scores, &k_head, 0.0, &mut grad_q_head);

            let mut grad_k_head = Array2::zeros((k_head.nrows(), k_head.ncols()));
            general_mat_mul(
                1.0 / sqrt_dk,
                &grad_scores.t(),
                &q_head,
                0.0,
                &mut grad_k_head,
            );

            let start = head_idx * head_dim;
            let end = start + head_dim;

            grad_q_total
                .slice_mut(s![.., start..end])
                .assign(&grad_q_head);
            grad_k_total
                .slice_mut(s![.., start..end])
                .assign(&grad_k_head);
            grad_v_total
                .slice_mut(s![.., start..end])
                .assign(&grad_v_head);
        }

        // ========== 步骤3: W_q/W_k/W_v 梯度 ==========
        let grad_w_q = input.t().dot(&grad_q_total);
        let grad_w_k = input.t().dot(&grad_k_total);
        let grad_w_v = input.t().dot(&grad_v_total);

        // ========== 步骤4: 输入梯度 ==========
        let grad_input_from_q = grad_q_total.dot(&self.w_q.t());
        let grad_input_from_k = grad_k_total.dot(&self.w_k.t());
        let grad_input_from_v = grad_v_total.dot(&self.w_v.t());
        let grad_input = grad_input_from_q + grad_input_from_k + grad_input_from_v;

        Some((grad_input, grad_w_o, grad_w_q, grad_w_k, grad_w_v))
    }

    /// 用于梯度累积：只累加参数梯度，不更新参数。
    #[deprecated(note = "旧接口依赖层内缓存字段，已废弃；请改用 backward_accumulate_with_ctx(ctx, grads)")]
    pub fn backward_accumulate(&mut self, grads: &Array2<f32>) -> Array2<f32> {
        let _ = grads;
        // 历史接口依赖 `self.cached_*` 保存前向中间量；本轮重构已移除这些缓存字段。
        //
        // 正确姿势：
        // - forward 时保留 ctx：`let (_out, ctx) = attn.forward(&input);`
        // - 累积反传：`attn.backward_accumulate_with_ctx(&ctx, &grads);`
        panic!("SelfAttention.backward_accumulate 已废弃：请改用 backward_accumulate_with_ctx(ctx, grads)")
    }

    /// 用于梯度累积：只累加参数梯度，不更新参数（ctx 驱动，不依赖 cached_*）。
    ///
    /// 教学说明：
    /// - SelfAttention 的 backward 依赖大量中间量（Q/K/V、权重、attention_output 等）；
    /// - 新版 `Layer::forward()` 已经把这些量打包进 `SelfAttentionContext`；
    /// - 因此累积接口也应当接收 ctx，才能避免 cached_* 覆盖并逐步删掉缓存字段。
    pub fn backward_accumulate_with_ctx(
        &mut self,
        ctx: &LayerContext,
        grads: &Array2<f32>,
    ) -> Array2<f32> {
        let Some(ctx) = ctx.downcast_ref::<SelfAttentionContext>() else {
            log::warn!("SelfAttention.backward_accumulate_with_ctx 收到未知 ctx，直接传递梯度");
            return grads.clone();
        };

        let Some((grad_input, grad_w_o, grad_w_q, grad_w_k, grad_w_v)) =
            self.compute_grads_from_ctx(ctx, grads)
        else {
            log::warn!("SelfAttention.backward_accumulate_with_ctx 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        self.grad_w_o_accum += &grad_w_o;
        self.grad_w_q_accum += &grad_w_q;
        self.grad_w_k_accum += &grad_w_k;
        self.grad_w_v_accum += &grad_w_v;

        grad_input
    }

    /// 对累积的梯度执行一次参数更新，并清空累积 buffer。
    pub fn step_accumulated(&mut self, lr: f32, scale: f32) {
        if self.freeze_updates {
            // 即便冻结，也要清空累积，避免后续解除冻结时带入旧梯度。
            self.zero_grad_accum();
            return;
        }

        self.optimizer_w_o
            .step(&mut self.w_o, &(&self.grad_w_o_accum * scale), lr);
        self.optimizer_w_q
            .step(&mut self.w_q, &(&self.grad_w_q_accum * scale), lr);
        self.optimizer_w_k
            .step(&mut self.w_k, &(&self.grad_w_k_accum * scale), lr);
        self.optimizer_w_v
            .step(&mut self.w_v, &(&self.grad_w_v_accum * scale), lr);

        self.zero_grad_accum();
    }

    /// **获取或创建因果掩码**
    ///
    /// 预生成并缓存下三角因果掩码，避免每次forward时逐元素填充。
    ///
    /// # 参数
    /// - `seq_len`: 序列长度
    ///
    /// # 返回值
    /// 因果掩码矩阵 (seq_len, seq_len)，下三角为0，上三角为-∞
    fn get_or_create_causal_mask(&mut self, seq_len: usize) -> &Array2<f32> {
        self.causal_mask_cache.entry(seq_len).or_insert_with(|| {
            let mut mask = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask[[i, j]] = f32::NEG_INFINITY;
                }
            }
            mask
        })
    }

    /// **计算 Q、K、V 矩阵**
    ///
    /// 这是注意力机制的第一步：将输入投影到三个不同的"表示空间"。
    ///
    /// # 计算公式
    /// ```text
    /// Q = X · W_Q  (查询："我要找什么？")
    /// K = X · W_K  (键："我是什么？")
    /// V = X · W_V  (值："我的内容是什么？")
    /// ```
    ///
    /// # 参数
    /// - `input`: 输入张量 (seq_len, 512)
    ///
    /// # 返回值
    /// (Q, K, V) 三个矩阵，形状都是 (seq_len, 512)
    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);
        (q, k, v)
    }

    /// **单头注意力计算（带缓存掩码）**
    ///
    /// 这是注意力机制的核心：通过 Q 和 K 的相似度，对 V 进行加权求和。
    ///
    /// # 算法步骤
    ///
    /// 1. **计算注意力分数**: scores = Q · K^T / √d_k
    ///    - 点积表示相似度
    ///    - 除以√d_k 防止值过大
    ///
    /// 2. **应用因果掩码**: 将未来位置设为 -∞
    ///    - 确保自回归性质（不能看到未来）
    ///    - 使用预缓存的掩码矩阵
    ///
    /// 3. **Softmax 归一化**: weights = softmax(scores)
    ///    - 转换为概率分布（总和为1）
    ///
    /// 4. **加权求和**: output = weights · V
    ///    - 根据注意力权重组合值向量
    ///
    /// # 参数
    /// - `q`: Query 矩阵 (seq_len, head_dim=64)
    /// - `k`: Key 矩阵 (seq_len, head_dim=64)
    /// - `v`: Value 矩阵 (seq_len, head_dim=64)
    /// - `mask`: 因果掩码 (seq_len, seq_len)
    ///
    /// # 返回值
    /// - `output`: 注意力输出 (seq_len, head_dim)
    /// - `weights`: 注意力权重 (seq_len, seq_len)，用于反向传播
    fn attention_with_mask(
        q: ArrayView2<f32>,
        k: ArrayView2<f32>,
        v: ArrayView2<f32>,
        mask: ArrayView2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let dk = (q.ncols() as f32).sqrt();

        // 使用 BLAS 加速的通用矩阵乘法
        let mut scores = Array2::zeros((q.nrows(), k.nrows()));
        general_mat_mul(1.0 / dk, &q, &k.t(), 0.0, &mut scores);

        // 应用掩码（数值稳定版）
        //
        // 为什么不直接做 `scores + mask`？
        // - mask 通常用 `0` 表示“可见”，用 `-∞` 表示“不可见”（因果/Pad 屏蔽）；
        // - 若某些 score 因点积溢出成为 `+∞`，那么 `(+∞) + (-∞)` 会得到 `NaN`，
        //   进而导致 softmax 权重整行被置 0（训练会静默退化）。
        //
        // 因此这里采用“覆盖式掩码”：
        // - mask 为 `-∞` 的位置直接写成 `-∞`；
        // - 其它位置按需叠加（保留将来 mask 不是 0/-∞ 的扩展空间）。
        let mut masked_scores = scores;
        for (ms, &m) in masked_scores.iter_mut().zip(mask.iter()) {
            if m.is_infinite() && m.is_sign_negative() {
                *ms = f32::NEG_INFINITY;
            } else if m != 0.0 {
                *ms += m;
            }
        }

        // 稳定的 softmax
        let weights = stable_softmax(&masked_scores);

        // 计算输出
        let mut output = Array2::zeros((weights.nrows(), v.ncols()));
        general_mat_mul(1.0, &weights, &v, 0.0, &mut output);

        (output, weights)
    }

    /// **旧版单头注意力计算（教学/对照用，当前实现未直接调用）**
    ///
    /// 这是注意力机制的核心：通过 Q 和 K 的相似度，对 V 进行加权求和。
    ///
    /// # 参数
    /// - `q`: Query 矩阵 (seq_len, head_dim=64)
    /// - `k`: Key 矩阵 (seq_len, head_dim=64)
    /// - `v`: Value 矩阵 (seq_len, head_dim=64)
    ///
    /// # 返回值
    /// - `output`: 注意力输出 (seq_len, head_dim)
    /// - `weights`: 注意力权重 (seq_len, seq_len)，用于反向传播
    ///
    /// 说明：
    /// - 当前 SelfAttention 的 forward 实现走的是“多头 +（可选）分块/更稳定的实现路径”，
    ///   因此这里暂时不会被调用；
    /// - 之所以保留，是为了方便读者对照最朴素的“缩放点积注意力（single-head）”公式实现。
    #[allow(dead_code)]
    fn attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // 步骤 1: 计算缩放点积注意力分数
        let dk = (q.ncols() as f32).sqrt(); // √d_k = √64 = 8
        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk; // (seq_len, seq_len)

        // 步骤 2: 应用因果掩码（Causal Mask）
        // 将未来位置设为 -∞，确保模型只能看到过去和当前的信息
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            if i + 1 < seq_len {
                // 使用切片操作一次性设置整行的后续位置为负无穷
                // 例如：位置0只能看自己，位置1可以看0和1，位置2可以看0、1、2
                scores.slice_mut(s![i, i + 1..]).fill(f32::NEG_INFINITY);
            }
        }

        // 步骤 3: Softmax 归一化
        // softmax 将 -∞ 转换为 0，其他值转换为 0-1 之间的概率
        let weights = stable_softmax(&scores);

        // 步骤 4: 使用注意力权重加权 V
        let output = weights.dot(v);

        (output, weights)
    }

    /// **将矩阵重塑为多头格式**
    ///
    /// 这个函数将 (seq_len, 512) 的矩阵转换为多头格式 (seq_len×8, 64)。
    ///
    /// # 转换逻辑
    ///
    /// **输入**: (seq_len, 512) - 一个大的嵌入矩阵
    /// ```text
    /// [向量0: [d0, d1, ..., d511]]
    /// [向量1: [d0, d1, ..., d511]]
    /// ```
    ///
    /// **输出**: (seq_len×8, 64) - 8个头，每个头64维
    /// ```text
    /// Head 0: [向量0的d0-d63, 向量1的d0-d63, ...]
    /// Head 1: [向量0的d64-d127, 向量1的d64-d127, ...]
    /// ...
    /// Head 7: [向量0的d448-d511, 向量1的d448-d511, ...]
    /// ```
    ///
    /// # 示例
    /// ```text
    /// 输入: seq_len=2, embedding_dim=512
    /// [[向量0: 512维], [向量1: 512维]]
    ///
    /// 输出: seq_len×num_heads=16 行，每行 64 维
    /// 行0: 向量0的第0-63维 (Head 0)
    /// 行1: 向量0的第64-127维 (Head 1)
    /// ...
    /// 行7: 向量0的第448-511维 (Head 7)
    /// 行8: 向量1的第0-63维 (Head 0)
    /// ...
    /// ```
    fn reshape_for_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _embedding_dim) = x.dim();

        // 预分配结果矩阵
        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));

        for seq_idx in 0..seq_len {
            let row = x.row(seq_idx);
            for head_idx in 0..self.num_heads {
                let start_dim = head_idx * self.head_dim;
                let end_dim = start_dim + self.head_dim;
                let result_row_idx = seq_idx * self.num_heads + head_idx;

                result
                    .row_mut(result_row_idx)
                    .assign(&row.slice(s![start_dim..end_dim]));
            }
        }

        result
    }

    /// **将多头格式转换回正常矩阵**
    ///
    /// 这是 `reshape_for_heads` 的逆操作，将多头输出拼接回单个大矩阵。
    ///
    /// # 转换逻辑
    ///
    /// **输入**: (seq_len×8, 64) - 8个头的输出
    /// ```text
    /// 行0: 向量0_Head0 (64维)
    /// 行1: 向量0_Head1 (64维)
    /// ...
    /// 行7: 向量0_Head7 (64维)
    /// 行8: 向量1_Head0 (64维)
    /// ...
    /// ```
    ///
    /// **输出**: (seq_len, 512) - 拼接所有头
    /// ```text
    /// 向量0: [Head0的64维 | Head1的64维 | ... | Head7的64维] = 512维
    /// 向量1: [Head0的64维 | Head1的64维 | ... | Head7的64维] = 512维
    /// ```
    fn reverse_reshape_from_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len_times_heads, _head_dim) = x.dim();
        let seq_len = seq_len_times_heads / self.num_heads;

        let mut result = Array2::zeros((seq_len, self.num_heads * self.head_dim));

        for seq_idx in 0..seq_len {
            for head_idx in 0..self.num_heads {
                let src_row_idx = seq_idx * self.num_heads + head_idx;
                let dst_start = head_idx * self.head_dim;
                let dst_end = dst_start + self.head_dim;

                result
                    .slice_mut(s![seq_idx, dst_start..dst_end])
                    .assign(&x.row(src_row_idx));
            }
        }

        result
    }

    /// 多头自注意力的前向传播（优化版：使用缓存掩码与优化矩阵运算）
    ///
    /// # 算法流程
    /// 1. 计算Q、K、V矩阵：Q=XW_q, K=XW_k, V=XW_v
    /// 2. 获取或创建因果掩码（缓存）
    /// 3. 分割为多个注意力头 (num_heads=8)
    /// 4. 对每个头计算：Attention(Q,K,V) = softmax(QK^T/√d_k)V
    /// 5. 拼接所有头的输出
    /// 6. 通过输出投影：output = concat(heads)W_o
    ///
    /// # 优化策略
    /// - 使用引用减少不必要的clone
    /// - 缓存注意力分数和权重用于稳定的梯度计算
    /// - 使用BLAS加速的矩阵乘法
    ///
    /// # 参数
    /// - `input`: 输入张量，形状为 (seq_len, embedding_dim)
    ///
    /// # 返回
    /// 注意力输出，形状与输入相同
    fn multi_head_attention(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.multi_head_attention_with_padding_mask(input, None)
    }

    /// 多头注意力前向传播（支持 padding mask）。
    ///
    /// # 为什么需要 padding mask？
    /// 当我们做 batch 训练时，为了对齐长度会对短序列补 PAD token。
    /// 如果不在注意力里显式屏蔽 PAD，模型会把 PAD 当成真实内容参与注意力计算，产生系统性噪声。
    ///
    /// 这里我们采用“Key padding mask”的做法：
    /// - 对所有 query 位置 i，若 key 位置 j 是 PAD，则把 score(i,j) 设为 -∞；
    /// - 这样 softmax 后该位置权重为 0，从而不会读取 PAD 的 value。
    fn multi_head_attention_with_padding_mask(
        &mut self,
        input: &Array2<f32>,
        key_padding_mask: Option<&Array1<f32>>,
    ) -> Array2<f32> {
        let (_q, _k, _v, attention_output, _weights) =
            self.multi_head_attention_with_padding_mask_core(input, key_padding_mask);
        attention_output.dot(&self.w_o)
    }

    /// 多头注意力前向传播的核心计算：返回 backward 所需的中间量。
    ///
    /// 返回值：
    /// - q/k/v: (seq_len, embedding_dim)
    /// - attention_output: (seq_len, embedding_dim)（拼接多个头之后、输出投影之前）
    /// - attention_weights: 每个 head 的 softmax 权重 (seq_len, seq_len)
    ///
    /// 教学说明：
    /// - 旧实现会把这些中间量写入 `self.cached_*`；
    /// - 本轮重构改为“由调用方持有 ctx”，因此这里直接把值返回给上层组装 ctx。
    fn multi_head_attention_with_padding_mask_core(
        &mut self,
        input: &Array2<f32>,
        key_padding_mask: Option<&Array1<f32>>,
    ) -> (
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<Array2<f32>>,
    ) {
        let (seq_len, _embedding_dim) = input.dim();

        // 1. 计算Q, K, V（使用优化的矩阵乘法）
        let (q, k, v) = self.compute_qkv(input);

        // 2. 构造掩码：causal mask + (可选) padding mask
        let mut mask = self.get_or_create_causal_mask(seq_len).clone();
        if let Some(pad_mask) = key_padding_mask {
            // pad_mask: 1.0=真实token，0.0=PAD
            for (j, &m) in pad_mask.iter().enumerate() {
                if m < 0.5 {
                    // 将该列全部置为 -∞：任何 query 都不能 attend 到这个 key（PAD）
                    mask.slice_mut(s![.., j]).fill(f32::NEG_INFINITY);
                }
            }
        }
        let num_heads = self.num_heads;

        // 3. 分割为多个头
        let q_heads = self.reshape_for_heads(&q);
        let k_heads = self.reshape_for_heads(&k);
        let v_heads = self.reshape_for_heads(&v);

        // 4. 对每个头计算注意力，收集weights用于反向传播
        let mut head_outputs = Vec::with_capacity(num_heads);
        let mut all_weights = Vec::with_capacity(num_heads);

        for head in 0..num_heads {
            let q_head = q_heads.slice(s![head..seq_len * num_heads; num_heads, ..]);
            let k_head = k_heads.slice(s![head..seq_len * num_heads; num_heads, ..]);
            let v_head = v_heads.slice(s![head..seq_len * num_heads; num_heads, ..]);

            // 使用带掩码的注意力计算，返回权重
            let (head_output, weights) =
                Self::attention_with_mask(q_head, k_head, v_head, mask.view());

            head_outputs.push(head_output);
            all_weights.push(weights);
        }

        // 5. 合并所有头
        let mut result = Array2::zeros((seq_len * num_heads, self.head_dim));
        for (head_idx, head_output) in head_outputs.iter().enumerate() {
            for seq_idx in 0..seq_len {
                let row_idx = seq_idx * num_heads + head_idx;
                result.row_mut(row_idx).assign(&head_output.row(seq_idx));
            }
        }

        let combined = self.reverse_reshape_from_heads(&result);
        (q, k, v, combined, all_weights)
    }

    /// 显式暴露“带 padding mask 的前向”，供 batch/教学代码调用。
    pub fn forward_with_padding_mask(
        &mut self,
        input: &Array2<f32>,
        key_padding_mask: Option<&Array1<f32>>,
    ) -> Array2<f32> {
        self.multi_head_attention_with_padding_mask(input, key_padding_mask)
    }

    /// 训练路径：前向传播并返回显式 ctx（owned 输入版本）。
    ///
    /// 为什么要一个 “owned 输入” 的版本？
    /// - SelfAttention 的 ctx 需要保存 `input`，以便 backward 计算 W_q/W_k/W_v 的梯度；
    /// - `Layer::forward(&Array2)` 只能借用输入，因此我们必须在某处 clone 一份输入；
    /// - 在 `forward_batch` 里我们本来就会 `to_owned()` 得到 sample，因此用该接口可以避免二次 clone。
    fn forward_with_padding_mask_owned_and_ctx(
        &mut self,
        input: Array2<f32>,
        key_padding_mask: Option<&Array1<f32>>,
    ) -> (Array2<f32>, SelfAttentionContext) {
        let (q, k, v, attention_output, weights) =
            self.multi_head_attention_with_padding_mask_core(&input, key_padding_mask);
        let out = attention_output.dot(&self.w_o);

        let ctx = SelfAttentionContext {
            input,
            q,
            k,
            v,
            attention_output,
            attention_weights: Some(weights),
        };

        (out, ctx)
    }

    /// 启用KV缓存模式
    pub fn enable_kv_cache(&mut self) {
        self.use_kv_cache = true;
    }

    /// 禁用KV缓存模式并清空缓存
    pub fn disable_kv_cache(&mut self) {
        self.use_kv_cache = false;
        self.kv_cache = None;
    }

    /// 清空KV缓存（保持启用状态）
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    /// 带KV缓存的多头自注意力前向传播
    pub fn forward_with_kv_cache(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if !self.use_kv_cache {
            return self.multi_head_attention(input);
        }

        let (seq_len_new, _embedding_dim) = input.dim();
        let (q_new, k_new, v_new) = self.compute_qkv(input);

        // 合并缓存
        let (k_all, v_all) = if let Some((k_cache, v_cache)) = &self.kv_cache {
            use ndarray::concatenate;
            let k_all = concatenate(Axis(0), &[k_cache.view(), k_new.view()])
                .unwrap_or_else(|_| k_new.clone());
            let v_all = concatenate(Axis(0), &[v_cache.view(), v_new.view()])
                .unwrap_or_else(|_| v_new.clone());
            (k_all, v_all)
        } else {
            (k_new.clone(), v_new.clone())
        };

        self.kv_cache = Some((k_all.clone(), v_all.clone()));

        let total_len = k_all.nrows();
        let num_heads = self.num_heads;

        let mask_full = self.get_or_create_causal_mask(total_len).clone();
        let mask_view = mask_full.slice(s![total_len - seq_len_new.., ..]);

        let q_heads = self.reshape_for_heads(&q_new);
        let k_heads = self.reshape_for_heads(&k_all);
        let v_heads = self.reshape_for_heads(&v_all);

        let mut head_outputs = Vec::with_capacity(num_heads);
        for head in 0..num_heads {
            let q_head = q_heads.slice(s![head..seq_len_new * num_heads; num_heads, ..]);
            let k_head = k_heads.slice(s![head..total_len * num_heads; num_heads, ..]);
            let v_head = v_heads.slice(s![head..total_len * num_heads; num_heads, ..]);

            let (head_output, _weights) =
                Self::attention_with_mask(q_head, k_head, v_head, mask_view.view());
            head_outputs.push(head_output);
        }

        let mut result = Array2::zeros((seq_len_new * num_heads, self.head_dim));
        for (head_idx, head_output) in head_outputs.iter().enumerate() {
            for seq_idx in 0..seq_len_new {
                let row_idx = seq_idx * num_heads + head_idx;
                result.row_mut(row_idx).assign(&head_output.row(seq_idx));
            }
        }

        let combined = self.reverse_reshape_from_heads(&result);
        combined.dot(&self.w_o)
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        // ==========================
        // 前向传播（显式 ctx）
        // ==========================
        //
        // 教学说明：
        // - multi_head_attention 内部会计算 Q/K/V、注意力权重、attention_output 等；
        // - 旧版把这些中间量缓存在 self.cached_*，在 batch 场景会被覆盖；
        // - 新版把中间量打包进 ctx 返回，由调用方在 backward 时显式传回。
        let (out, ctx) = self.forward_with_padding_mask_owned_and_ctx(input.to_owned(), None);
        (out, Box::new(ctx))
    }

    fn forward_batch(
        &mut self,
        input: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
    ) -> (Array3<f32>, Vec<LayerContext>) {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_dim = input.shape()[2];

        let mut output = Array3::zeros((batch_size, seq_len, hidden_dim));
        let mut ctxs: Vec<LayerContext> = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let sample = input.slice(s![b, .., ..]).to_owned();
            let key_padding_mask = attention_mask.map(|m| m.row(b).to_owned());
            let (sample_out, ctx) =
                self.forward_with_padding_mask_owned_and_ctx(sample, key_padding_mask.as_ref());
            output.slice_mut(s![b, .., ..]).assign(&sample_out);
            ctxs.push(Box::new(ctx));
        }

        (output, ctxs)
    }

    fn backward(&mut self, ctx: &LayerContext, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let Some(ctx) = ctx.downcast_ref::<SelfAttentionContext>() else {
            log::warn!("SelfAttention.backward 收到未知 ctx，直接传递梯度");
            return grads.clone();
        };

        let Some((grad_input, grad_w_o, grad_w_q, grad_w_k, grad_w_v)) =
            self.compute_grads_from_ctx(ctx, grads)
        else {
            log::warn!("SelfAttention.backward 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        if !self.freeze_updates {
            self.optimizer_w_o.step(&mut self.w_o, &grad_w_o, lr);
            self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
            self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
            self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);
        }

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w_k.len() + self.w_q.len() + self.w_v.len() + self.w_o.len()
    }

    fn set_training_mode(&mut self, _training: bool) {}
}
