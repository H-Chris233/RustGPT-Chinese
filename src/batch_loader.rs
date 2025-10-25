//! # 批量数据加载器（Batch Loader）
//!
//! 该模块负责将可变长度的序列数据组织成批次，并提供动态填充（PAD）和注意力掩码。
//!
//! ## 核心功能
//!
//! 1. **数据分桶（Bucketing）**：将长度相近的序列分组，减少填充开销
//! 2. **动态填充（Dynamic Padding）**：每个批次填充到该批次的最大长度
//! 3. **注意力掩码（Attention Mask）**：标记真实token和PAD，确保PAD不参与梯度
//!
//! ## 批处理优势
//!
//! - **并行计算**：同时处理多个样本，提高GPU/CPU利用率
//! - **训练稳定性**：批量统计更稳定，减少梯度噪声
//! - **内存效率**：分桶策略减少不必要的填充
//!
//! ## 数据格式
//!
//! ### 输入
//! ```text
//! Vec<Vec<usize>>: 多个tokenized序列
//! [[12, 45, 78],           // 长度3
//!  [23, 67, 89, 12],       // 长度4
//!  [45, 90]]               // 长度2
//! ```
//!
//! ### 输出（Batch）
//! ```text
//! tokens: Array2<usize> shape (batch_size, max_seq_len)
//! [[12, 45, 78, 0],        // 填充到长度4
//!  [23, 67, 89, 12],       // 无需填充
//!  [45, 90, 0,  0]]        // 填充到长度4
//!
//! mask: Array2<f32> shape (batch_size, max_seq_len)
//! [[1.0, 1.0, 1.0, 0.0],   // 最后一个是PAD
//!  [1.0, 1.0, 1.0, 1.0],   // 全是真实token
//!  [1.0, 1.0, 0.0, 0.0]]   // 后两个是PAD
//! ```

use ndarray::Array2;
use std::collections::HashMap;

/// **PAD token ID**（与 vocab.rs 保持一致）
pub const PAD_TOKEN_ID: usize = 0;

/// **批次数据结构**
#[derive(Debug, Clone)]
pub struct Batch {
    /// **Token IDs**: (batch_size, seq_len)
    pub tokens: Array2<usize>,

    /// **注意力掩码**: (batch_size, seq_len)
    /// - 1.0 表示真实 token
    /// - 0.0 表示 PAD token
    pub attention_mask: Array2<f32>,

    /// **批次大小**
    pub batch_size: usize,

    /// **序列长度**（该批次的最大长度）
    pub seq_len: usize,
}

impl Batch {
    /// 创建新的批次
    pub fn new(tokens: Array2<usize>, attention_mask: Array2<f32>) -> Self {
        let batch_size = tokens.nrows();
        let seq_len = tokens.ncols();
        Self {
            tokens,
            attention_mask,
            batch_size,
            seq_len,
        }
    }

    /// 将 token IDs 转换为 f32 Array2（供 Embeddings 使用）
    pub fn tokens_as_f32(&self) -> Array2<f32> {
        self.tokens.mapv(|x| x as f32)
    }
}

/// **批量数据加载器**
pub struct BatchLoader {
    /// 批次大小
    pub batch_size: usize,

    /// 是否启用分桶策略
    pub use_bucketing: bool,

    /// 分桶的长度阈值（序列长度差异在此范围内的分到同一桶）
    pub bucket_width: usize,
}

impl BatchLoader {
    /// 创建新的批量加载器
    ///
    /// # 参数
    /// - `batch_size`: 批次大小
    /// - `use_bucketing`: 是否启用分桶策略（推荐 true）
    /// - `bucket_width`: 分桶宽度（推荐 8-16）
    pub fn new(batch_size: usize, use_bucketing: bool, bucket_width: usize) -> Self {
        Self {
            batch_size,
            use_bucketing,
            bucket_width,
        }
    }

    /// 将数据划分为多个批次
    ///
    /// # 参数
    /// - `sequences`: tokenized 序列列表
    ///
    /// # 返回值
    /// 批次列表
    pub fn create_batches(&self, sequences: &[Vec<usize>]) -> Vec<Batch> {
        if sequences.is_empty() {
            return Vec::new();
        }

        if self.use_bucketing {
            self.create_bucketed_batches(sequences)
        } else {
            self.create_simple_batches(sequences)
        }
    }

    /// 简单批处理（不分桶）
    fn create_simple_batches(&self, sequences: &[Vec<usize>]) -> Vec<Batch> {
        let mut batches = Vec::new();
        let mut i = 0;

        while i < sequences.len() {
            let end = (i + self.batch_size).min(sequences.len());
            let batch_sequences = &sequences[i..end];

            // 找到该批次的最大长度
            let max_len = batch_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

            if max_len == 0 {
                i = end;
                continue;
            }

            let batch = self.pad_and_create_batch(batch_sequences, max_len);
            batches.push(batch);

            i = end;
        }

        batches
    }

    /// 分桶批处理（按长度分桶）
    fn create_bucketed_batches(&self, sequences: &[Vec<usize>]) -> Vec<Batch> {
        // 按长度分桶
        let mut buckets: HashMap<usize, Vec<&Vec<usize>>> = HashMap::new();

        for seq in sequences {
            let bucket_id = (seq.len() / self.bucket_width) * self.bucket_width;
            buckets.entry(bucket_id).or_default().push(seq);
        }

        // 为每个桶创建批次
        let mut batches = Vec::new();

        for (_bucket_id, mut bucket_sequences) in buckets {
            // 按序列长度排序（可选，但有助于进一步减少填充）
            bucket_sequences.sort_by_key(|s| s.len());

            let mut i = 0;
            while i < bucket_sequences.len() {
                let end = (i + self.batch_size).min(bucket_sequences.len());
                let batch_sequences: Vec<Vec<usize>> = bucket_sequences[i..end]
                    .iter()
                    .map(|&s| s.clone())
                    .collect();

                let max_len = batch_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

                if max_len > 0 {
                    let batch = self.pad_and_create_batch(&batch_sequences, max_len);
                    batches.push(batch);
                }

                i = end;
            }
        }

        batches
    }

    /// 填充并创建批次
    fn pad_and_create_batch(&self, sequences: &[Vec<usize>], max_len: usize) -> Batch {
        let batch_size = sequences.len();

        // 初始化 tokens 和 mask
        let mut tokens = Array2::from_elem((batch_size, max_len), PAD_TOKEN_ID);
        let mut attention_mask = Array2::zeros((batch_size, max_len));

        // 填充每个序列
        for (i, seq) in sequences.iter().enumerate() {
            let seq_len = seq.len();

            // 复制真实 token
            for (j, &token) in seq.iter().enumerate() {
                tokens[[i, j]] = token;
            }

            // 设置注意力掩码（1.0 表示真实 token）
            for j in 0..seq_len {
                attention_mask[[i, j]] = 1.0;
            }
        }

        Batch::new(tokens, attention_mask)
    }
}

/// **为训练数据创建批次（teacher forcing 模式）**
///
/// 在训练时，我们需要 input 和 target：
/// - input: tokens[:-1]（除了最后一个token）
/// - target: tokens[1:]（除了第一个token）
///
/// # 参数
/// - `batch_loader`: 批量加载器
/// - `sequences`: tokenized 序列
///
/// # 返回值
/// (input_batches, target_batches) - 输入和目标批次对
pub fn create_training_batches(
    batch_loader: &BatchLoader,
    sequences: &[Vec<usize>],
) -> Vec<(Batch, Vec<Vec<usize>>)> {
    let batches = batch_loader.create_batches(sequences);
    let mut result = Vec::new();

    for batch in batches {
        // 为每个样本创建 target（右移一位）
        let mut targets = Vec::new();

        for i in 0..batch.batch_size {
            // 获取该样本的真实长度（通过 attention_mask）
            let seq_len = batch
                .attention_mask
                .row(i)
                .iter()
                .filter(|&&x| x > 0.0)
                .count();

            if seq_len > 1 {
                // target 是 tokens[1:seq_len]
                let target: Vec<usize> = (1..seq_len).map(|j| batch.tokens[[i, j]]).collect();
                targets.push(target);
            } else {
                // 序列太短，跳过
                targets.push(Vec::new());
            }
        }

        // 创建 input batch（移除最后一列）
        let input_seq_len = batch.seq_len.saturating_sub(1);
        if input_seq_len == 0 {
            continue;
        }

        let mut input_tokens = Array2::from_elem((batch.batch_size, input_seq_len), PAD_TOKEN_ID);
        let mut input_mask = Array2::zeros((batch.batch_size, input_seq_len));

        for i in 0..batch.batch_size {
            for j in 0..input_seq_len {
                input_tokens[[i, j]] = batch.tokens[[i, j]];
                input_mask[[i, j]] = batch.attention_mask[[i, j]];
            }
        }

        let input_batch = Batch::new(input_tokens, input_mask);
        result.push((input_batch, targets));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_batching() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];

        let loader = BatchLoader::new(2, false, 8);
        let batches = loader.create_batches(&sequences);

        // 应该创建2个批次：[seq0, seq1] 和 [seq2]
        assert_eq!(batches.len(), 2);

        // 第一个批次：长度应该是3（max(3, 2)）
        assert_eq!(batches[0].batch_size, 2);
        assert_eq!(batches[0].seq_len, 3);

        // 第二个批次：长度应该是4
        assert_eq!(batches[1].batch_size, 1);
        assert_eq!(batches[1].seq_len, 4);

        // 检查注意力掩码
        assert_eq!(batches[0].attention_mask[[0, 0]], 1.0);
        assert_eq!(batches[0].attention_mask[[1, 2]], 0.0); // PAD
    }

    #[test]
    fn test_bucketing() {
        let sequences = vec![
            vec![1, 2],           // 长度2 -> bucket 0
            vec![3, 4, 5],        // 长度3 -> bucket 0
            vec![6, 7, 8, 9, 10], // 长度5 -> bucket 0
            vec![11; 10],         // 长度10 -> bucket 8
        ];

        let loader = BatchLoader::new(2, true, 8);
        let batches = loader.create_batches(&sequences);

        // 应该有2个桶
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_training_batches() {
        let sequences = vec![vec![1, 2, 3, 4], vec![5, 6, 7]];

        let loader = BatchLoader::new(2, false, 8);
        let training_batches = create_training_batches(&loader, &sequences);

        assert_eq!(training_batches.len(), 1);

        let (input_batch, targets) = &training_batches[0];

        // Input 应该是 tokens[:-1]，所以长度是3
        assert_eq!(input_batch.seq_len, 3);

        // Target 应该是 tokens[1:]
        assert_eq!(targets[0], vec![2, 3, 4]);
        assert_eq!(targets[1], vec![6, 7]);
    }

    #[test]
    fn test_attention_mask() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5]];

        let loader = BatchLoader::new(2, false, 8);
        let batches = loader.create_batches(&sequences);

        let batch = &batches[0];

        // 第一个序列：全1
        assert_eq!(batch.attention_mask[[0, 0]], 1.0);
        assert_eq!(batch.attention_mask[[0, 1]], 1.0);
        assert_eq!(batch.attention_mask[[0, 2]], 1.0);

        // 第二个序列：前两个是1，最后一个是0（PAD）
        assert_eq!(batch.attention_mask[[1, 0]], 1.0);
        assert_eq!(batch.attention_mask[[1, 1]], 1.0);
        assert_eq!(batch.attention_mask[[1, 2]], 0.0);
    }
}
