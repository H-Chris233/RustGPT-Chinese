//! 批量训练测试
//!
//! 测试批量训练和动态掩码功能

use llm::{
    batch_loader::*, utils::softmax, Embeddings, LLM, OutputProjection, TransformerBlock, Vocab,
    EMBEDDING_DIM, HIDDEN_DIM, LayerContext,
};
use ndarray::{Array2, Axis};

#[test]
fn test_batch_loader_basic() {
    let sequences = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);

    // 应该创建2个批次
    assert!(batches.len() >= 1);

    // 第一个批次应该有2个样本
    if batches.len() >= 1 {
        assert_eq!(batches[0].batch_size, 2);

        // 检查注意力掩码
        // 第一个样本长度3，全是1.0
        assert_eq!(batches[0].attention_mask[[0, 0]], 1.0);
        assert_eq!(batches[0].attention_mask[[0, 1]], 1.0);
        assert_eq!(batches[0].attention_mask[[0, 2]], 1.0);

        // 第二个样本长度2，前两个是1.0，第三个是0.0（PAD）
        assert_eq!(batches[0].attention_mask[[1, 0]], 1.0);
        assert_eq!(batches[0].attention_mask[[1, 1]], 1.0);
        assert_eq!(batches[0].attention_mask[[1, 2]], 0.0);
    }
}

#[test]
fn test_batch_loader_bucketing() {
    let sequences = vec![
        vec![1, 2],           // 长度2
        vec![3, 4, 5],        // 长度3
        vec![6, 7, 8, 9, 10], // 长度5
        vec![11; 10],         // 长度10
    ];

    let loader = BatchLoader::new(2, true, 8);
    let batches = loader.create_batches(&sequences);

    // 使用分桶策略应该创建合理的批次
    assert!(!batches.is_empty());
}

#[test]
fn test_training_batches_creation() {
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
fn test_attention_mask_with_pad() {
    let sequences = vec![vec![1, 2, 3], vec![4, 5]];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);

    let batch = &batches[0];

    // 验证PAD token被正确填充
    assert_eq!(batch.tokens[[0, 0]], 1);
    assert_eq!(batch.tokens[[0, 1]], 2);
    assert_eq!(batch.tokens[[0, 2]], 3);

    assert_eq!(batch.tokens[[1, 0]], 4);
    assert_eq!(batch.tokens[[1, 1]], 5);
    assert_eq!(batch.tokens[[1, 2]], PAD_TOKEN_ID); // PAD填充

    // 验证注意力掩码正确
    assert_eq!(batch.attention_mask[[0, 0]], 1.0);
    assert_eq!(batch.attention_mask[[0, 1]], 1.0);
    assert_eq!(batch.attention_mask[[0, 2]], 1.0);

    assert_eq!(batch.attention_mask[[1, 0]], 1.0);
    assert_eq!(batch.attention_mask[[1, 1]], 1.0);
    assert_eq!(batch.attention_mask[[1, 2]], 0.0); // PAD位置为0
}

#[test]
fn test_batch_training_with_small_model() {
    // 创建一个小的词汇表（从训练数据构建）
    let training_texts = vec!["你好 世界".to_string(), "测试".to_string()];

    let vocab = Vocab::build_from_texts(&training_texts);

    // 创建一个简单的网络
    let embeddings = Embeddings::new(vocab.clone());
    let transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    let network: Vec<Box<dyn llm::Layer>> = vec![
        Box::new(embeddings),
        Box::new(transformer),
        Box::new(output_projection),
    ];

    let mut model = LLM::new(vocab, network);

    // 准备训练数据
    let data = vec!["你好 世界", "测试"];

    // 使用批量训练（小批次，少epoch）
    let epochs_trained = model.train_bucketed_sequential(
        data, 2,     // max_epochs
        0.001, // initial_lr
        10,    // patience
        2,     // batch_size
    );

    // 验证训练完成
    assert!(epochs_trained > 0);
    assert!(epochs_trained <= 2);
}

#[test]
fn test_pad_token_gradient_masking() {
    // 测试PAD位置的梯度是否被正确屏蔽
    let sequences = vec![
        vec![1, 2, 3],
        vec![4, 5], // 这个会被PAD到长度3
    ];

    let loader = BatchLoader::new(2, false, 8);
    let batches = loader.create_batches(&sequences);
    let batch = &batches[0];

    // 验证PAD位置的掩码
    assert_eq!(batch.attention_mask[[1, 2]], 0.0, "PAD位置掩码应该为0");

    // 在实际训练中，PAD位置的梯度应该被清零
    // 这里只是验证掩码的正确性
}

#[derive(Clone)]
struct BiasOnlyProbeLayer {
    bias: Array2<f32>,
    seen_input_lengths: Vec<usize>,
}

impl BiasOnlyProbeLayer {
    fn new(vocab_size: usize) -> Self {
        Self {
            bias: Array2::zeros((1, vocab_size)),
            seen_input_lengths: Vec::new(),
        }
    }
}

impl llm::Layer for BiasOnlyProbeLayer {
    fn layer_type(&self) -> &str {
        "BiasOnlyProbeLayer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        let seq_len = input.shape()[1];
        self.seen_input_lengths.push(seq_len);
        let mut out = Array2::zeros((seq_len, self.bias.ncols()));
        for row in 0..seq_len {
            out.row_mut(row).assign(&self.bias.row(0));
        }
        (out, Box::new(()))
    }

    fn backward(&mut self, _ctx: &LayerContext, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let grad_b = grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        self.bias -= &(lr * &grad_b);
        Array2::zeros((1, grads.nrows()))
    }

    fn parameters(&self) -> usize {
        self.bias.len()
    }

    fn set_training_mode(&mut self, _training: bool) {}
}

fn build_probe_model(texts: &[String]) -> LLM {
    let vocab = Vocab::build_from_texts(texts);
    let vocab_size = vocab.len();
    LLM::new(vocab, vec![Box::new(BiasOnlyProbeLayer::new(vocab_size))])
}

fn probe_bias(model: &LLM) -> Array2<f32> {
    model.network[0]
        .as_any()
        .downcast_ref::<BiasOnlyProbeLayer>()
        .expect("expected BiasOnlyProbeLayer")
        .bias
        .clone()
}

fn probe_seen_lengths(model: &LLM) -> Vec<usize> {
    model.network[0]
        .as_any()
        .downcast_ref::<BiasOnlyProbeLayer>()
        .expect("expected BiasOnlyProbeLayer")
        .seen_input_lengths
        .clone()
}

fn compute_probe_loss(model: &mut LLM, text: &str) -> f32 {
    let tokens = LLM::tokenize_training_with_vocab(&model.vocab, text);
    let input_ids = &tokens[..tokens.len() - 1];
    let target_ids = &tokens[1..];

    let mut input = Array2::from_shape_fn((1, input_ids.len()), |(_, j)| input_ids[j] as f32);
    for layer in &mut model.network {
        let (out, _ctx) = layer.forward(&input);
        input = out;
    }

    let probs = softmax(&input);
    LLM::cross_entropy_loss_step(&probs, target_ids, model.vocab.pad_token_id())
}

#[test]
fn test_bucketed_sequential_order_affects_parameters() {
    let texts = vec!["a".to_string(), "b".to_string()];

    let mut model_ab = build_probe_model(&texts);
    let mut model_ba = build_probe_model(&texts);

    let epochs_ab = model_ab.train_bucketed_sequential(vec!["a", "b"], 1, 0.5, 10, 2);
    let epochs_ba = model_ba.train_bucketed_sequential(vec!["b", "a"], 1, 0.5, 10, 2);

    assert_eq!(epochs_ab, 1);
    assert_eq!(epochs_ba, 1);

    let bias_ab = probe_bias(&model_ab);
    let bias_ba = probe_bias(&model_ba);
    let max_diff = bias_ab
        .iter()
        .zip(bias_ba.iter())
        .fold(0.0_f32, |m, (&x, &y)| m.max((x - y).abs()));

    assert!(
        max_diff > 1e-5,
        "reversing sample order should change parameters because updates are sequential, got max_diff={}",
        max_diff
    );
}

#[test]
fn test_bucketed_sequential_improves_probe_loss() {
    let texts = vec!["a".to_string()];
    let mut model = build_probe_model(&texts);

    let loss_before = compute_probe_loss(&mut model, "a");
    let epochs = model.train_bucketed_sequential(vec!["a"], 6, 0.4, 20, 2);
    let loss_after = compute_probe_loss(&mut model, "a");

    assert!(epochs > 0);
    assert!(
        loss_after < loss_before,
        "expected training to reduce probe loss, before={}, after={}",
        loss_before,
        loss_after
    );
}

#[test]
fn test_bucketed_sequential_short_samples_are_forwarded_without_pad_tokens() {
    let texts = vec!["a b c".to_string(), "d".to_string()];
    let mut model = build_probe_model(&texts);

    let epochs = model.train_bucketed_sequential(vec!["a b c", "d"], 1, 0.2, 10, 2);
    assert_eq!(epochs, 1);

    let mut seen = probe_seen_lengths(&model);
    seen.sort_unstable();

    // `a b c` => [BOS, a, b, c, EOS]，input_len=4；`d` => [BOS, d, EOS]，input_len=2。
    assert_eq!(seen, vec![2, 4]);
}
