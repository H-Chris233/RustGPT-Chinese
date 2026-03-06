use llm::{
    EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM, Layer, Vocab, output_projection::OutputProjection,
    transformer::TransformerBlock,
};
use llm::LayerContext;
use ndarray::Array2;

struct TestOutputProjectionLayer {
    pub cache_input: Option<Array2<f32>>,
    pub loop_count: usize,
    pub stop_index: usize,
    pub stop_loop_count: usize,
    pub vocab_size: usize,
    pub cached_grads: Option<Array2<f32>>,
}

impl Layer for TestOutputProjectionLayer {
    fn layer_type(&self) -> &str {
        "TestOutputProjectionLayer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, LayerContext) {
        // 教学说明：
        // - 该测试 layer 需要在 backward 中使用 forward 的输入；
        // - 新版 Layer trait 使用 ctx 显式传递中间量；
        // - 因此我们把 input 克隆后放入 ctx，同时也保留到 self.cache_input（兼容旧逻辑）。
        self.cache_input = Some(input.clone());
        let mut mock_output = Array2::zeros((input.shape()[1], self.vocab_size));

        let last_token_index = input.shape()[1] - 1;

        // 循环达到阈值后，强制输出结束词对应的位置。
        if self.loop_count >= self.stop_loop_count {
            mock_output[[last_token_index, self.stop_index]] = 1.0;
        } else {
            mock_output[[last_token_index, 0]] = 1.0;
        }

        self.loop_count += 1;
        (mock_output, Box::new(input.clone()))
    }

    // 该测试层当前只需覆盖基础 backward 行为。
    fn backward(
        &mut self,
        ctx: &LayerContext,
        grads: &Array2<f32>,
        _lr: f32,
    ) -> Array2<f32> {
        let input = if let Some(input) = ctx.downcast_ref::<Array2<f32>>() {
            input
        } else if let Some(input) = self.cache_input.as_ref() {
            // fallback：若 ctx 类型不匹配，仍尽量用旧缓存（测试层容错）
            input
        } else {
            // 如果没有可用的 forward 输入，就直接透传梯度。
            return grads.clone();
        };

        // 使用链式法则把梯度传回输入侧。
        let grad_input = input.dot(grads);
        self.cached_grads = Some(grad_input.clone());

        grad_input
    }

    fn parameters(&self) -> usize {
        const NUM_PARAMETERS_TEST_LAYER: usize = 0;
        NUM_PARAMETERS_TEST_LAYER
    }
}

impl TestOutputProjectionLayer {
    pub fn new(stop_index: usize, stop_loop_count: usize, vocab_size: usize) -> Self {
        TestOutputProjectionLayer {
            cache_input: None,
            loop_count: 0,
            stop_index,
            stop_loop_count,
            vocab_size,
            cached_grads: None,
        }
    }
}

#[test]
fn test_llm_tokenize() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let llm = LLM::new(
        vocab,
        vec![Box::new(TestOutputProjectionLayer::new(5, 5, vocab_size))],
    );

    // 验证分词结果非空。
    let tokens = llm.tokenize("hello world");
    assert!(!tokens.is_empty());

    // 验证 token 可以解码回词表项。
    for token in tokens {
        assert!(llm.vocab.decode(token).is_some());
    }
}

#[test]
fn test_llm_predict() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let mut llm = LLM::new(
        vocab.clone(),
        vec![Box::new(TestOutputProjectionLayer::new(5, 5, vocab_size))],
    );

    // 验证预测流程能正常运行。
    let input_text = "hello world this is rust";
    let result = llm.predict(input_text);
    assert!(!result.is_empty());

    // 这里只验证结果非空且包含预期结束词。
    assert!(result.contains("</s>"));
}

#[test]
fn test_llm_train() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();
    let layer = Box::new(TestOutputProjectionLayer::new(5, 1, vocab_size));
    let mut llm = LLM::new(vocab.clone(), vec![layer]);

    let training_data = vec!["hello world this is rust."];

    llm.train(training_data, 10, 0.01);
}

#[test]
fn test_llm_integration() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();

    let embeddings = Box::new(Embeddings::new(vocab.clone()));
    let output_projection = Box::new(OutputProjection::new(EMBEDDING_DIM, vocab_size));

    let mut llm = LLM::new(vocab.clone(), vec![embeddings, output_projection]);

    let input_text = "hello world this is rust";
    llm.train(vec![input_text], 10, 0.01);
}

#[test]
fn test_llm_total_parameters() {
    let vocab = Vocab::default();
    let vocab_size = vocab.encode.len();

    // 用真实层组合一个 LLM，以得到有意义的参数量统计。
    let embeddings = Box::new(Embeddings::new(vocab.clone()));
    let transformer_block = Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM));
    let output_projection = Box::new(OutputProjection::new(EMBEDDING_DIM, vocab_size));

    let llm = LLM::new(
        vocab.clone(),
        vec![embeddings, transformer_block, output_projection],
    );

    // 含真实层的模型参数量应大于 0。
    let param_count = llm.total_parameters();
    assert!(param_count > 0);

    // 总参数量应等于各层参数量之和。
    let embeddings_param_count = LLM::new(
        vocab.clone(),
        vec![Box::new(Embeddings::new(vocab.clone()))],
    )
    .total_parameters();
    let transformer_param_count = LLM::new(
        vocab.clone(),
        vec![Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM))],
    )
    .total_parameters();
    let output_param_count = LLM::new(
        vocab.clone(),
        vec![Box::new(OutputProjection::new(EMBEDDING_DIM, vocab_size))],
    )
    .total_parameters();

    assert_eq!(
        param_count,
        embeddings_param_count + transformer_param_count + output_param_count
    );
}
