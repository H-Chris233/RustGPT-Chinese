use llm::{EMBEDDING_DIM, HIDDEN_DIM, Layer, transformer::TransformerBlock};
use ndarray::Array2;

#[test]
fn test_transformer_block() {
    let mut transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);

    // 构造一个简单输入张量。
    let input = Array2::ones((1, EMBEDDING_DIM));

    // 执行前向传播。
    let (output, _ctx) = transformer.forward(&input);

    // 输出形状应与输入匹配。
    assert_eq!(output.shape(), [1, EMBEDDING_DIM]);
}
