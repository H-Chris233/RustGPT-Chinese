// 模型保存和加载示例程序
//
// 使用方法：
// 1. 训练并保存: cargo run --example model_persistence -- save
// 2. 加载并使用: cargo run --example model_persistence -- load
// 3. 从检查点继续训练: cargo run --example model_persistence -- continue

use llm::{
    Dataset, EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM, OutputProjection, TransformerBlock, Vocab,
    load_model_binary, save_model_binary,
};
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    if let Err(e) = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
    {
        eprintln!("日志初始化失败: {}", e);
    }

    // 检查命令行参数
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "save" => train_and_save()?,
            "load" => load_and_use()?,
            "continue" => continue_training()?,
            _ => print_usage(),
        }
    } else {
        print_usage();
    }

    Ok(())
}

fn print_usage() {
    println!(
        "
╔═══════════════════════════════════════════════════════════╗
║         RustGPT-Chinese 模型保存/加载工具                 ║
╚═══════════════════════════════════════════════════════════╝

使用方法:
  cargo run --example model_persistence -- save       # 训练并保存模型
  cargo run --example model_persistence -- load       # 加载并使用模型
  cargo run --example model_persistence -- continue   # 从检查点继续训练

示例:
  # 训练100个epoch并保存
  cargo run --example model_persistence -- save

  # 加载模型并进行对话
  cargo run --example model_persistence -- load

  # 从检查点继续训练 50 个 epoch
  cargo run --example model_persistence -- continue
"
    );
}

/// 训练模型并保存
fn train_and_save() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 开始训练模型...\n");

    // 1. 加载数据
    println!("📂 加载训练数据...");
    let dataset = Dataset::new(
        String::from("data/pretraining"),
        String::from("data/chat"),
    );

    // 2. 构建词汇表
    println!("📝 构建词汇表...");
    let mut vocab_set = HashSet::new();
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    println!("✓ 词汇表创建完成: {} 个词元\n", vocab.len());

    // 3. 创建模型
    println!("🏗️  初始化模型...");
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_4 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());

    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(transformer_block_4),
            Box::new(output_projection),
        ],
    );

    println!("✓ 模型初始化完成");
    println!("  • 总参数量: {}", llm.total_parameters());
    println!("  • 网络架构: {}\n", llm.network_description());

    // 4. 预训练
    println!("🎯 阶段1: 预训练 (100 epochs, lr=0.0005)");
    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train(pretraining_examples, 100, 0.0005);

    // 保存预训练阶段的检查点。
    println!("\n💾 保存预训练checkpoint...");
    std::fs::create_dir_all("checkpoints")?;
    save_model_binary(&llm, "checkpoints/model_pretrained.bin")?;

    // 5. 指令微调
    println!("\n🎯 阶段2: 指令微调 (100 epochs, lr=0.0001)");
    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train(chat_training_examples, 100, 0.0001);

    // 6. 保存最终模型
    println!("\n💾 保存最终模型...");
    save_model_binary(&llm, "checkpoints/model_final.bin")?;

    println!("\n✅ 训练完成!");
    println!("   模型已保存到:");
    println!("   • checkpoints/model_pretrained.bin (预训练checkpoint)");
    println!("   • checkpoints/model_final.bin (最终模型)");
    println!("\n💡 提示: 使用 'cargo run --example model_persistence -- load' 加载模型\n");

    Ok(())
}

/// 加载模型并使用
fn load_and_use() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📂 加载模型...\n");

    // 加载模型
    let mut llm = load_model_binary("checkpoints/model_final.bin")?;
    llm.set_training_mode(false);

    println!("\n✅ 模型加载成功!");
    println!("   • 词汇量: {}", llm.vocab.len());
    println!("   • 总参数: {}", llm.total_parameters());

    // 测试对话
    println!("\n--- 进入交互模式 ---");
    println!("输入问题按回车生成回答,输入 'exit' 退出\n");

    let mut input = String::new();
    loop {
        input.clear();

        print!("👤 用户: ");
        if let Err(e) = std::io::stdout().flush() {
            log::warn!("刷新标准输出失败: {}", e);
        }

        if let Err(e) = std::io::stdin().read_line(&mut input) {
            log::warn!("读取输入失败: {}", e);
            continue;
        }

        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("👋 再见!");
            break;
        }

        let formatted_input = format!("用户：{}", trimmed_input);
        print!("🤖 模型: ");
        if let Err(e) = std::io::stdout().flush() {
            log::warn!("刷新标准输出失败: {}", e);
        }

        let prediction = llm.predict_with_beam_search(&formatted_input, 3, 20);
        println!("{}\n", prediction);

        // 检测到结束符时清空上下文
        if prediction.contains("</s>") {
            llm.clear_context();
        }
    }

    Ok(())
}

/// 从检查点继续训练
fn continue_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📂 从checkpoint继续训练...\n");

    // 1. 加载检查点
    println!("加载预训练checkpoint...");
    let mut llm = load_model_binary("checkpoints/model_pretrained.bin")?;
    llm.set_training_mode(true);

    println!("✓ Checkpoint加载成功\n");

    // 2. 加载数据
    println!("📂 加载训练数据...");
    let dataset = Dataset::new(
        String::from("data/pretraining"),
        String::from("data/chat"),
    );

    // 3. 继续训练
    println!("\n🎯 继续指令微调 (50 epochs, lr=0.0001)");
    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train(chat_training_examples, 50, 0.0001);

    // 4. 保存新模型
    println!("\n💾 保存继续训练后的模型...");
    save_model_binary(&llm, "checkpoints/model_continued.bin")?;

    println!("\n✅ 继续训练完成!");
    println!("   模型已保存到: checkpoints/model_continued.bin\n");

    Ok(())
}

use std::io::Write;
