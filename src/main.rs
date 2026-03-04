use std::io::Write;

// 从lib.rs导入所有需要的类型和常量
use llm::{
    CheckpointManager, CheckpointStrategy, Dataset, EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM,
    MAX_SEQ_LEN, OutputProjection, PerformanceMonitor, TransformerBlock, Vocab, load_model_binary,
    save_model_binary, save_model_json,
};

// 🔥 导入训练性能优化模块

// CLI 解析辅助函数
fn arg_has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_usize_arg(args: &[String], key: &str) -> Option<usize> {
    let prefix = format!("{}=", key);
    for a in args {
        if a.starts_with(&prefix) {
            if let Ok(v) = a[prefix.len()..].parse::<usize>() {
                return Some(v);
            }
        }
    }
    None
}

fn parse_f32_arg(args: &[String], key: &str) -> Option<f32> {
    let prefix = format!("{}=", key);
    for a in args {
        if a.starts_with(&prefix) {
            if let Ok(v) = a[prefix.len()..].parse::<f32>() {
                return Some(v);
            }
        }
    }
    None
}

fn parse_string_arg(args: &[String], key: &str) -> Option<String> {
    let prefix = format!("{}=", key);
    for a in args {
        if a.starts_with(&prefix) {
            return Some(a[prefix.len()..].to_string());
        }
    }
    None
}

// 快速预训练入口（非交互短跑）
fn run_quick(
    perf_monitor: &mut PerformanceMonitor,
    freeze_attn: bool,
    pretrain_epochs: usize,
    lr: f32,
    patience: usize,
    accum: usize,
) {
    println!("\n⚡ 启动快速预训练 (--quick) 模式");

    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from("data/pretraining"),
        String::from("data/chat"),
    );
    perf_monitor.stop("加载训练数据");

    // 构建词汇表
    let mut vocab_set = std::collections::HashSet::new();

    perf_monitor.start("构建词汇表 - 预训练数据");
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    perf_monitor.stop("构建词汇表 - 预训练数据");

    perf_monitor.start("构建词汇表 - 对话数据");
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);
    perf_monitor.stop("构建词汇表 - 对话数据");

    perf_monitor.start("创建词汇表对象");
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);
    perf_monitor.stop("创建词汇表对象");

    // 初始化模型
    perf_monitor.start("初始化神经网络");
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());

    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(output_projection),
        ],
    );
    perf_monitor.stop("初始化神经网络");

    // 可选冻结注意力参数更新
    if freeze_attn {
        llm.set_attention_freeze_updates(true);
        println!("🔒 注意力层参数更新已冻结 (--freeze-attn)");
    }

    // 预训练
    println!(
        "\n[Quick] 预训练: epochs={}, lr={:.6}, patience={}, accum={} (cosine, 无重启, clip=1.0)",
        pretrain_epochs, lr, patience, accum
    );

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("预训练阶段");
    let actual_epochs =
        llm.train_monitored(pretraining_examples, pretrain_epochs, lr, patience, accum);
    perf_monitor.stop("预训练阶段");

    println!("✓ 快速预训练完成，实际训练 {} epochs", actual_epochs);

    perf_monitor.stop("程序总执行时间");
    perf_monitor.print_report();
}

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║          RustGPT-Chinese - 中文GPT模型训练系统            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // 初始化日志系统
    if let Err(e) = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
    {
        eprintln!("日志初始化失败: {}", e);
    }

    // 创建性能监控器
    let mut perf_monitor = PerformanceMonitor::new();
    perf_monitor.start("程序总执行时间");

    // 解析命令行参数
    let args: Vec<String> = std::env::args().skip(1).collect();
    let freeze_attn = arg_has_flag(&args, "--freeze-attn");
    let no_interactive = arg_has_flag(&args, "--no-interactive");

    // 🔄 Resume 训练入口：从检查点恢复训练
    if arg_has_flag(&args, "--resume") {
        let checkpoint_path = parse_string_arg(&args, "--resume-from").or_else(|| {
            // 如果未指定路径，尝试加载最佳或最新检查点
            let checkpoint_dir = parse_string_arg(&args, "--checkpoint-dir")
                .unwrap_or_else(|| "checkpoints".to_string());

            if let Ok(manager) =
                CheckpointManager::new(&checkpoint_dir, CheckpointStrategy::Best, 3)
            {
                if let Some(best) = manager.get_best_checkpoint() {
                    return Some(best.to_string_lossy().to_string());
                }
                if let Some(last) = manager.get_last_checkpoint() {
                    return Some(last.to_string_lossy().to_string());
                }
            }
            None
        });

        if let Some(path) = checkpoint_path {
            println!("\n🔄 从检查点恢复训练: {}", path);
            match CheckpointManager::load_checkpoint(&path) {
                Ok((mut llm, metadata)) => {
                    println!("\n✅ 检查点加载成功!");
                    println!("   • 训练阶段: {}", metadata.phase);
                    println!("   • Epoch: {}", metadata.epoch);
                    println!("   • Loss: {:.4}", metadata.loss);
                    println!("   • 学习率: {:.6}", metadata.learning_rate);
                    println!("   • 时间戳: {}", metadata.timestamp);
                    println!("   • 词汇量: {}", llm.vocab.len());
                    println!("   • 总参数: {}", llm.total_parameters());

                    // 加载训练数据
                    perf_monitor.start("加载训练数据");
                    let dataset = Dataset::new(
                        String::from("data/pretraining"),
                        String::from("data/chat"),
                    );
                    perf_monitor.stop("加载训练数据");

                    // 准备继续训练
                    let resume_epochs = parse_usize_arg(&args, "--epochs").unwrap_or(500);
                    let lr = parse_f32_arg(&args, "--lr").unwrap_or(metadata.learning_rate);
                    let patience = parse_usize_arg(&args, "--patience").unwrap_or(30);
                    let checkpoint_dir = parse_string_arg(&args, "--checkpoint-dir")
                        .unwrap_or_else(|| "checkpoints".to_string());

                    let mut checkpoint_manager =
                        CheckpointManager::new(&checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
                            .expect("无法创建检查点管理器");

                    // 根据阶段决定继续训练哪个阶段
                    let phase = if metadata.phase == "pretraining" {
                        "pretraining"
                    } else {
                        "instruction_tuning"
                    };

                    println!(
                        "\n▶️  继续{}训练 (从epoch {} 开始)",
                        phase,
                        metadata.epoch + 1
                    );
                    println!("   • 最大epochs: {}", resume_epochs);
                    println!("   • 学习率: {:.6}", lr);
                    println!("   • 早停patience: {}", patience);
                    println!("   • 检查点目录: {}\n", checkpoint_dir);

                    // 准备tokenized数据
                    let data = if phase == "pretraining" {
                        &dataset.pretraining_data
                    } else {
                        &dataset.chat_training_data
                    };

                    perf_monitor.start("Tokenize训练数据");
                    let tokenized_data: Vec<Vec<usize>> = data
                        .iter()
                        .map(|text| LLM::tokenize_with_vocab(&llm.vocab, text))
                        .collect();
                    perf_monitor.stop("Tokenize训练数据");

                    // 继续训练
                    perf_monitor.start(&format!("Resume {} 训练", phase));
                    let actual_epochs = llm.train_with_checkpointing(
                        tokenized_data,
                        resume_epochs,
                        lr,
                        patience,
                        Some(&mut checkpoint_manager),
                        phase,
                        metadata.epoch + 1,
                    );
                    perf_monitor.stop(&format!("Resume {} 训练", phase));

                    println!("\n✓ Resume训练完成，实际训练到 epoch {}", actual_epochs);

                    // 保存最终模型
                    println!("\n💾 保存最终模型...");
                    if let Err(e) = save_model_binary(&llm, "checkpoints/model_final.bin") {
                        log::error!("保存模型失败: {}", e);
                    } else {
                        println!("✅ 模型已保存到 checkpoints/model_final.bin");
                    }

                    perf_monitor.stop("程序总执行时间");
                    perf_monitor.print_report();

                    return;
                }
                Err(e) => {
                    eprintln!("\n❌ 加载检查点失败: {}", e);
                    eprintln!("请检查检查点文件是否存在且格式正确");
                    return;
                }
            }
        } else {
            eprintln!("\n❌ 未找到可用的检查点");
            eprintln!("请使用 --resume-from=<path> 指定检查点路径");
            eprintln!("或确保检查点目录存在有效的检查点文件");
            return;
        }
    }

    // 快速预训练入口：仅运行预训练，适合自动化验证
    if arg_has_flag(&args, "--quick") {
        let pretrain_epochs = parse_usize_arg(&args, "--pretrain-epochs").unwrap_or(30);
        let lr = parse_f32_arg(&args, "--lr").unwrap_or(0.0001);
        let patience = parse_usize_arg(&args, "--patience").unwrap_or(10);
        let accum = parse_usize_arg(&args, "--accum").unwrap_or(1);
        run_quick(
            &mut perf_monitor,
            freeze_attn,
            pretrain_epochs,
            lr,
            patience,
            accum,
        );
        return;
    }

    // 检查是否存在已保存的模型
    let model_path = "checkpoints/model_final.bin";
    let pretrain_checkpoint = "checkpoints/model_pretrained.bin";

    let mut llm = if std::path::Path::new(model_path).exists()
        || std::path::Path::new(pretrain_checkpoint).exists()
    {
        println!("🔍 检测到已保存的模型:");
        if std::path::Path::new(model_path).exists() {
            println!("   ✓ {}", model_path);
        }
        if std::path::Path::new(pretrain_checkpoint).exists() {
            println!("   ✓ {}", pretrain_checkpoint);
        }
        println!();

        print!("是否加载已有模型? (y/n): ");
        if let Err(e) = std::io::stdout().flush() {
            log::warn!("刷新标准输出失败: {}", e);
        }

        let mut choice = String::new();
        if std::io::stdin().read_line(&mut choice).is_err() {
            log::warn!("读取输入失败，默认不加载已有模型");
            choice.clear();
        }

        if choice.trim().eq_ignore_ascii_case("y") {
            // 选择加载哪个模型
            let load_path = if std::path::Path::new(model_path).exists() {
                print!(
                    "\n选择要加载的模型:\n   1) {} (最终模型)\n   2) {} (预训练checkpoint)\n请选择 (1/2): ",
                    model_path, pretrain_checkpoint
                );
                if let Err(e) = std::io::stdout().flush() {
                    log::warn!("刷新标准输出失败: {}", e);
                }

                let mut model_choice = String::new();
                if std::io::stdin().read_line(&mut model_choice).is_err() {
                    log::warn!("读取模型选择失败，默认选择最终模型");
                    model_choice.clear();
                }

                if model_choice.trim() == "2" && std::path::Path::new(pretrain_checkpoint).exists()
                {
                    pretrain_checkpoint
                } else {
                    model_path
                }
            } else {
                pretrain_checkpoint
            };

            println!("\n📂 正在加载模型: {}...", load_path);
            perf_monitor.start("加载模型");

            match load_model_binary(load_path) {
                Ok(mut loaded_llm) => {
                    perf_monitor.stop("加载模型");
                    loaded_llm.set_training_mode(false);

                    println!("\n✅ 模型加载成功!");
                    println!("   • 词汇量: {}", loaded_llm.vocab.len());
                    println!("   • 总参数: {}", loaded_llm.total_parameters());
                    println!("   • 网络架构: {}", loaded_llm.network_description());

                    // 询问是否继续训练
                    print!("\n是否继续训练此模型? (y/n): ");
                    if let Err(e) = std::io::stdout().flush() {
                        log::warn!("刷新标准输出失败: {}", e);
                    }

                    let mut train_choice = String::new();
                    if std::io::stdin().read_line(&mut train_choice).is_err() {
                        log::warn!("读取输入失败，默认不继续训练");
                        train_choice.clear();
                    }

                    if train_choice.trim().eq_ignore_ascii_case("y") {
                        continue_training_loaded_model(loaded_llm, &mut perf_monitor, freeze_attn)
                    } else {
                        println!("\n✓ 跳过训练，直接进入交互模式");
                        loaded_llm
                    }
                }
                Err(e) => {
                    println!("\n❌ 加载模型失败: {}", e);
                    println!("将重新训练模型...\n");
                    train_new_model(&mut perf_monitor, freeze_attn)
                }
            }
        } else {
            println!("\n🔄 将训练新模型...\n");
            train_new_model(&mut perf_monitor, freeze_attn)
        }
    } else {
        println!("📝 未检测到已保存的模型，将开始训练新模型...\n");
        train_new_model(&mut perf_monitor, freeze_attn)
    };

    // 训练完成后，如指定 --no-interactive 则直接退出
    if no_interactive {
        perf_monitor.stop("程序总执行时间");
        perf_monitor.print_report();
        return;
    }

    // 训练完成后询问是否保存
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                    模型保存选项                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    print!("是否保存当前模型? (y/n): ");
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }

    let mut save_choice = String::new();
    if std::io::stdin().read_line(&mut save_choice).is_err() {
        log::warn!("读取输入失败，默认不保存");
        save_choice.clear();
    }

    if save_choice.trim().eq_ignore_ascii_case("y") {
        save_model_interactive(&llm);
    } else {
        println!("✓ 跳过保存");
    }

    // 测试模型
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      模型测试                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test_input = String::from("用户：山脉是如何形成的？");
    println!("测试输入: {}", test_input);

    llm.set_training_mode(false);
    perf_monitor.start("测试预测 (Beam Search)");
    let result = llm.predict_with_beam_search(&test_input, 3, 20);
    perf_monitor.stop("测试预测 (Beam Search)");

    println!("模型输出: {}", result);

    perf_monitor.stop("程序总执行时间");

    // 打印性能报告
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      性能报告                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    perf_monitor.print_report();

    // 进入交互模式
    interactive_mode(&mut llm);
}

/// 训练新模型（使用性能优化）
fn train_new_model(perf_monitor: &mut PerformanceMonitor, freeze_attn: bool) -> LLM {
    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from("data/pretraining"),
        String::from("data/chat"),
    );
    perf_monitor.stop("加载训练数据");

    // 构建词汇表
    let mut vocab_set = std::collections::HashSet::new();

    perf_monitor.start("构建词汇表 - 预训练数据");
    println!("📝 处理预训练数据构建词汇表...");
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    println!("✓ 预训练数据处理完成，当前词汇量: {}", vocab_set.len());
    perf_monitor.stop("构建词汇表 - 预训练数据");

    perf_monitor.start("构建词汇表 - 对话数据");
    println!("📝 处理对话数据构建词汇表...");
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);
    println!("✓ 对话数据处理完成，最终词汇量: {}", vocab_set.len());
    perf_monitor.stop("构建词汇表 - 对话数据");

    perf_monitor.start("创建词汇表对象");
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    println!("📚 创建词汇表，共 {} 个唯一词元...", vocab_words.len());
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);
    println!("✓ 词汇表创建成功，总计 {} 个词元 (含特殊词元)", vocab.len());
    perf_monitor.stop("创建词汇表对象");

    perf_monitor.start("初始化神经网络");
    println!("\n🏗️  初始化神经网络...");
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());

    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(output_projection),
        ],
    );

    perf_monitor.stop("初始化神经网络");

    // 可选冻结注意力参数更新
    if freeze_attn {
        llm.set_attention_freeze_updates(true);
        println!("🔒 注意力层参数更新已冻结 (--freeze-attn)");
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      模型信息                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 网络架构: {}", llm.network_description());
    println!(
        "   • 配置: max_seq_len={}, embedding_dim={}, hidden_dim={}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("   • 总参数量: {}", llm.total_parameters());

    // 训练前测试
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                  训练前模型测试                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let test_input = String::from("用户：山脉是如何形成的？");
    println!("\n测试输入: {}", test_input);

    llm.set_training_mode(false);
    perf_monitor.start("训练前预测");
    let before_output = llm.predict_with_beam_search(&test_input, 3, 20);
    perf_monitor.stop("训练前预测");

    println!("训练前输出: {}\n", before_output);

    // 🔥 阶段1：预训练（使用优化的训练方法）
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║            阶段1: 预训练 (Pre-training) - 优化版          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 训练样本: {}", dataset.pretraining_data.len());
    println!("   • 最大epochs: 500 (早停patience=30)");
    println!("   • 学习率: 0.0001 (余弦退火, 无重启)");
    println!("   • 梯度累积: 1步 (暂时禁用以提升稳定性)");
    println!("   • 优化: 数据缓存 + 余弦退火(无重启) + 早停 + 梯度裁剪\n");

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("预训练阶段");
    let actual_epochs = llm.train_monitored(
        pretraining_examples,
        500,    // max_epochs
        0.0001, // initial_lr（更低学习率提升稳定性）
        30,     // patience（小数据集快速迭代）
        1,      // accumulation_steps（暂时禁用累积）
    );
    perf_monitor.stop("预训练阶段");

    println!("✓ 预训练完成，实际训练 {} epochs", actual_epochs);

    // 询问是否保存预训练checkpoint
    print!("\n💾 是否保存预训练checkpoint? (y/n): ");
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }

    let mut checkpoint_choice = String::new();
    if std::io::stdin().read_line(&mut checkpoint_choice).is_err() {
        log::warn!("读取输入失败，将跳过checkpoint保存");
        checkpoint_choice.clear();
    }

    if checkpoint_choice.trim().eq_ignore_ascii_case("y") {
        std::fs::create_dir_all("checkpoints").ok();
        match save_model_binary(&llm, "checkpoints/model_pretrained.bin") {
            Ok(_) => println!("✓ 预训练checkpoint已保存"),
            Err(e) => println!("❌ 保存失败: {}", e),
        }
    }

    // 🔥 阶段2：指令微调（使用优化的训练方法）
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║        阶段2: 指令微调 (Instruction Tuning) - 优化版     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 训练样本: {}", dataset.chat_training_data.len());
    println!("   • 最大epochs: 500 (早停patience=30)");
    println!("   • 学习率: 0.0001 (余弦退火, 无重启)");
    println!("   • 梯度累积: 1步 (稳定优先，后续可渐进恢复)\n");

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("指令微调阶段");
    let actual_epochs = llm.train_monitored(chat_training_examples, 500, 0.0001, 30, 1);
    perf_monitor.stop("指令微调阶段");

    println!("✓ 指令微调完成，实际训练 {} epochs", actual_epochs);

    println!("\n✅ 训练完成!");

    llm
}

/// 继续训练已加载的模型
fn continue_training_loaded_model(
    mut llm: LLM,
    perf_monitor: &mut PerformanceMonitor,
    freeze_attn: bool,
) -> LLM {
    println!("\n🔄 继续训练模式");

    // 加载数据
    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from("data/pretraining"),
        String::from("data/chat"),
    );
    perf_monitor.stop("加载训练数据");

    // 询问训练参数
    print!("\n训练轮数 (默认50): ");
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }
    let mut epochs_input = String::new();
    if std::io::stdin().read_line(&mut epochs_input).is_err() {
        log::warn!("读取训练轮数失败，使用默认值 50");
        epochs_input.clear();
    }
    let epochs: usize = epochs_input.trim().parse().unwrap_or(50);

    print!("学习率 (默认0.0001): ");
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }
    let mut lr_input = String::new();
    if std::io::stdin().read_line(&mut lr_input).is_err() {
        log::warn!("读取学习率失败，使用默认值 0.0001");
        lr_input.clear();
    }
    let lr: f32 = lr_input.trim().parse().unwrap_or(0.0001);

    println!("\n开始继续训练 ({} epochs, lr={})...\n", epochs, lr);

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    // 可选冻结注意力参数更新
    if freeze_attn {
        llm.set_attention_freeze_updates(true);
        println!("🔒 注意力层参数更新已冻结 (--freeze-attn)");
    }

    llm.set_training_mode(true);
    perf_monitor.start("继续训练");

    // 🔥 使用优化的训练方法
    llm.train_monitored(
        chat_training_examples,
        epochs,
        lr,
        30, // patience (稳定配置：约30)
        1,  // accumulation_steps（稳定优先）
    );

    perf_monitor.stop("继续训练");

    println!("\n✅ 继续训练完成!");

    llm
}

/// 交互式保存模型
fn save_model_interactive(llm: &LLM) {
    println!("\n选择保存格式:");
    println!("   1) 二进制格式 (.bin) - 推荐，文件小、速度快");
    println!("   2) JSON格式 (.json) - 人类可读，便于调试");
    println!("   3) 两种格式都保存");

    print!("\n请选择 (1/2/3): ");
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }

    let mut format_choice = String::new();
    if std::io::stdin().read_line(&mut format_choice).is_err() {
        log::warn!("读取输入失败，默认跳过保存");
        format_choice.clear();
    }

    std::fs::create_dir_all("checkpoints").ok();
    std::fs::create_dir_all("exports").ok();

    match format_choice.trim() {
        "1" => {
            print!("文件名 (默认: checkpoints/model_final.bin): ");
            if let Err(e) = std::io::stdout().flush() {
                log::warn!("刷新标准输出失败: {}", e);
            }

            let mut filename = String::new();
            if std::io::stdin().read_line(&mut filename).is_err() {
                log::warn!("读取文件名失败，使用默认路径");
                filename.clear();
            }
            let path = if filename.trim().is_empty() {
                "checkpoints/model_final.bin"
            } else {
                filename.trim()
            };

            match save_model_binary(llm, path) {
                Ok(_) => println!("✅ 模型已保存: {}", path),
                Err(e) => println!("❌ 保存失败: {}", e),
            }
        }
        "2" => {
            print!("文件名 (默认: exports/model_final.json): ");
            if let Err(e) = std::io::stdout().flush() {
                log::warn!("刷新标准输出失败: {}", e);
            }

            let mut filename = String::new();
            if std::io::stdin().read_line(&mut filename).is_err() {
                log::warn!("读取文件名失败，使用默认路径");
                filename.clear();
            }
            let path = if filename.trim().is_empty() {
                "exports/model_final.json"
            } else {
                filename.trim()
            };

            match save_model_json(llm, path) {
                Ok(_) => println!("✅ 模型已保存: {}", path),
                Err(e) => println!("❌ 保存失败: {}", e),
            }
        }
        "3" => {
            println!("\n保存二进制格式...");
            match save_model_binary(llm, "checkpoints/model_final.bin") {
                Ok(_) => println!("✓ 二进制格式已保存: checkpoints/model_final.bin"),
                Err(e) => println!("✗ 二进制保存失败: {}", e),
            }

            println!("保存JSON格式...");
            match save_model_json(llm, "exports/model_final.json") {
                Ok(_) => println!("✓ JSON格式已保存: exports/model_final.json"),
                Err(e) => println!("✗ JSON保存失败: {}", e),
            }
        }
        _ => println!("❌ 无效选项，跳过保存"),
    }
}

/// 交互模式
fn interactive_mode(llm: &mut LLM) {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      交互模式                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n💡 输入问题后按回车生成回答");
    println!("💡 输入 'exit' 退出程序");
    println!("💡 输入 'clear' 清空对话上下文");
    println!("💡 输入 'save' 保存当前模型");
    println!("💡 使用KV缓存加速推理（约10-100倍）\n");

    // 启用KV缓存加速推理
    llm.enable_kv_cache();

    let mut input = String::new();
    loop {
        input.clear();

        print!("👤 用户: ");
        if let Err(e) = std::io::stdout().flush() {
            log::warn!("刷新标准输出失败: {}", e);
        }

        if std::io::stdin().read_line(&mut input).is_err() {
            log::warn!("读取输入失败，已跳过本次交互");
            continue;
        }

        let trimmed_input = input.trim();

        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("👋 感谢使用，再见!");
            break;
        }

        if trimmed_input.eq_ignore_ascii_case("clear") {
            llm.clear_context();
            llm.clear_kv_cache(); // 同时清空KV缓存
            println!("✓ 对话上下文和KV缓存已清空\n");
            continue;
        }

        if trimmed_input.eq_ignore_ascii_case("save") {
            save_model_interactive(llm);
            println!();
            continue;
        }

        let formatted_input = format!("用户：{}", trimmed_input);
        print!("🤖 模型: ");
        if let Err(e) = std::io::stdout().flush() {
            log::warn!("刷新标准输出失败: {}", e);
        }

        let prediction = llm.predict_with_context(&formatted_input, 0.8, 0.9, 5);
        println!("{}\n", prediction);

        if prediction.contains("</s>") {
            llm.clear_context();
        }
    }
}
