use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

mod cli;

use llm::{
    CheckpointManager, CheckpointStrategy, Dataset, EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM,
    MAX_SEQ_LEN, OutputProjection, PerformanceMonitor, TransformerBlock, Vocab, load_model_auto,
    save_model_binary, save_model_json,
};

const DEFAULT_PRETRAINING_DIR: &str = "data/pretraining";
const DEFAULT_CHAT_DIR: &str = "data/chat";
const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints";
const DEFAULT_MODEL_PATH: &str = "checkpoints/model_final.bin";
const DEFAULT_PRETRAINED_MODEL_PATH: &str = "checkpoints/model_pretrained.bin";

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    init_logger();

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let action = match cli::parse(argv) {
        Ok(action) => action,
        Err(error) => {
            eprintln!("{}", error);
            return 2;
        }
    };

    match action {
        cli::Action::PrintVersion => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            0
        }
        cli::Action::PrintHelp { topic, exit_code } => {
            match topic {
                cli::HelpTopic::Global => println!("{}", cli::help_global()),
                cli::HelpTopic::Command(command) => println!("{}", cli::help_command(&command)),
            }
            exit_code
        }
        cli::Action::Run { global, command } => {
            let result = match command {
                cli::Command::Train(options) => cmd_train(&options),
                cli::Command::Resume(options) => cmd_resume(&options),
                cli::Command::Chat(options) => cmd_chat(&global, &options),
                cli::Command::Export(options) => cmd_export(&options),
                cli::Command::Info => cmd_info(),
            };

            match result {
                Ok(()) => 0,
                Err(message) => {
                    eprintln!("{}", message);
                    1
                }
            }
        }
    }
}

fn init_logger() {
    if let Err(e) = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
    {
        eprintln!("日志初始化失败: {}", e);
    }
}

fn print_app_banner() {
    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║          RustGPT-Chinese - 中文GPT模型训练系统            ║");
    println!("╚═══════════════════════════════════════════════════════════╝
");
}

fn flush_stdout() {
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }
}

fn ensure_parent_dir(path: &str) -> Result<(), String> {
    let p = Path::new(path);
    let Some(parent) = p.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("创建目录失败: {}: {}", parent.display(), error))
}

fn save_model_by_extension(llm: &LLM, path: &str) -> Result<(), String> {
    ensure_parent_dir(path)?;
    if path.to_ascii_lowercase().ends_with(".json") {
        save_model_json(llm, path).map_err(|error| format!("保存 JSON 失败: {}", error))
    } else {
        save_model_binary(llm, path).map_err(|error| format!("保存二进制失败: {}", error))
    }
}

fn load_dataset(
    perf_monitor: &mut PerformanceMonitor,
    pretraining_dir: &str,
    chat_dir: &str,
) -> Dataset {
    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(pretraining_dir.to_string(), chat_dir.to_string());
    perf_monitor.stop("加载训练数据");
    dataset
}

fn build_vocab_from_dataset(
    perf_monitor: &mut PerformanceMonitor,
    dataset: &Dataset,
    verbose: bool,
) -> Vocab {
    let mut vocab_set = HashSet::new();

    perf_monitor.start("构建词汇表 - 预训练数据");
    if verbose {
        println!("📝 处理预训练数据构建词汇表...");
    }
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    if verbose {
        println!("✓ 预训练数据处理完成，当前词汇量: {}", vocab_set.len());
    }
    perf_monitor.stop("构建词汇表 - 预训练数据");

    perf_monitor.start("构建词汇表 - 对话数据");
    if verbose {
        println!("📝 处理对话数据构建词汇表...");
    }
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);
    if verbose {
        println!("✓ 对话数据处理完成，最终词汇量: {}", vocab_set.len());
    }
    perf_monitor.stop("构建词汇表 - 对话数据");

    perf_monitor.start("创建词汇表对象");
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    if verbose {
        println!("📚 创建词汇表，共 {} 个唯一词元...", vocab_words.len());
    }
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);
    if verbose {
        println!("✓ 词汇表创建成功，总计 {} 个词元 (含特殊词元)", vocab.len());
    }
    perf_monitor.stop("创建词汇表对象");

    vocab
}

fn build_default_llm(
    perf_monitor: &mut PerformanceMonitor,
    vocab: Vocab,
    verbose: bool,
) -> LLM {
    perf_monitor.start("初始化神经网络");
    if verbose {
        println!("
🏗️  初始化神经网络...");
    }

    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());

    let llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(output_projection),
        ],
    );

    perf_monitor.stop("初始化神经网络");
    llm
}

fn apply_attention_freeze_if_requested(llm: &mut LLM, freeze_attn: bool) {
    if freeze_attn {
        llm.set_attention_freeze_updates(true);
        println!("🔒 注意力层参数更新已冻结 (--freeze-attn)");
    }
}

fn cmd_train(options: &cli::TrainOptions) -> Result<(), String> {
    print_app_banner();

    let mut perf_monitor = PerformanceMonitor::new();
    perf_monitor.start("程序总执行时间");

    let dataset = load_dataset(
        &mut perf_monitor,
        &options.pretraining_dir,
        &options.chat_dir,
    );

    let vocab = build_vocab_from_dataset(&mut perf_monitor, &dataset, true);
    let mut llm = build_default_llm(&mut perf_monitor, vocab, true);
    apply_attention_freeze_if_requested(&mut llm, options.freeze_attn);

    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║                      模型信息                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 网络架构: {}", llm.network_description());
    println!(
        "   • 配置: max_seq_len={}, embedding_dim={}, hidden_dim={}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("   • 总参数量: {}", llm.total_parameters());

    if options.pretrain_epochs > 0 {
        println!("
╔═══════════════════════════════════════════════════════════╗");
        println!("║            阶段1: 预训练 (Pre-training)                   ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!("   • 训练样本: {}", dataset.pretraining_data.len());
        println!("   • 最大epochs: {} (早停patience={})", options.pretrain_epochs, options.patience);
        println!("   • 学习率: {:.6}", options.lr);
        println!("   • 梯度累积: {}", options.accum);

        let pretraining_examples: Vec<&str> = dataset
            .pretraining_data
            .iter()
            .map(|s| s.as_str())
            .collect();

        perf_monitor.start("预训练阶段");
        let actual_epochs = llm.train_monitored(
            pretraining_examples,
            options.pretrain_epochs,
            options.lr,
            options.patience,
            options.accum,
        );
        perf_monitor.stop("预训练阶段");

        println!("✓ 预训练完成，实际训练 {} epochs", actual_epochs);

        if let Some(path) = &options.save_pretrained {
            println!("💾 保存预训练模型: {}", path);
            save_model_by_extension(&llm, path)?;
        }
    } else {
        println!("⚠️ 已跳过预训练阶段（--pretrain-epochs=0）");
    }

    if options.finetune_epochs > 0 {
        println!("
╔═══════════════════════════════════════════════════════════╗");
        println!("║        阶段2: 指令微调 (Instruction Tuning)               ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!("   • 训练样本: {}", dataset.chat_training_data.len());
        println!("   • 最大epochs: {} (早停patience={})", options.finetune_epochs, options.patience);
        println!("   • 学习率: {:.6}", options.lr);
        println!("   • 梯度累积: {}", options.accum);

        let chat_training_examples: Vec<&str> = dataset
            .chat_training_data
            .iter()
            .map(|s| s.as_str())
            .collect();

        perf_monitor.start("指令微调阶段");
        let actual_epochs = llm.train_monitored(
            chat_training_examples,
            options.finetune_epochs,
            options.lr,
            options.patience,
            options.accum,
        );
        perf_monitor.stop("指令微调阶段");

        println!("✓ 指令微调完成，实际训练 {} epochs", actual_epochs);
    } else {
        println!("⚠️ 已跳过指令微调阶段（--finetune-epochs=0）");
    }

    if let Some(path) = &options.save_final {
        println!("✅ 训练完成，保存最终模型: {}", path);
        save_model_by_extension(&llm, path)?;
    } else {
        println!("✅ 训练完成（未保存最终模型：--no-save-final）");
    }

    perf_monitor.stop("程序总执行时间");
    perf_monitor.print_report();

    Ok(())
}

fn resolve_resume_checkpoint_path(
    checkpoint_dir: &str,
    resume_from: &Option<String>,
) -> Result<String, String> {
    if let Some(path) = resume_from {
        return Ok(path.clone());
    }

    let Ok(manager) = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3) else {
        return Err(format!("无法创建检查点管理器: {}", checkpoint_dir));
    };

    if let Some(best) = manager.get_best_checkpoint() {
        return Ok(best.to_string_lossy().to_string());
    }
    if let Some(last) = manager.get_last_checkpoint() {
        return Ok(last.to_string_lossy().to_string());
    }

    Err(format!("未找到可用的检查点（目录: {}）", checkpoint_dir))
}

fn cmd_resume(options: &cli::ResumeOptions) -> Result<(), String> {
    print_app_banner();

    let mut perf_monitor = PerformanceMonitor::new();
    perf_monitor.start("程序总执行时间");

    let path = resolve_resume_checkpoint_path(&options.checkpoint_dir, &options.resume_from)?;

    println!("
🔄 从检查点恢复训练: {}", path);
    let (mut llm, metadata) =
        CheckpointManager::load_checkpoint(&path).map_err(|e| format!("加载检查点失败: {}", e))?;

    println!("
✅ 检查点加载成功!");
    println!("   • 训练阶段: {}", metadata.phase);
    println!("   • Epoch: {}", metadata.epoch);
    println!("   • Loss: {:.4}", metadata.loss);
    println!("   • 学习率: {:.6}", metadata.learning_rate);
    println!("   • 时间戳: {}", metadata.timestamp);
    println!("   • 词汇量: {}", llm.vocab.len());
    println!("   • 总参数: {}", llm.total_parameters());

    let dataset = load_dataset(
        &mut perf_monitor,
        &options.pretraining_dir,
        &options.chat_dir,
    );
    let lr = options.lr.unwrap_or(metadata.learning_rate);

    let mut checkpoint_manager = CheckpointManager::new(
        &options.checkpoint_dir,
        CheckpointStrategy::BestAndLast,
        3,
    )
    .map_err(|e| format!("无法创建检查点管理器: {}", e))?;

    let phase = if metadata.phase == "pretraining" {
        "pretraining"
    } else {
        "instruction_tuning"
    };

    println!(
        "
▶️  继续{}训练 (从epoch {} 开始)",
        phase,
        metadata.epoch + 1
    );
    println!("   • 最大epochs: {}", options.epochs);
    println!("   • 学习率: {:.6}", lr);
    println!("   • 早停patience: {}", options.patience);
    println!("   • 检查点目录: {}
", options.checkpoint_dir);

    let data = if phase == "pretraining" {
        &dataset.pretraining_data
    } else {
        &dataset.chat_training_data
    };

    perf_monitor.start("Tokenize训练数据");
    let tokenized_data: Vec<Vec<usize>> = data
        .iter()
        .map(|text| LLM::tokenize_training_with_vocab(&llm.vocab, text))
        .collect();
    perf_monitor.stop("Tokenize训练数据");

    perf_monitor.start(&format!("Resume {} 训练", phase));
    let actual_epochs = llm.train_with_checkpointing(
        tokenized_data,
        options.epochs,
        lr,
        options.patience,
        Some(&mut checkpoint_manager),
        phase,
        metadata.epoch + 1,
    );
    perf_monitor.stop(&format!("Resume {} 训练", phase));

    println!("✓ Resume训练完成，实际训练到 epoch {}", actual_epochs);

    println!("💾 保存最终模型: {}", options.save_final);
    save_model_by_extension(&llm, &options.save_final)?;

    perf_monitor.stop("程序总执行时间");
    perf_monitor.print_report();

    Ok(())
}

fn cmd_chat(global: &cli::GlobalOptions, options: &cli::ChatOptions) -> Result<(), String> {
    let mut llm =
        load_model_auto(&options.model).map_err(|e| format!("加载模型失败: {}", e))?;

    llm.set_training_mode(false);
    llm.enable_kv_cache();

    let mut temperature = options.temperature;
    let mut top_p = options.top_p;
    let mut top_k = options.top_k;

    if let Some(prompt) = &options.prompt {
        let formatted = format!("用户：{}", prompt.trim());
        let prediction = llm.predict_with_context(&formatted, temperature, top_p, top_k);
        println!("{}", prediction);
        return Ok(());
    }

    if global.no_interactive {
        return Err("已启用 --no-interactive，但 chat 未提供 --prompt=...，无法进入 REPL。".to_string());
    }

    print_app_banner();
    println!("
进入推理 REPL（输入 :help 查看命令）");
    println!("提示：输入普通文本将作为“用户”输入进行推理。
");

    let mut input = String::new();
    loop {
        input.clear();
        print!("👤 用户: ");
        flush_stdout();

        if std::io::stdin().read_line(&mut input).is_err() {
            return Err("读取输入失败，已退出 REPL。".to_string());
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with(':') {
            if handle_repl_command(
                trimmed,
                &mut llm,
                &mut temperature,
                &mut top_p,
                &mut top_k,
            )? {
                break;
            }
            continue;
        }

        let formatted_input = format!("用户：{}", trimmed);
        print!("🤖 模型: ");
        flush_stdout();

        let prediction = llm.predict_with_context(&formatted_input, temperature, top_p, top_k);
        println!("{}
", prediction);

        if prediction.contains("</s>") {
            llm.clear_context();
        }
    }

    Ok(())
}

fn handle_repl_command(
    line: &str,
    llm: &mut LLM,
    temperature: &mut f32,
    top_p: &mut f32,
    top_k: &mut usize,
) -> Result<bool, String> {
    let mut parts = line.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };

    match cmd {
        ":help" => {
            println!(
                "
REPL 命令:
  :help
  :exit
  :clear
  :save <path>
  :set temperature=<f>
  :set top_p=<f>
  :set top_k=<n>
"
            );
        }
        ":exit" => return Ok(true),
        ":clear" => {
            llm.clear_context();
            llm.clear_kv_cache();
            println!("✓ 已清空对话上下文与 KV 缓存");
        }
        ":save" => {
            let Some(path) = parts.next() else {
                return Err("用法: :save <path>".to_string());
            };
            save_model_by_extension(llm, path)?;
            println!("✓ 已保存模型: {}", path);
        }
        ":set" => {
            let Some(kv) = parts.next() else {
                return Err("用法: :set temperature=<f> | top_p=<f> | top_k=<n>".to_string());
            };
            let Some((k, v)) = kv.split_once('=') else {
                return Err("用法: :set temperature=<f> | top_p=<f> | top_k=<n>".to_string());
            };
            match k {
                "temperature" => {
                    *temperature = v
                        .parse()
                        .map_err(|_| format!("temperature 需要是浮点数: {}", v))?;
                    println!("✓ temperature={}", temperature);
                }
                "top_p" => {
                    *top_p = v.parse().map_err(|_| format!("top_p 需要是浮点数: {}", v))?;
                    println!("✓ top_p={}", top_p);
                }
                "top_k" => {
                    *top_k = v.parse().map_err(|_| format!("top_k 需要是整数: {}", v))?;
                    println!("✓ top_k={}", top_k);
                }
                _ => {
                    return Err(format!(
                        "未知设置项: {}（支持: temperature, top_p, top_k）",
                        k
                    ));
                }
            }
        }
        _ => {
            return Err(format!(
                "未知 REPL 命令: {}（输入 :help 查看可用命令）",
                cmd
            ));
        }
    }

    Ok(false)
}

fn cmd_export(options: &cli::ExportOptions) -> Result<(), String> {
    let llm =
        load_model_auto(&options.model).map_err(|e| format!("加载模型失败: {}", e))?;

    if let Some(path) = &options.out_bin {
        ensure_parent_dir(path)?;
        save_model_binary(&llm, path).map_err(|e| format!("导出二进制失败: {}", e))?;
        println!("✓ 已导出二进制: {}", path);
    }

    if let Some(path) = &options.out_json {
        ensure_parent_dir(path)?;
        save_model_json(&llm, path).map_err(|e| format!("导出 JSON 失败: {}", e))?;
        println!("✓ 已导出 JSON: {}", path);
    }

    Ok(())
}

fn cmd_info() -> Result<(), String> {
    println!("RustGPT-Chinese v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("默认路径约定:");
    println!("  预训练数据: {}", DEFAULT_PRETRAINING_DIR);
    println!("  对话数据:   {}", DEFAULT_CHAT_DIR);
    println!("  检查点目录: {}", DEFAULT_CHECKPOINT_DIR);
    println!("  最终模型:   {}", DEFAULT_MODEL_PATH);
    println!("  预训练模型: {}", DEFAULT_PRETRAINED_MODEL_PATH);
    println!();
    println!("推荐命令:");
    println!("  llm train");
    println!("  llm resume --checkpoint-dir=checkpoints");
    println!("  llm chat --model=checkpoints/model_final.bin");
    println!("  llm export --model=checkpoints/model_final.bin --out-json=exports/model_final.json");
    Ok(())
}
