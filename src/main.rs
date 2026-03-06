use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

// 从 lib.rs 导入当前 CLI 需要的核心类型。
use llm::{
    CheckpointManager, CheckpointStrategy, Dataset, EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM,
    MAX_SEQ_LEN, OutputProjection, PerformanceMonitor, TransformerBlock, Vocab, load_model_binary,
    save_model_binary, save_model_json,
};

const DEFAULT_PRETRAINING_DIR: &str = "data/pretraining";
const DEFAULT_CHAT_DIR: &str = "data/chat";
const DEFAULT_MODEL_PATH: &str = "checkpoints/model_final.bin";
const DEFAULT_PRETRAINED_MODEL_PATH: &str = "checkpoints/model_pretrained.bin";
const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints";
const DEFAULT_EXPORT_JSON_PATH: &str = "exports/model_final.json";

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

fn flush_stdout() {
    if let Err(e) = std::io::stdout().flush() {
        log::warn!("刷新标准输出失败: {}", e);
    }
}

fn read_line_or_empty(read_error_message: &str) -> String {
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        log::warn!("{}", read_error_message);
        return String::new();
    }
    input.trim().to_string()
}

fn prompt_yes_no(prompt: &str, read_error_message: &str) -> bool {
    print!("{}", prompt);
    flush_stdout();
    read_line_or_empty(read_error_message).eq_ignore_ascii_case("y")
}

fn print_app_banner() {
    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║          RustGPT-Chinese - 中文GPT模型训练系统            ║");
    println!("╚═══════════════════════════════════════════════════════════╝
");
}

fn init_logger() {
    if let Err(e) = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
    {
        eprintln!("日志初始化失败: {}", e);
    }
}

/// 加载默认训练数据目录。
fn load_default_dataset(perf_monitor: &mut PerformanceMonitor) -> Dataset {
    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from(DEFAULT_PRETRAINING_DIR),
        String::from(DEFAULT_CHAT_DIR),
    );
    perf_monitor.stop("加载训练数据");
    dataset
}

/// 从默认数据集构建词汇表。
///
/// `verbose=true` 时保留训练入口中的教学日志；`false` 时保持 quick 模式更精简。
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

/// 用默认两层 Transformer 结构构建模型。
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

fn resolve_resume_checkpoint_path(args: &[String]) -> Option<String> {
    parse_string_arg(args, "--resume-from").or_else(|| {
        let checkpoint_dir = parse_string_arg(args, "--checkpoint-dir")
            .unwrap_or_else(|| DEFAULT_CHECKPOINT_DIR.to_string());

        if let Ok(manager) = CheckpointManager::new(&checkpoint_dir, CheckpointStrategy::Best, 3) {
            if let Some(best) = manager.get_best_checkpoint() {
                return Some(best.to_string_lossy().to_string());
            }
            if let Some(last) = manager.get_last_checkpoint() {
                return Some(last.to_string_lossy().to_string());
            }
        }
        None
    })
}

/// 如用户要求从 checkpoint 恢复训练，则在此完整处理并结束程序。
fn try_resume_training(args: &[String], perf_monitor: &mut PerformanceMonitor) -> bool {
    if !arg_has_flag(args, "--resume") {
        return false;
    }

    let Some(path) = resolve_resume_checkpoint_path(args) else {
        eprintln!("
❌ 未找到可用的检查点");
        eprintln!("请使用 --resume-from=<path> 指定检查点路径");
        eprintln!("或确保检查点目录存在有效的检查点文件");
        return true;
    };

    println!("
🔄 从检查点恢复训练: {}", path);
    match CheckpointManager::load_checkpoint(&path) {
        Ok((mut llm, metadata)) => {
            println!("
✅ 检查点加载成功!");
            println!("   • 训练阶段: {}", metadata.phase);
            println!("   • Epoch: {}", metadata.epoch);
            println!("   • Loss: {:.4}", metadata.loss);
            println!("   • 学习率: {:.6}", metadata.learning_rate);
            println!("   • 时间戳: {}", metadata.timestamp);
            println!("   • 词汇量: {}", llm.vocab.len());
            println!("   • 总参数: {}", llm.total_parameters());

            let dataset = load_default_dataset(perf_monitor);
            let resume_epochs = parse_usize_arg(args, "--epochs").unwrap_or(500);
            let lr = parse_f32_arg(args, "--lr").unwrap_or(metadata.learning_rate);
            let patience = parse_usize_arg(args, "--patience").unwrap_or(30);
            let checkpoint_dir = parse_string_arg(args, "--checkpoint-dir")
                .unwrap_or_else(|| DEFAULT_CHECKPOINT_DIR.to_string());

            let mut checkpoint_manager =
                CheckpointManager::new(&checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
                    .expect("无法创建检查点管理器");

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
            println!("   • 最大epochs: {}", resume_epochs);
            println!("   • 学习率: {:.6}", lr);
            println!("   • 早停patience: {}", patience);
            println!("   • 检查点目录: {}
", checkpoint_dir);

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
                resume_epochs,
                lr,
                patience,
                Some(&mut checkpoint_manager),
                phase,
                metadata.epoch + 1,
            );
            perf_monitor.stop(&format!("Resume {} 训练", phase));

            println!("
✓ Resume训练完成，实际训练到 epoch {}", actual_epochs);

            std::fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).ok();
            if let Err(e) = save_model_binary(&llm, DEFAULT_MODEL_PATH) {
                eprintln!("
⚠️ 保存最终模型失败: {}", e);
            } else {
                println!("✅ 模型已保存到 {}", DEFAULT_MODEL_PATH);
            }

            perf_monitor.stop("程序总执行时间");
            perf_monitor.print_report();
        }
        Err(e) => {
            eprintln!("
❌ 加载检查点失败: {}", e);
            eprintln!("请检查检查点文件是否存在且格式正确");
        }
    }

    true
}

fn select_existing_model_path(model_path: &str, pretrain_checkpoint: &str) -> &'static str {
    if Path::new(model_path).exists() {
        print!(
            "
选择要加载的模型:
   1) {} (最终模型)
   2) {} (预训练checkpoint)
请选择 (1/2): ",
            model_path, pretrain_checkpoint
        );
        flush_stdout();

        let model_choice = read_line_or_empty("读取模型选择失败，默认选择最终模型");
        if model_choice == "2" && Path::new(pretrain_checkpoint).exists() {
            DEFAULT_PRETRAINED_MODEL_PATH
        } else {
            DEFAULT_MODEL_PATH
        }
    } else {
        DEFAULT_PRETRAINED_MODEL_PATH
    }
}

/// 尝试加载已有模型；若用户不加载或加载失败，则退回训练新模型。
fn try_load_existing_model(
    perf_monitor: &mut PerformanceMonitor,
    freeze_attn: bool,
) -> LLM {
    let model_exists = Path::new(DEFAULT_MODEL_PATH).exists();
    let pretrained_exists = Path::new(DEFAULT_PRETRAINED_MODEL_PATH).exists();

    if !(model_exists || pretrained_exists) {
        println!("📝 未检测到已保存的模型，将开始训练新模型...
");
        return train_new_model(perf_monitor, freeze_attn);
    }

    println!("🔍 检测到已保存的模型:");
    if model_exists {
        println!("   ✓ {}", DEFAULT_MODEL_PATH);
    }
    if pretrained_exists {
        println!("   ✓ {}", DEFAULT_PRETRAINED_MODEL_PATH);
    }
    println!();

    if !prompt_yes_no("是否加载已有模型? (y/n): ", "读取输入失败，默认不加载已有模型") {
        println!("
🔄 将训练新模型...
");
        return train_new_model(perf_monitor, freeze_attn);
    }

    let load_path = select_existing_model_path(DEFAULT_MODEL_PATH, DEFAULT_PRETRAINED_MODEL_PATH);
    println!("
📂 正在加载模型: {}...", load_path);
    perf_monitor.start("加载模型");

    match load_model_binary(load_path) {
        Ok(mut loaded_llm) => {
            perf_monitor.stop("加载模型");
            loaded_llm.set_training_mode(false);

            println!("
✅ 模型加载成功!");
            println!("   • 词汇量: {}", loaded_llm.vocab.len());
            println!("   • 总参数: {}", loaded_llm.total_parameters());
            println!("   • 网络架构: {}", loaded_llm.network_description());

            if prompt_yes_no(
                "
是否继续训练此模型? (y/n): ",
                "读取输入失败，默认不继续训练",
            ) {
                continue_training_loaded_model(loaded_llm, perf_monitor, freeze_attn)
            } else {
                println!("
✓ 跳过训练，直接进入交互模式");
                loaded_llm
            }
        }
        Err(e) => {
            println!("
❌ 加载模型失败: {}", e);
            println!("将重新训练模型...
");
            train_new_model(perf_monitor, freeze_attn)
        }
    }
}

/// 训练结束后，询问用户是否保存模型。
fn prompt_save_after_training(llm: &LLM) {
    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║                    模型保存选项                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝
");

    if prompt_yes_no("是否保存当前模型? (y/n): ", "读取输入失败，默认不保存") {
        save_model_interactive(llm);
    } else {
        println!("✓ 跳过保存");
    }
}

/// 训练主流程完成后的演示：可选保存、测试预测、打印报告、进入交互模式。
fn run_post_training_demo(
    llm: &mut LLM,
    perf_monitor: &mut PerformanceMonitor,
    no_interactive: bool,
) {
    if no_interactive {
        perf_monitor.stop("程序总执行时间");
        perf_monitor.print_report();
        return;
    }

    prompt_save_after_training(llm);

    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║                      模型测试                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝
");

    let test_input = String::from("用户：山脉是如何形成的？");
    println!("测试输入: {}", test_input);

    llm.set_training_mode(false);
    perf_monitor.start("测试预测 (Beam Search)");
    let result = llm.predict_with_beam_search(&test_input, 3, 20);
    perf_monitor.stop("测试预测 (Beam Search)");

    println!("模型输出: {}", result);

    perf_monitor.stop("程序总执行时间");

    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║                      性能报告                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝
");
    perf_monitor.print_report();

    interactive_mode(llm);
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
    println!("
⚡ 启动快速预训练 (--quick) 模式");

    let dataset = load_default_dataset(perf_monitor);
    let vocab = build_vocab_from_dataset(perf_monitor, &dataset, false);
    let mut llm = build_default_llm(perf_monitor, vocab, false);

    apply_attention_freeze_if_requested(&mut llm, freeze_attn);

    println!(
        "
[Quick] 预训练: epochs={}, lr={:.6}, patience={}, accum={} (cosine, 无重启, clip=1.0)",
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
    print_app_banner();
    init_logger();

    let mut perf_monitor = PerformanceMonitor::new();
    perf_monitor.start("程序总执行时间");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let freeze_attn = arg_has_flag(&args, "--freeze-attn");
    let no_interactive = arg_has_flag(&args, "--no-interactive");

    if try_resume_training(&args, &mut perf_monitor) {
        return;
    }

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

    let mut llm = try_load_existing_model(&mut perf_monitor, freeze_attn);
    run_post_training_demo(&mut llm, &mut perf_monitor, no_interactive);
}

/// 训练新模型（使用性能优化）
fn train_new_model(perf_monitor: &mut PerformanceMonitor, freeze_attn: bool) -> LLM {
    let dataset = load_default_dataset(perf_monitor);
    let vocab = build_vocab_from_dataset(perf_monitor, &dataset, true);
    let mut llm = build_default_llm(perf_monitor, vocab, true);

    apply_attention_freeze_if_requested(&mut llm, freeze_attn);

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

    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║                  训练前模型测试                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let test_input = String::from("用户：山脉是如何形成的？");
    println!("
测试输入: {}", test_input);

    llm.set_training_mode(false);
    perf_monitor.start("训练前预测");
    let before_output = llm.predict_with_beam_search(&test_input, 3, 20);
    perf_monitor.stop("训练前预测");

    println!("训练前输出: {}
", before_output);

    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║            阶段1: 预训练 (Pre-training) - 优化版          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 训练样本: {}", dataset.pretraining_data.len());
    println!("   • 最大epochs: 500 (早停patience=30)");
    println!("   • 学习率: 0.0001 (余弦退火, 无重启)");
    println!("   • 梯度累积: 1步 (暂时禁用以提升稳定性)");
    println!("   • 优化: 数据缓存 + 余弦退火(无重启) + 早停 + 梯度裁剪
");

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("预训练阶段");
    let actual_epochs = llm.train_monitored(
        pretraining_examples,
        500,
        0.0001,
        30,
        1,
    );
    perf_monitor.stop("预训练阶段");

    println!("✓ 预训练完成，实际训练 {} epochs", actual_epochs);

    if prompt_yes_no(
        "
💾 是否保存预训练checkpoint? (y/n): ",
        "读取输入失败，将跳过 checkpoint 保存",
    ) {
        std::fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).ok();
        match save_model_binary(&llm, DEFAULT_PRETRAINED_MODEL_PATH) {
            Ok(_) => println!("✓ 预训练checkpoint已保存"),
            Err(e) => println!("❌ 保存失败: {}", e),
        }
    }

    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║        阶段2: 指令微调 (Instruction Tuning) - 优化版     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 训练样本: {}", dataset.chat_training_data.len());
    println!("   • 最大epochs: 500 (早停patience=30)");
    println!("   • 学习率: 0.0001 (余弦退火, 无重启)");
    println!("   • 梯度累积: 1步 (稳定优先，后续可渐进恢复)
");

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("指令微调阶段");
    let actual_epochs = llm.train_monitored(chat_training_examples, 500, 0.0001, 30, 1);
    perf_monitor.stop("指令微调阶段");

    println!("✓ 指令微调完成，实际训练 {} epochs", actual_epochs);
    println!("
✅ 训练完成!");

    llm
}

/// 继续训练已加载的模型
fn continue_training_loaded_model(
    mut llm: LLM,
    perf_monitor: &mut PerformanceMonitor,
    freeze_attn: bool,
) -> LLM {
    println!("
🔄 继续训练模式");

    let dataset = load_default_dataset(perf_monitor);

    print!("
训练轮数 (默认50): ");
    flush_stdout();
    let epochs_input = read_line_or_empty("读取训练轮数失败，使用默认值 50");
    let epochs: usize = epochs_input.parse().unwrap_or(50);

    print!("学习率 (默认0.0001): ");
    flush_stdout();
    let lr_input = read_line_or_empty("读取学习率失败，使用默认值 0.0001");
    let lr: f32 = lr_input.parse().unwrap_or(0.0001);

    println!("
开始继续训练 ({} epochs, lr={})...
", epochs, lr);

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    apply_attention_freeze_if_requested(&mut llm, freeze_attn);

    llm.set_training_mode(true);
    perf_monitor.start("继续训练");
    llm.train_monitored(chat_training_examples, epochs, lr, 30, 1);
    perf_monitor.stop("继续训练");

    println!("
✅ 继续训练完成!");
    llm
}

/// 交互式保存模型
fn save_model_interactive(llm: &LLM) {
    println!("
选择保存格式:");
    println!("   1) 二进制格式 (.bin) - 推荐，文件小、速度快");
    println!("   2) JSON格式 (.json) - 人类可读，便于调试");
    println!("   3) 两种格式都保存");

    print!("
请选择 (1/2/3): ");
    flush_stdout();

    let format_choice = read_line_or_empty("读取输入失败，默认跳过保存");

    std::fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).ok();
    std::fs::create_dir_all("exports").ok();

    match format_choice.as_str() {
        "1" => {
            print!("文件名 (默认: {}): ", DEFAULT_MODEL_PATH);
            flush_stdout();

            let filename = read_line_or_empty("读取文件名失败，使用默认路径");
            let path = if filename.is_empty() {
                DEFAULT_MODEL_PATH
            } else {
                filename.as_str()
            };

            match save_model_binary(llm, path) {
                Ok(_) => println!("✅ 模型已保存: {}", path),
                Err(e) => println!("❌ 保存失败: {}", e),
            }
        }
        "2" => {
            print!("文件名 (默认: {}): ", DEFAULT_EXPORT_JSON_PATH);
            flush_stdout();

            let filename = read_line_or_empty("读取文件名失败，使用默认路径");
            let path = if filename.is_empty() {
                DEFAULT_EXPORT_JSON_PATH
            } else {
                filename.as_str()
            };

            match save_model_json(llm, path) {
                Ok(_) => println!("✅ 模型已保存: {}", path),
                Err(e) => println!("❌ 保存失败: {}", e),
            }
        }
        "3" => {
            println!("
保存二进制格式...");
            match save_model_binary(llm, DEFAULT_MODEL_PATH) {
                Ok(_) => println!("✓ 二进制格式已保存: {}", DEFAULT_MODEL_PATH),
                Err(e) => println!("✗ 二进制保存失败: {}", e),
            }

            println!("保存JSON格式...");
            match save_model_json(llm, DEFAULT_EXPORT_JSON_PATH) {
                Ok(_) => println!("✓ JSON格式已保存: {}", DEFAULT_EXPORT_JSON_PATH),
                Err(e) => println!("✗ JSON保存失败: {}", e),
            }
        }
        _ => println!("❌ 无效选项，跳过保存"),
    }
}

/// 交互模式
fn interactive_mode(llm: &mut LLM) {
    println!("
╔═══════════════════════════════════════════════════════════╗");
    println!("║                      交互模式                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("
💡 输入问题后按回车生成回答");
    println!("💡 输入 'exit' 退出程序");
    println!("💡 输入 'clear' 清空对话上下文");
    println!("💡 输入 'save' 保存当前模型");
    println!("💡 使用KV缓存加速推理（约10-100倍）
");

    llm.enable_kv_cache();

    let mut input = String::new();
    loop {
        input.clear();

        print!("👤 用户: ");
        flush_stdout();

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
            llm.clear_kv_cache();
            println!("✓ 对话上下文和KV缓存已清空
");
            continue;
        }

        if trimmed_input.eq_ignore_ascii_case("save") {
            save_model_interactive(llm);
            println!();
            continue;
        }

        let formatted_input = format!("用户：{}", trimmed_input);
        print!("🤖 模型: ");
        flush_stdout();

        let prediction = llm.predict_with_context(&formatted_input, 0.8, 0.9, 5);
        println!("{}
", prediction);

        if prediction.contains("</s>") {
            llm.clear_context();
        }
    }
}
