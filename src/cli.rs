use std::fmt;

#[derive(Debug, Clone, Default)]
pub struct GlobalOptions {
    pub no_interactive: bool,
}

#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub pretraining_dir: String,
    pub chat_dir: String,
    pub checkpoint_dir: String,
    pub pretrain_epochs: usize,
    pub finetune_epochs: usize,
    pub lr: f32,
    pub patience: usize,
    pub accum: usize,
    pub freeze_attn: bool,
    pub save_final: Option<String>,
    pub save_pretrained: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResumeOptions {
    pub checkpoint_dir: String,
    pub pretraining_dir: String,
    pub chat_dir: String,
    pub resume_from: Option<String>,
    pub epochs: usize,
    pub lr: Option<f32>,
    pub patience: usize,
    pub save_final: String,
}

#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub model: String,
    pub prompt: Option<String>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub model: String,
    pub out_bin: Option<String>,
    pub out_json: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Command {
    Train(TrainOptions),
    Resume(ResumeOptions),
    Chat(ChatOptions),
    Export(ExportOptions),
    Info,
}

#[derive(Debug, Clone)]
pub enum HelpTopic {
    Global,
    Command(String),
}

#[derive(Debug, Clone)]
pub enum Action {
    Run { global: GlobalOptions, command: Command },
    PrintHelp { topic: HelpTopic, exit_code: i32 },
    PrintVersion,
}

#[derive(Debug, Clone)]
pub struct CliError {
    message: String,
}

impl CliError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn help_global() -> String {
    [
        "用法:",
        "  llm <命令> [选项]",
        "",
        "命令:",
        "  train     从头训练（预训练 + 指令微调）",
        "  resume    从检查点继续训练",
        "  chat      推理（REPL 或单次 --prompt）",
        "  export    导出模型（bin/json）",
        "  info      打印默认路径与说明",
        "",
        "全局选项:",
        "  --no-interactive    禁止任何交互/读取 stdin（用于脚本/CI）",
        "  --help              显示帮助（也可：llm <命令> --help）",
        "  --version           显示版本",
        "",
        "约定:",
        "  - 仅支持 --key=value 与 --flag 形式（不支持 --key value）",
        "",
        "示例:",
        "  llm train",
        "  llm resume --checkpoint-dir=checkpoints",
        "  llm chat --model=checkpoints/model_final.bin",
        "  llm chat --model=checkpoints/model_final.bin --prompt=山脉是如何形成的？",
        "  llm export --model=checkpoints/model_final.bin --out-json=exports/model_final.json",
        "",
    ]
    .join("\n")
}

pub fn help_command(command: &str) -> String {
    match command {
        "train" => [
            "用法:",
            "  llm train [选项]",
            "",
            "选项（仅支持 --key=value 与 --flag）:",
            "  --pretraining-dir=data/pretraining",
            "  --chat-dir=data/chat",
            "  --checkpoint-dir=checkpoints",
            "  --pretrain-epochs=500",
            "  --finetune-epochs=500",
            "  --lr=0.0001",
            "  --patience=30",
            "  --accum=1",
            "  --freeze-attn",
            "  --save-final=checkpoints/model_final.bin",
            "  --no-save-final",
            "  --save-pretrained=checkpoints/model_pretrained.bin",
            "",
            "说明:",
            "  - pretrain/finetune epochs 为 0 表示跳过该阶段",
            "",
        ]
        .join("\n"),
        "resume" => [
            "用法:",
            "  llm resume [选项]",
            "",
            "选项:",
            "  --checkpoint-dir=checkpoints",
            "  --pretraining-dir=data/pretraining",
            "  --chat-dir=data/chat",
            "  --resume-from=checkpoints/checkpoint_best_epoch_XX_loss_*.bin    （可选）",
            "  --epochs=500",
            "  --lr=0.0001                                                     （可选，不填则继承 checkpoint）",
            "  --patience=30",
            "  --save-final=checkpoints/model_final.bin",
            "",
        ]
        .join("\n"),
        "chat" => [
            "用法:",
            "  llm chat --model=PATH [选项]",
            "",
            "选项:",
            "  --model=checkpoints/model_final.bin         （必填，支持 .bin 或 .json）",
            "  --prompt=你好                                （可选：单次推理并退出）",
            "  --temperature=0.8",
            "  --top-p=0.9",
            "  --top-k=5",
            "",
            "交互模式（REPL）命令:",
            "  :help  :exit  :clear  :save <path>  :set temperature=...  :set top_p=...  :set top_k=...",
            "",
            "说明:",
            "  - 若启用 --no-interactive，则必须提供 --prompt（否则直接报错退出）",
            "",
        ]
        .join("\n"),
        "export" => [
            "用法:",
            "  llm export --model=PATH (--out-bin=PATH | --out-json=PATH) [选项]",
            "",
            "选项:",
            "  --model=checkpoints/model_final.bin         （必填，支持 .bin 或 .json）",
            "  --out-bin=exports/model.bin",
            "  --out-json=exports/model.json",
            "",
        ]
        .join("\n"),
        "info" => ["用法:", "  llm info", ""].join("\n"),
        _ => help_global(),
    }
}

pub fn parse(argv: Vec<String>) -> Result<Action, CliError> {
    if argv.is_empty() {
        return Ok(Action::PrintHelp {
            topic: HelpTopic::Global,
            exit_code: 2,
        });
    }

    let mut global = GlobalOptions::default();
    let mut want_help = false;
    let mut want_version = false;
    let mut tokens: Vec<String> = Vec::new();

    for arg in argv {
        match arg.as_str() {
            "--no-interactive" => global.no_interactive = true,
            "--help" => want_help = true,
            "--version" => want_version = true,
            _ => tokens.push(arg),
        }
    }

    if want_version {
        return Ok(Action::PrintVersion);
    }

    if tokens.is_empty() {
        return Ok(Action::PrintHelp {
            topic: HelpTopic::Global,
            exit_code: if want_help { 0 } else { 2 },
        });
    }

    let command = tokens[0].as_str();
    if want_help {
        return Ok(Action::PrintHelp {
            topic: HelpTopic::Command(command.to_string()),
            exit_code: 0,
        });
    }

    let rest = &tokens[1..];
    let command = match command {
        "train" => Command::Train(parse_train(rest)?),
        "resume" => Command::Resume(parse_resume(rest)?),
        "chat" => Command::Chat(parse_chat(rest)?),
        "export" => Command::Export(parse_export(rest)?),
        "info" => {
            ensure_no_options("info", rest)?;
            Command::Info
        }
        other => {
            return Err(CliError::new(format!(
                "未知命令: {}\n\n{}",
                other,
                help_global()
            )));
        }
    };

    Ok(Action::Run { global, command })
}

fn ensure_no_options(command: &str, args: &[String]) -> Result<(), CliError> {
    if args.is_empty() {
        return Ok(());
    }
    Err(CliError::new(format!(
        "{} 不接受任何选项，但收到了: {:?}",
        command, args
    )))
}

fn ensure_flag_form(arg: &str) -> Result<(), CliError> {
    if !arg.starts_with("--") {
        return Err(CliError::new(format!(
            "只支持 --key=value 与 --flag，发现了位置参数: {}",
            arg
        )));
    }

    if arg.starts_with("--") && !arg.contains('=') {
        // 允许纯 flag（无 '='），其余情况在各命令解析里校验是否为已知 flag。
        return Ok(());
    }

    if arg.contains('=') {
        // 拒绝 --key value：即 arg 为 --key 而 value 在下一个 token。
        // 这里无法直接识别，但会在命令解析中把不含 '=' 的 --key 当未知 flag 报错。
        return Ok(());
    }

    Ok(())
}

fn split_kv(arg: &str) -> Result<(&str, &str), CliError> {
    let Some((k, v)) = arg.split_once('=') else {
        return Err(CliError::new(format!(
            "参数需要使用 --key=value 形式: {}",
            arg
        )));
    };
    if v.is_empty() {
        return Err(CliError::new(format!("参数值不能为空: {}", arg)));
    }
    Ok((k, v))
}

fn parse_train(args: &[String]) -> Result<TrainOptions, CliError> {
    let mut save_final_is_default = true;
    let mut options = TrainOptions {
        pretraining_dir: "data/pretraining".to_string(),
        chat_dir: "data/chat".to_string(),
        checkpoint_dir: "checkpoints".to_string(),
        pretrain_epochs: 500,
        finetune_epochs: 500,
        lr: 0.0001,
        patience: 30,
        accum: 1,
        freeze_attn: false,
        save_final: Some("checkpoints/model_final.bin".to_string()),
        save_pretrained: None,
    };

    for arg in args {
        ensure_flag_form(arg)?;
        match arg.as_str() {
            "--freeze-attn" => options.freeze_attn = true,
            "--no-save-final" => {
                options.save_final = None;
                save_final_is_default = false;
            }
            _ if arg.starts_with("--") && arg.contains('=') => {
                let (k, v) = split_kv(arg)?;
                match k {
                    "--pretraining-dir" => options.pretraining_dir = v.to_string(),
                    "--chat-dir" => options.chat_dir = v.to_string(),
                    "--checkpoint-dir" => {
                        options.checkpoint_dir = v.to_string();
                        if save_final_is_default {
                            options.save_final =
                                Some(format!("{}/model_final.bin", options.checkpoint_dir));
                        }
                    }
                    "--pretrain-epochs" => {
                        options.pretrain_epochs = v.parse().map_err(|_| {
                            CliError::new(format!("--pretrain-epochs 需要是整数: {}", v))
                        })?
                    }
                    "--finetune-epochs" => {
                        options.finetune_epochs = v.parse().map_err(|_| {
                            CliError::new(format!("--finetune-epochs 需要是整数: {}", v))
                        })?
                    }
                    "--lr" => {
                        options.lr = v
                            .parse()
                            .map_err(|_| CliError::new(format!("--lr 需要是浮点数: {}", v)))?
                    }
                    "--patience" => {
                        options.patience = v.parse().map_err(|_| {
                            CliError::new(format!("--patience 需要是整数: {}", v))
                        })?
                    }
                    "--accum" => {
                        options.accum = v
                            .parse()
                            .map_err(|_| CliError::new(format!("--accum 需要是整数: {}", v)))?
                    }
                    "--save-final" => {
                        options.save_final = Some(v.to_string());
                        save_final_is_default = false;
                    }
                    "--save-pretrained" => options.save_pretrained = Some(v.to_string()),
                    other => {
                        return Err(CliError::new(format!(
                            "train 不支持的参数: {}\n\n{}",
                            other,
                            help_command("train")
                        )));
                    }
                }
            }
            other => {
                return Err(CliError::new(format!(
                    "train 不支持的 flag: {}\n\n{}",
                    other,
                    help_command("train")
                )));
            }
        }
    }

    Ok(options)
}

fn parse_resume(args: &[String]) -> Result<ResumeOptions, CliError> {
    let mut save_final_is_default = true;
    let mut options = ResumeOptions {
        checkpoint_dir: "checkpoints".to_string(),
        pretraining_dir: "data/pretraining".to_string(),
        chat_dir: "data/chat".to_string(),
        resume_from: None,
        epochs: 500,
        lr: None,
        patience: 30,
        save_final: "checkpoints/model_final.bin".to_string(),
    };

    for arg in args {
        ensure_flag_form(arg)?;
        match arg.as_str() {
            _ if arg.starts_with("--") && arg.contains('=') => {
                let (k, v) = split_kv(arg)?;
                match k {
                    "--checkpoint-dir" => {
                        options.checkpoint_dir = v.to_string();
                        if save_final_is_default {
                            options.save_final =
                                format!("{}/model_final.bin", options.checkpoint_dir);
                        }
                    }
                    "--pretraining-dir" => options.pretraining_dir = v.to_string(),
                    "--chat-dir" => options.chat_dir = v.to_string(),
                    "--resume-from" => options.resume_from = Some(v.to_string()),
                    "--epochs" => {
                        options.epochs = v
                            .parse()
                            .map_err(|_| CliError::new(format!("--epochs 需要是整数: {}", v)))?
                    }
                    "--lr" => {
                        options.lr = Some(
                            v.parse()
                                .map_err(|_| CliError::new(format!("--lr 需要是浮点数: {}", v)))?,
                        )
                    }
                    "--patience" => {
                        options.patience = v.parse().map_err(|_| {
                            CliError::new(format!("--patience 需要是整数: {}", v))
                        })?
                    }
                    "--save-final" => {
                        options.save_final = v.to_string();
                        save_final_is_default = false;
                    }
                    other => {
                        return Err(CliError::new(format!(
                            "resume 不支持的参数: {}\n\n{}",
                            other,
                            help_command("resume")
                        )));
                    }
                }
            }
            other => {
                return Err(CliError::new(format!(
                    "resume 不支持的 flag: {}\n\n{}",
                    other,
                    help_command("resume")
                )));
            }
        }
    }

    Ok(options)
}

fn parse_chat(args: &[String]) -> Result<ChatOptions, CliError> {
    let mut model: Option<String> = None;
    let mut options = ChatOptions {
        model: String::new(),
        prompt: None,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 5,
    };

    for arg in args {
        ensure_flag_form(arg)?;
        match arg.as_str() {
            _ if arg.starts_with("--") && arg.contains('=') => {
                let (k, v) = split_kv(arg)?;
                match k {
                    "--model" => model = Some(v.to_string()),
                    "--prompt" => options.prompt = Some(v.to_string()),
                    "--temperature" => {
                        options.temperature = v.parse().map_err(|_| {
                            CliError::new(format!("--temperature 需要是浮点数: {}", v))
                        })?
                    }
                    "--top-p" => {
                        options.top_p = v
                            .parse()
                            .map_err(|_| CliError::new(format!("--top-p 需要是浮点数: {}", v)))?
                    }
                    "--top-k" => {
                        options.top_k = v
                            .parse()
                            .map_err(|_| CliError::new(format!("--top-k 需要是整数: {}", v)))?
                    }
                    other => {
                        return Err(CliError::new(format!(
                            "chat 不支持的参数: {}\n\n{}",
                            other,
                            help_command("chat")
                        )));
                    }
                }
            }
            other => {
                return Err(CliError::new(format!(
                    "chat 不支持的 flag: {}\n\n{}",
                    other,
                    help_command("chat")
                )));
            }
        }
    }

    let Some(model) = model else {
        return Err(CliError::new(format!(
            "chat 需要 --model=PATH\n\n{}",
            help_command("chat")
        )));
    };
    options.model = model;

    Ok(options)
}

fn parse_export(args: &[String]) -> Result<ExportOptions, CliError> {
    let mut model: Option<String> = None;
    let mut out_bin: Option<String> = None;
    let mut out_json: Option<String> = None;

    for arg in args {
        ensure_flag_form(arg)?;
        if arg.starts_with("--") && arg.contains('=') {
            let (k, v) = split_kv(arg)?;
            match k {
                "--model" => model = Some(v.to_string()),
                "--out-bin" => out_bin = Some(v.to_string()),
                "--out-json" => out_json = Some(v.to_string()),
                other => {
                    return Err(CliError::new(format!(
                        "export 不支持的参数: {}\n\n{}",
                        other,
                        help_command("export")
                    )));
                }
            }
        } else {
            return Err(CliError::new(format!(
                "export 不支持的 flag: {}\n\n{}",
                arg,
                help_command("export")
            )));
        }
    }

    let Some(model) = model else {
        return Err(CliError::new(format!(
            "export 需要 --model=PATH\n\n{}",
            help_command("export")
        )));
    };

    if out_bin.is_none() && out_json.is_none() {
        return Err(CliError::new(format!(
            "export 需要至少一个输出参数：--out-bin=PATH 或 --out-json=PATH\n\n{}",
            help_command("export")
        )));
    }

    Ok(ExportOptions {
        model,
        out_bin,
        out_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_prints_help_with_nonzero_code() {
        let action = parse(Vec::new()).unwrap();
        match action {
            Action::PrintHelp { topic, exit_code } => {
                assert!(matches!(topic, HelpTopic::Global));
                assert_ne!(exit_code, 0);
            }
            _ => panic!("expected help"),
        }
    }

    #[test]
    fn parse_help_is_exit_zero() {
        let action = parse(vec!["--help".to_string()]).unwrap();
        match action {
            Action::PrintHelp { exit_code, .. } => assert_eq!(exit_code, 0),
            _ => panic!("expected help"),
        }
    }

    #[test]
    fn parse_train_defaults() {
        let action = parse(vec!["train".to_string()]).unwrap();
        match action {
            Action::Run { command, .. } => match command {
                Command::Train(options) => {
                    assert_eq!(options.pretrain_epochs, 500);
                    assert_eq!(options.finetune_epochs, 500);
                    assert_eq!(options.checkpoint_dir, "checkpoints");
                    assert_eq!(options.save_final.as_deref(), Some("checkpoints/model_final.bin"));
                }
                _ => panic!("expected train"),
            },
            _ => panic!("expected run"),
        }
    }

    #[test]
    fn parse_rejects_positional_args() {
        let err = parse(vec!["train".to_string(), "oops".to_string()]).unwrap_err();
        assert!(err.to_string().contains("位置参数"));
    }

    #[test]
    fn parse_resume_defaults() {
        let action = parse(vec!["resume".to_string()]).unwrap();
        match action {
            Action::Run { command, .. } => match command {
                Command::Resume(options) => {
                    assert_eq!(options.checkpoint_dir, "checkpoints");
                    assert_eq!(options.pretraining_dir, "data/pretraining");
                    assert_eq!(options.chat_dir, "data/chat");
                    assert_eq!(options.epochs, 500);
                    assert_eq!(options.lr, None);
                    assert_eq!(options.patience, 30);
                    assert_eq!(options.save_final, "checkpoints/model_final.bin");
                }
                _ => panic!("expected resume"),
            },
            _ => panic!("expected run"),
        }
    }
}
