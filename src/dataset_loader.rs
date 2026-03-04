//! # 数据集加载器
//!
//! 负责从 JSON 文件加载训练数据，支持预训练和对话微调两种数据格式。
//!
//! ## 数据格式
//!
//! 两种数据文件都是简单的 JSON 字符串数组：
//!
//! **预训练数据**（文件模式示例，目录模式见下文）:
//! ```json
//! [
//!   "太阳从东方升起，在西方落下",
//!   "水由于重力而从高处流向低处",
//!   "山脉是高大而多岩石的地形"
//! ]
//! ```
//!
//! **对话训练数据**（文件模式示例，目录模式见下文）:
//! ```json
//! [
//!   "用户：你好\n助手：你好！有什么可以帮助你的吗？",
//!   "用户：天气怎么样？\n助手：抱歉，我无法获取实时天气信息。"
//! ]
//! ```
//!
//! ### 目录模式（推荐）
//! 传入目录路径时，会加载目录下所有 `.json` 文件（按文件名中的数字序号排序），并将合并后的语料按
//! `DATASET_TOTAL_MULTIPLIER` 进行重复采样扩展。
//!
//! 例如：
//! - `data/pretraining/set1.json`
//! - `data/pretraining/dataset2.json`
//! - `data/pretraining/dataset3.json`
//!
//! 以后新增 `dataset4.json` / `dataset5.json` / `dataset6.json` 会自动被加载器纳入训练。
//!
//! ### 文件模式（兼容）
//! 传入单个文件路径时，会自动尝试追加同目录下的 `*_extra.json`（若存在），然后再按
//! `DATASET_TOTAL_MULTIPLIER` 扩展。

use std::fs;
use std::path::{Path, PathBuf};

/// 数据集 JSON 文件大小上限（防止恶意/损坏文件导致 OOM）。
const MAX_DATASET_JSON_BYTES: u64 = 128 * 1024 * 1024; // 128MiB
/// 训练时使用的数据量相对基础数据集的倍数（通过重复采样实现）。
///
/// 说明：为满足“数据集总量扩大”的需求，这里会将加载到的语料重复采样到指定倍数。
const DATASET_TOTAL_MULTIPLIER: usize = 8;
/// 防止误配置/误加载导致内存爆炸：扩展后的数据条目上限。
const MAX_EXPANDED_DATASET_ITEMS: usize = 1_000_000;

/// **数据集结构体**
///
/// 包含预训练数据和对话训练数据两部分。
pub struct Dataset {
    /// 预训练数据：用于学习世界知识和语言模式
    pub pretraining_data: Vec<String>,
    /// 对话训练数据：用于学习对话和指令遵循能力
    pub chat_training_data: Vec<String>,
}

impl Dataset {
    /// **创建新的数据集实例**
    ///
    /// 从指定路径加载预训练数据和对话数据。
    ///
    /// # 参数
    /// - `pretraining_data_path`: 预训练数据 JSON 文件路径
    /// - `chat_training_data_path`: 对话数据 JSON 文件路径
    ///
    /// # 返回值
    /// 初始化的数据集实例
    ///
    /// # 错误处理
    /// 如果文件读取或解析失败，会记录错误并返回空向量
    pub fn new(pretraining_data_path: String, chat_training_data_path: String) -> Self {
        let pretraining_data = get_data_from_json(&pretraining_data_path);
        let chat_training_data = get_data_from_json(&chat_training_data_path);

        Dataset {
            pretraining_data,
            chat_training_data,
        }
    }
}

/// **从 JSON 文件加载数据（并按需扩展）**
///
/// 读取 `path` 指向的 JSON 字符串数组，并支持：
/// - 自动追加同目录下的 `*_extra.json`（若存在）
/// - 将最终语料重复采样到“基础语料”的指定倍数（见 `DATASET_TOTAL_MULTIPLIER`）
///
/// # 参数
/// - `path`: JSON 文件路径
///
/// # 返回值
/// - 成功：包含所有字符串的向量（已按倍数扩展）
/// - 失败：空向量，并记录错误日志
///
/// # 示例
/// ```json
/// ["句子1", "句子2", "句子3"]
/// ```
fn get_data_from_json(path: &str) -> Vec<String> {
    let meta = match fs::metadata(path) {
        Ok(meta) => meta,
        Err(e) => {
            log::error!("读取数据路径元信息失败 ({}): {}", path, e);
            return Vec::new();
        }
    };

    if meta.is_dir() {
        return load_dataset_from_dir(path);
    }

    load_dataset_from_file(path)
}

fn load_dataset_from_file(path: &str) -> Vec<String> {
    let base_data = load_json_string_list(path);
    if base_data.is_empty() {
        return Vec::new();
    }

    let mut data = base_data;

    // 约定：若存在同目录下的 *_extra.json，则自动追加；用于放置额外语料而不改动原始文件。
    if let Some(extra_path) = build_extra_dataset_path(path) {
        if extra_path.exists() {
            let extra_str = extra_path.to_string_lossy().to_string();
            let extra_data = load_json_string_list(&extra_str);
            data.extend(extra_data);
        }
    }

    let base_len = data.len();
    expand_dataset_to_multiplier(&mut data, DATASET_TOTAL_MULTIPLIER, base_len);
    data
}

fn load_dataset_from_dir(dir: &str) -> Vec<String> {
    let mut json_files: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
            .collect(),
        Err(e) => {
            log::error!("读取数据目录失败 ({}): {}", dir, e);
            return Vec::new();
        }
    };

    if json_files.is_empty() {
        log::error!("数据目录为空或无 .json 文件: {}", dir);
        return Vec::new();
    }

    json_files.sort_by(|a, b| compare_dataset_paths(a, b));

    let mut data: Vec<String> = Vec::new();
    for p in &json_files {
        let p_str = p.to_string_lossy().to_string();
        data.extend(load_json_string_list(&p_str));
    }

    let base_len = data.len();
    if base_len == 0 {
        log::error!("数据目录全部为空或解析失败: {}", dir);
        return Vec::new();
    }

    expand_dataset_to_multiplier(&mut data, DATASET_TOTAL_MULTIPLIER, base_len);
    data
}

fn compare_dataset_paths(a: &Path, b: &Path) -> std::cmp::Ordering {
    let a_name = a.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
    let b_name = b.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();

    let a_num = extract_first_number(&a_name);
    let b_num = extract_first_number(&b_name);

    match (a_num, b_num) {
        (Some(x), Some(y)) => x.cmp(&y).then_with(|| a_name.cmp(&b_name)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a_name.cmp(&b_name),
    }
}

fn extract_first_number(s: &str) -> Option<u64> {
    let mut start: Option<usize> = None;
    for (i, ch) in s.char_indices() {
        if ch.is_ascii_digit() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(st) = start {
            return s[st..i].parse().ok();
        }
    }

    start.and_then(|st| s[st..].parse().ok())
}

fn build_extra_dataset_path(path: &str) -> Option<PathBuf> {
    let p = Path::new(path);
    let parent = p.parent()?;
    let stem = p.file_stem()?.to_string_lossy();
    let ext = p.extension().map(|s| s.to_string_lossy()).unwrap_or_default();

    let extra_file = if ext.is_empty() {
        format!("{stem}_extra")
    } else {
        format!("{stem}_extra.{ext}")
    };

    Some(parent.join(extra_file))
}

fn expand_dataset_to_multiplier(data: &mut Vec<String>, multiplier: usize, base_len: usize) {
    if multiplier <= 1 {
        return;
    }

    let Some(target_len) = base_len.checked_mul(multiplier) else {
        log::error!("数据集目标长度计算溢出，跳过扩展");
        return;
    };

    if target_len > MAX_EXPANDED_DATASET_ITEMS {
        log::error!(
            "扩展后的数据集条目数过大(>{})，跳过扩展",
            MAX_EXPANDED_DATASET_ITEMS
        );
        return;
    }

    if data.len() >= target_len {
        data.truncate(target_len);
        return;
    }

    // 用当前 data 作为“种子”，重复采样直到达到目标长度（extra 语料也会参与重复采样）。
    let seed = data.clone();
    if seed.is_empty() {
        return;
    }

    data.reserve(target_len.saturating_sub(data.len()));
    while data.len() < target_len {
        let remaining = target_len - data.len();
        let take = remaining.min(seed.len());
        data.extend_from_slice(&seed[..take]);
    }
}

/// **从 JSON 文件加载数据**
///
/// 读取 JSON 文件并解析为字符串向量。
///
/// # 参数
/// - `path`: JSON 文件路径
///
/// # 返回值
/// - 成功：包含所有字符串的向量
/// - 失败：空向量，并记录错误日志
///
/// # 示例
/// ```json
/// ["句子1", "句子2", "句子3"]
/// ```
fn load_json_string_list(path: &str) -> Vec<String> {
    // 先做文件大小上限校验，避免一次性 read_to_string 导致内存耗尽
    match fs::metadata(path) {
        Ok(meta) if meta.len() > MAX_DATASET_JSON_BYTES => {
            log::error!(
                "数据文件过大(>{} bytes)，拒绝加载: {}",
                MAX_DATASET_JSON_BYTES,
                path
            );
            return Vec::new();
        }
        Ok(_) => {}
        Err(e) => {
            log::error!("读取数据文件元信息失败 ({}): {}", path, e);
            return Vec::new();
        }
    }

    match fs::read_to_string(path) {
        Ok(data_json) => match serde_json::from_str::<Vec<String>>(&data_json) {
            Ok(data) => data,
            Err(e) => {
                log::error!("解析JSON数据失败 ({}): {}", path, e);
                Vec::new()
            }
        },
        Err(e) => {
            log::error!("读取数据文件失败 ({}): {}", path, e);
            Vec::new()
        }
    }
}
