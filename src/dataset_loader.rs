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
//! 传入目录路径时，会加载目录下所有 `.json` 文件（按文件名中的所有数字段做自然排序）并合并语料。
//!
//! 例如：
//! - `data/pretraining/set1.json`
//! - `data/pretraining/dataset2.json`
//! - `data/pretraining/dataset3.json`
//!
//! 以后新增 `dataset4.json` / `dataset5.json` / `dataset6.json` 会自动被加载器纳入训练。
//!
//! ### 文件模式（兼容）
//! 传入单个文件路径时，会自动尝试拼接同目录下的 `*_extra.json`：
//! - 例如 `foo.json` 会额外尝试加载 `foo_extra.json`
//! - 这是兼容性约定；教学上更推荐直接传目录路径，规则更显式

use std::fs;
use std::path::{Path, PathBuf};

/// 数据集 JSON 文件大小上限（防止恶意/损坏文件导致 OOM）。
const MAX_DATASET_JSON_BYTES: u64 = 128 * 1024 * 1024; // 128MiB

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

/// **从 JSON 文件加载数据**
///
/// 读取 `path` 指向的 JSON 字符串数组，并支持：
/// - 自动拼接同目录下的 `*_extra.json`（例如 `foo.json -> foo_extra.json`）
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
    data
}

fn compare_dataset_paths(a: &Path, b: &Path) -> std::cmp::Ordering {
    let a_name = a.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
    let b_name = b.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();

    let a_numbers = extract_number_segments(&a_name);
    let b_numbers = extract_number_segments(&b_name);

    compare_number_segments(&a_numbers, &b_numbers)
        .then_with(|| natural_compare_names(&a_name, &b_name))
}

fn extract_number_segments(s: &str) -> Vec<u64> {
    let mut numbers = Vec::new();
    let mut start: Option<usize> = None;

    for (i, ch) in s.char_indices() {
        if ch.is_ascii_digit() {
            start.get_or_insert(i);
        } else if let Some(st) = start.take() {
            if let Ok(value) = s[st..i].parse() {
                numbers.push(value);
            }
        }
    }

    if let Some(st) = start {
        if let Ok(value) = s[st..].parse() {
            numbers.push(value);
        }
    }

    numbers
}

fn compare_number_segments(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    match (a.is_empty(), b.is_empty()) {
        (false, false) => a
            .iter()
            .zip(b.iter())
            .find_map(|(x, y)| {
                let ord = x.cmp(y);
                (ord != std::cmp::Ordering::Equal).then_some(ord)
            })
            .unwrap_or_else(|| a.len().cmp(&b.len())),
        (false, true) => std::cmp::Ordering::Less,
        (true, false) => std::cmp::Ordering::Greater,
        (true, true) => std::cmp::Ordering::Equal,
    }
}

fn natural_compare_names(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_pos = 0usize;
    let mut b_pos = 0usize;

    loop {
        match (next_name_chunk(a, a_pos), next_name_chunk(b, b_pos)) {
            (Some((a_chunk, a_is_digit, next_a)), Some((b_chunk, b_is_digit, next_b))) => {
                let ord = match (a_is_digit, b_is_digit) {
                    (true, true) => compare_numeric_chunk(a_chunk, b_chunk),
                    _ => a_chunk.cmp(b_chunk),
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
                a_pos = next_a;
                b_pos = next_b;
            }
            (None, None) => return a.cmp(b),
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
        }
    }
}

fn next_name_chunk(s: &str, start: usize) -> Option<(&str, bool, usize)> {
    if start >= s.len() {
        return None;
    }

    let mut chars = s[start..].char_indices();
    let (_, first) = chars.next()?;
    let is_digit = first.is_ascii_digit();
    let mut end = s.len();

    for (offset, ch) in chars {
        if ch.is_ascii_digit() != is_digit {
            end = start + offset;
            break;
        }
    }

    Some((&s[start..end], is_digit, end))
}

fn compare_numeric_chunk(a: &str, b: &str) -> std::cmp::Ordering {
    let a_trimmed = a.trim_start_matches('0');
    let b_trimmed = b.trim_start_matches('0');
    let a_norm = if a_trimmed.is_empty() { "0" } else { a_trimmed };
    let b_norm = if b_trimmed.is_empty() { "0" } else { b_trimmed };

    a_norm
        .len()
        .cmp(&b_norm.len())
        .then_with(|| a_norm.cmp(b_norm))
        .then_with(|| a.len().cmp(&b.len()))
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


#[cfg(test)]
mod tests {
    use super::compare_dataset_paths;
    use std::path::PathBuf;

    #[test]
    fn compare_dataset_paths_orders_by_all_numeric_segments() {
        let mut paths = vec![
            PathBuf::from("dataset2_qa_p10.json"),
            PathBuf::from("dataset10_realtime_boundary_p2.json"),
            PathBuf::from("dataset2_qa_p2.json"),
            PathBuf::from("set1.json"),
            PathBuf::from("dataset9_realtime_boundary_p3.json"),
            PathBuf::from("dataset2_qa_p02.json"),
        ];

        paths.sort_by(|a, b| compare_dataset_paths(a, b));

        let names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert_eq!(
            names,
            vec![
                "set1.json",
                "dataset2_qa_p2.json",
                "dataset2_qa_p02.json",
                "dataset2_qa_p10.json",
                "dataset9_realtime_boundary_p3.json",
                "dataset10_realtime_boundary_p2.json",
            ]
        );
    }
}
