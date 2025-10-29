//! # 数据集加载器
//!
//! 负责从 JSON 文件加载训练数据，支持预训练和对话微调两种数据格式。
//!
//! ## 数据格式
//!
//! 两种数据文件都是简单的 JSON 字符串数组：
//!
//! **预训练数据** (`data/pretraining_data.json`):
//! ```json
//! [
//!   "太阳从东方升起，在西方落下",
//!   "水由于重力而从高处流向低处",
//!   "山脉是高大而多岩石的地形"
//! ]
//! ```
//!
//! **对话训练数据** (`data/chat_training_data.json`):
//! ```json
//! [
//!   "用户：你好\n助手：你好！有什么可以帮助你的吗？",
//!   "用户：天气怎么样？\n助手：抱歉，我无法获取实时天气信息。"
//! ]
//! ```

use std::fs;

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
fn get_data_from_json(path: &str) -> Vec<String> {
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
