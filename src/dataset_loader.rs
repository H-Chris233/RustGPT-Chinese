use std::fs;

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

impl Dataset {
    pub fn new(pretraining_data_path: String, chat_training_data_path: String) -> Self {
        let pretraining_data = get_data_from_json(&pretraining_data_path);
        let chat_training_data = get_data_from_json(&chat_training_data_path);

        Dataset {
            pretraining_data,
            chat_training_data,
        }
    }
}

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
