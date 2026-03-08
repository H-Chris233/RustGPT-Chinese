// Tests for the Dataset struct in dataset_loader.rs

use llm::Dataset;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("rustgpt_dataset_loader_{name}_{}_{}", std::process::id(), stamp))
}

fn write_json_array(path: &Path, values: &[&str]) {
    let text = serde_json::to_string(values).expect("json serialization should succeed");
    fs::write(path, text).expect("fixture json should be writable");
}

#[test]
fn test_dataset_new_orders_directory_inputs_by_numeric_segments() {
    let root = unique_temp_dir("numeric_sort");
    let pre_dir = root.join("pretraining");
    let chat_dir = root.join("chat");
    fs::create_dir_all(&pre_dir).expect("pretraining fixture dir should exist");
    fs::create_dir_all(&chat_dir).expect("chat fixture dir should exist");

    write_json_array(&pre_dir.join("set1.json"), &["pre-1"]);
    write_json_array(&pre_dir.join("dataset2.json"), &["pre-2"]);
    write_json_array(&pre_dir.join("dataset10.json"), &["pre-10"]);

    write_json_array(
        &chat_dir.join("dataset2_qa_p2.json"),
        &["用户：chat-2
助手：ok-2"],
    );
    write_json_array(
        &chat_dir.join("dataset2_qa_p10.json"),
        &["用户：chat-10
助手：ok-10"],
    );
    write_json_array(
        &chat_dir.join("dataset10_realtime_boundary_p2.json"),
        &["用户：chat-rt
助手：ok-rt"],
    );

    let dataset = Dataset::new(
        pre_dir.to_string_lossy().to_string(),
        chat_dir.to_string_lossy().to_string(),
    );

    assert_eq!(dataset.pretraining_data, vec!["pre-1", "pre-2", "pre-10"]);
    assert_eq!(
        dataset.chat_training_data,
        vec![
            "用户：chat-2
助手：ok-2",
            "用户：chat-10
助手：ok-10",
            "用户：chat-rt
助手：ok-rt",
        ]
    );

    fs::remove_dir_all(&root).expect("fixture dir should be removable");
}
