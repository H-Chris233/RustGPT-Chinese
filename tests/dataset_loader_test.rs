// Tests for the Dataset struct in dataset_loader.rs

use llm::Dataset;

#[test]
fn test_dataset_new_json() {
    let dataset = Dataset::new(
        "data/pretraining_data.json".to_string(),
        "data/chat_training_data.json".to_string(),
    );
    assert!(
        !dataset.pretraining_data.is_empty(),
        "Pretraining data should not be empty"
    );
    assert!(
        !dataset.chat_training_data.is_empty(),
        "Chat training data should not be empty"
    );
    assert_eq!(dataset.pretraining_data[0], "太阳从东方升起，在西方落下");
    assert!(dataset.chat_training_data[0].starts_with("User: 什么是雨？"));
}
