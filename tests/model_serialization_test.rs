// 模型序列化测试

use std::fs;

use llm::{
    Embeddings, LLM, load_model_binary, load_model_json,
    model_serialization::{SerializableAdam, SerializableLayer, SerializableModel},
    save_model_binary, save_model_json,
};

#[test]
fn test_binary_save_and_load() {
    let test_dir = "test_checkpoints_serialization_basic";
    assert!(fs::create_dir_all(test_dir).is_ok());

    let llm = LLM::default();
    let original_params = llm.total_parameters();
    let original_vocab_size = llm.vocab.len();

    let path = format!("{}/test_model.bin", test_dir);
    assert!(
        save_model_binary(&llm, &path).is_ok(),
        "Failed to save model"
    );
    assert!(std::path::Path::new(&path).exists());

    let loaded_llm = match load_model_binary(&path) {
        Ok(m) => m,
        Err(e) => {
            assert!(false, "Failed to load model: {}", e);
            return;
        }
    };

    assert_eq!(loaded_llm.total_parameters(), original_params);
    assert_eq!(loaded_llm.vocab.len(), original_vocab_size);
    assert_eq!(loaded_llm.network.len(), llm.network.len());

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(test_dir);

    println!("✓ 二进制格式保存/加载测试通过!");
}

#[test]
fn test_json_save_and_load() {
    let test_dir = "test_exports_serialization_basic";
    assert!(fs::create_dir_all(test_dir).is_ok());

    let llm = LLM::default();
    let original_params = llm.total_parameters();
    let original_vocab_size = llm.vocab.len();

    let path = format!("{}/test_model.json", test_dir);
    assert!(save_model_json(&llm, &path).is_ok(), "Failed to save model");
    assert!(std::path::Path::new(&path).exists());

    let loaded_llm = match load_model_json(&path) {
        Ok(m) => m,
        Err(e) => {
            assert!(false, "Failed to load model: {}", e);
            return;
        }
    };

    assert_eq!(loaded_llm.total_parameters(), original_params);
    assert_eq!(loaded_llm.vocab.len(), original_vocab_size);
    assert_eq!(loaded_llm.network.len(), llm.network.len());

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(test_dir);

    println!("✓ JSON格式保存/加载测试通过!");
}

#[test]
fn test_model_state_preservation() {
    use llm::adam::Adam;

    let adam = Adam::new((10, 20));
    let serialized = SerializableAdam::from_adam(&adam).expect("Adam 状态应能成功序列化");
    let deserialized = serialized.to_adam().expect("Adam 状态应能成功反序列化");

    assert_eq!(deserialized.timestep, adam.timestep);
    assert_eq!(deserialized.m.dim(), adam.m.dim());
    assert_eq!(deserialized.v.dim(), adam.v.dim());

    println!("✓ 优化器状态保存测试通过!");
}

#[test]
fn test_binary_save_rejects_non_finite_weights() {
    let test_dir = "test_checkpoints_non_finite";
    assert!(fs::create_dir_all(test_dir).is_ok());

    let mut llm = LLM::default();
    let embeddings = llm.network[0]
        .as_any_mut()
        .downcast_mut::<Embeddings>()
        .expect("首层应为 Embeddings");
    embeddings.token_embeddings[[0, 0]] = f32::NAN;

    let path = format!("{}/test_model_nan.bin", test_dir);
    let err = save_model_binary(&llm, &path).expect_err("包含 NaN 的模型不应被静默保存");
    assert!(
        err.to_string().contains("embeddings.token_embeddings"),
        "错误信息应指出损坏字段: {}",
        err
    );
    assert!(
        !std::path::Path::new(&path).exists(),
        "保存失败时不应落盘损坏模型"
    );

    let _ = fs::remove_dir_all(test_dir);
}

#[test]
fn test_json_load_rejects_shape_mismatch_in_serialized_weights() {
    let test_dir = "test_exports_corrupted_model";
    assert!(fs::create_dir_all(test_dir).is_ok());

    let llm = LLM::default();
    let path = format!("{}/test_model_corrupted.json", test_dir);
    save_model_json(&llm, &path).expect("基线 JSON 模型应保存成功");

    let json = fs::read_to_string(&path).expect("应能读取 JSON 模型");
    let mut model: SerializableModel =
        serde_json::from_str(&json).expect("应能解析刚写出的 JSON 模型");

    match model.layers.get_mut(0) {
        Some(SerializableLayer::Embeddings(embeddings)) => {
            embeddings.token_embeddings_shape.0 += 1;
        }
        other => panic!(
            "unexpected first layer: {:?}",
            other.map(|_| "non-embedding")
        ),
    }

    fs::write(
        &path,
        serde_json::to_string_pretty(&model).expect("应能重新写回损坏 JSON"),
    )
    .expect("应能覆盖损坏 JSON 文件");

    let err = match load_model_json(&path) {
        Ok(_) => panic!("损坏形状的 JSON 模型不应被静默加载"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("token_embeddings")
            || err.to_string().contains("Failed to rebuild layer 0"),
        "错误信息应指出损坏层: {}",
        err
    );

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(test_dir);
}
