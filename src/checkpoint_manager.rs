//! 检查点管理器模块
//!
//! 提供训练检查点的保存、加载和管理功能，支持多种保存策略：
//! - **Best**: 保存最佳模型（基于loss）
//! - **Last**: 保存最新模型
//! - **Periodic**: 周期性保存（每N个epoch）
//!
//! 检查点包含完整的训练状态：
//! - 模型参数（所有层的权重）
//! - 优化器状态（Adam的m、v动量和timestep）
//! - 训练元数据（epoch、loss、学习率等）
//! - 词汇表

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::llm::LLM;
use crate::model_serialization::SerializableModel;
use crate::transformer::TransformerBlock;

/// 反序列化输入大小上限（防止恶意/损坏文件导致 OOM）。
const BINCODE_DECODE_LIMIT_BYTES: usize = 512 * 1024 * 1024; // 512MiB

/// 检查点元数据
#[derive(Clone, Debug, Encode, Decode, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// 当前epoch
    pub epoch: usize,
    /// 平均loss
    pub loss: f32,
    /// 当前学习率
    pub learning_rate: f32,
    /// 保存时间戳
    pub timestamp: String,
    /// 训练阶段标识（如"pretraining", "instruction_tuning"）
    pub phase: String,
}

/// 完整的检查点数据
#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Checkpoint {
    /// 模型状态
    pub model: SerializableModel,
    /// 元数据
    pub metadata: CheckpointMetadata,
}

/// 检查点保存策略
#[derive(Clone, Debug)]
pub enum CheckpointStrategy {
    /// 仅保存最佳模型
    Best,
    /// 保存最新模型
    Last,
    /// 周期性保存（每N个epoch）
    Periodic(usize),
    /// 组合策略：最佳 + 最新
    BestAndLast,
    /// 组合策略：最佳 + 周期性
    BestAndPeriodic(usize),
}

/// 检查点管理器
pub struct CheckpointManager {
    /// 检查点保存目录
    checkpoint_dir: PathBuf,
    /// 保存策略
    strategy: CheckpointStrategy,
    /// 当前最佳loss
    best_loss: f32,
    /// 最佳检查点的epoch
    best_epoch: usize,
    /// 保留的最佳检查点数量
    keep_best_n: usize,
}

impl CheckpointManager {
    /// 创建新的检查点管理器
    ///
    /// # 参数
    /// - `checkpoint_dir`: 检查点保存目录
    /// - `strategy`: 保存策略
    /// - `keep_best_n`: 保留的最佳检查点数量（默认3）
    pub fn new<P: AsRef<Path>>(
        checkpoint_dir: P,
        strategy: CheckpointStrategy,
        keep_best_n: usize,
    ) -> Result<Self, String> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // 创建检查点目录
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir)
                .map_err(|e| format!("创建检查点目录失败: {}", e))?;
        }

        Ok(Self {
            checkpoint_dir,
            strategy,
            best_loss: f32::INFINITY,
            best_epoch: 0,
            keep_best_n,
        })
    }

    /// 检查是否应该保存检查点
    pub fn should_save(&self, epoch: usize, current_loss: f32) -> bool {
        match &self.strategy {
            CheckpointStrategy::Best => current_loss < self.best_loss,
            CheckpointStrategy::Last => true,
            // 说明：n==0 会导致 `% 0` panic，因此这里显式防御。
            // 与 `save_checkpoint()` 的语义对齐：n==0 视为“永不按周期保存”。
            CheckpointStrategy::Periodic(n) => *n > 0 && epoch % n == 0,
            CheckpointStrategy::BestAndLast => true,
            CheckpointStrategy::BestAndPeriodic(n) => {
                current_loss < self.best_loss || (*n > 0 && epoch % n == 0)
            }
        }
    }

    /// 保存检查点
    ///
    /// # 参数
    /// - `llm`: LLM模型
    /// - `metadata`: 检查点元数据
    pub fn save_checkpoint(
        &mut self,
        llm: &LLM,
        metadata: CheckpointMetadata,
    ) -> Result<PathBuf, String> {
        let is_best = metadata.loss < self.best_loss;

        if is_best {
            self.best_loss = metadata.loss;
            self.best_epoch = metadata.epoch;
        }

        // 构建需要写入的检查点列表。
        //
        // 教学说明（重要）：
        // - 旧实现的 BestAndLast/BestAndPeriodic 采用 “二选一” 策略：如果本轮是 best，就不写 last；
        // - 这会让 `checkpoint_last.bin` 不是“最新状态”，从而导致恢复训练/测试出现不一致甚至 flaky；
        // - 正确语义应当是：组合策略要同时写入多个目标（例如 best + last）。
        let mut checkpoint_names: Vec<String> = Vec::new();
        match &self.strategy {
            CheckpointStrategy::Best => {
                if is_best {
                    checkpoint_names.push(format!(
                        "checkpoint_best_epoch_{}_loss_{:.4}.bin",
                        metadata.epoch, metadata.loss
                    ));
                } else {
                    return Ok(PathBuf::new());
                }
            }
            CheckpointStrategy::Last => {
                checkpoint_names.push("checkpoint_last.bin".to_string());
            }
            CheckpointStrategy::Periodic(n) => {
                if *n > 0 && metadata.epoch % n == 0 {
                    checkpoint_names.push(format!("checkpoint_epoch_{}.bin", metadata.epoch));
                } else {
                    return Ok(PathBuf::new());
                }
            }
            CheckpointStrategy::BestAndLast => {
                // 永远写 last，若本轮是 best 也额外写 best。
                checkpoint_names.push("checkpoint_last.bin".to_string());
                if is_best {
                    checkpoint_names.push(format!(
                        "checkpoint_best_epoch_{}_loss_{:.4}.bin",
                        metadata.epoch, metadata.loss
                    ));
                }
            }
            CheckpointStrategy::BestAndPeriodic(n) => {
                if is_best {
                    checkpoint_names.push(format!(
                        "checkpoint_best_epoch_{}_loss_{:.4}.bin",
                        metadata.epoch, metadata.loss
                    ));
                }
                if *n > 0 && metadata.epoch % n == 0 {
                    checkpoint_names.push(format!("checkpoint_epoch_{}.bin", metadata.epoch));
                }
                if checkpoint_names.is_empty() {
                    return Ok(PathBuf::new());
                }
            }
        }

        // 序列化模型层
        let mut serializable_layers = Vec::new();
        for layer in llm.network.iter() {
            match crate::model_serialization::SerializableLayer::from_layer(layer) {
                Ok(s_layer) => {
                    serializable_layers.push(s_layer);
                }
                Err(e) => {
                    return Err(format!("序列化层失败: {}", e));
                }
            }
        }

        let serializable_model = SerializableModel {
            version: 1,
            vocab: llm.vocab.clone(),
            layers: serializable_layers,
            context_window: llm.context_window.clone(),
            metadata: crate::model_serialization::ModelMetadata {
                embedding_dim: crate::EMBEDDING_DIM,
                hidden_dim: crate::HIDDEN_DIM,
                num_transformer_blocks: llm
                    .network
                    .iter()
                    .filter(|layer| layer.as_any().is::<TransformerBlock>())
                    .count(),
                vocab_size: llm.vocab.len(),
                max_seq_len: llm.max_context_length,
                training_info: None,
            },
        };

        let checkpoint = Checkpoint {
            model: serializable_model,
            metadata: metadata.clone(),
        };

        // 保存为二进制格式（原子写入：tmp -> rename），并同时写入 JSON 元数据。
        //
        // 说明：
        // - 避免进程中断/并发覆盖导致生成截断文件（常见症状：UnexpectedEof）
        // - rename 在类 Unix 上原子；Windows 上若目标已存在则先删除
        let mut saved_last_this_call = false;
        let mut saved_paths: Vec<PathBuf> = Vec::new();
        for checkpoint_name in checkpoint_names {
            if checkpoint_name == "checkpoint_last.bin" {
                saved_last_this_call = true;
            }
            let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);
            let tmp_checkpoint_path = checkpoint_path.with_extension("bin.tmp");
            let file = fs::File::create(&tmp_checkpoint_path)
                .map_err(|e| format!("创建检查点临时文件失败: {}", e))?;
            let mut writer = std::io::BufWriter::new(file);

            bincode::encode_into_std_write(&checkpoint, &mut writer, bincode::config::standard())
                .map_err(|e| format!("序列化检查点失败: {}", e))?;
            writer
                .flush()
                .map_err(|e| format!("写入检查点失败(Flush): {}", e))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| format!("写入检查点失败(Sync): {}", e))?;
            drop(writer);

            if checkpoint_path.exists() {
                fs::remove_file(&checkpoint_path)
                    .map_err(|e| format!("覆盖旧检查点失败: {}", e))?;
            }
            fs::rename(&tmp_checkpoint_path, &checkpoint_path)
                .map_err(|e| format!("提交检查点文件失败(Rename): {}", e))?;

            // 同时保存JSON格式的元数据（方便查看，同样使用原子写入）
            let metadata_path = checkpoint_path.with_extension("json");
            let tmp_metadata_path = metadata_path.with_extension("json.tmp");
            let file = fs::File::create(&tmp_metadata_path)
                .map_err(|e| format!("创建元数据临时文件失败: {}", e))?;
            let mut writer = std::io::BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &metadata)
                .map_err(|e| format!("序列化元数据失败: {}", e))?;
            writer
                .flush()
                .map_err(|e| format!("写入元数据失败(Flush): {}", e))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| format!("写入元数据失败(Sync): {}", e))?;
            drop(writer);

            if metadata_path.exists() {
                fs::remove_file(&metadata_path).map_err(|e| format!("覆盖旧元数据失败: {}", e))?;
            }
            fs::rename(&tmp_metadata_path, &metadata_path)
                .map_err(|e| format!("提交元数据文件失败(Rename): {}", e))?;

            log::info!(
                "📦 检查点已保存: {} (epoch={}, loss={:.4}{})",
                checkpoint_path.display(),
                metadata.epoch,
                metadata.loss,
                if is_best { ", 🏆 NEW BEST!" } else { "" }
            );

            saved_paths.push(checkpoint_path);
        }

        // 清理旧的检查点
        if is_best {
            self.cleanup_old_checkpoints()?;
        }

        // 返回“最可能被调用方使用”的路径：
        // - 如果保存了 last，优先返回 last；
        // - 否则返回第一个写入的路径。
        let last_path = self.checkpoint_dir.join("checkpoint_last.bin");
        // 注意：不能用 `last_path.exists()` 判断，因为目录里可能有历史遗留的 last 文件，
        // 但本轮策略未必写入 last；这种情况下返回旧 last 会误导调用方。
        if saved_last_this_call {
            Ok(last_path)
        } else {
            Ok(saved_paths.into_iter().next().unwrap_or_default())
        }
    }

    /// 加载检查点
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<(LLM, CheckpointMetadata), String> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(format!("检查点文件不存在: {}", path.display()));
        }

        log::info!("📂 正在加载检查点: {}", path.display());

        let file = fs::File::open(path).map_err(|e| format!("打开检查点文件失败: {}", e))?;
        let mut reader = std::io::BufReader::new(file).take(BINCODE_DECODE_LIMIT_BYTES as u64);

        let config = bincode::config::standard().with_limit::<BINCODE_DECODE_LIMIT_BYTES>();
        let checkpoint: Checkpoint = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| format!("反序列化检查点失败: {}", e))?;

        // 重建模型
        let mut network: Vec<Box<dyn crate::llm::Layer>> = Vec::new();
        for s_layer in checkpoint.model.layers.iter() {
            let layer = s_layer.to_layer(checkpoint.model.vocab.len());
            network.push(layer);
        }

        let llm = LLM {
            vocab: checkpoint.model.vocab,
            network,
            context_window: checkpoint.model.context_window,
            max_context_length: checkpoint.model.metadata.max_seq_len,
            training: false,
            parallel_training: false,
            sampling_prob_buffer: Vec::new(),
            sampling_idx_buffer: Vec::new(),
            beam_candidates_buffer: Vec::new(),
        };

        log::info!(
            "✅ 检查点加载成功: epoch={}, loss={:.4}, phase={}",
            checkpoint.metadata.epoch,
            checkpoint.metadata.loss,
            checkpoint.metadata.phase
        );

        Ok((llm, checkpoint.metadata))
    }

    /// 获取当前“最佳 checkpoint”对应的文件路径。
    ///
    /// 说明：
    /// - 这里依赖管理器内部记录的 `best_loss` 判断当前是否存在 best checkpoint；
    /// - 真正选中的文件来自磁盘扫描结果，并按文件修改时间取最新一个。
    pub fn get_best_checkpoint(&self) -> Option<PathBuf> {
        if self.best_loss == f32::INFINITY {
            return None;
        }

        // 查找所有best检查点文件
        let mut best_checkpoints: Vec<_> = fs::read_dir(&self.checkpoint_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("checkpoint_best") && name_str.ends_with(".bin")
            })
            .collect();

        // 按修改时间排序，最新的在前
        best_checkpoints.sort_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok());
        best_checkpoints.reverse();

        best_checkpoints.first().map(|entry| entry.path())
    }

    /// 获取 `checkpoint_last.bin` 的路径。
    ///
    /// 这个接口只反映“最近一次保存的 last checkpoint 是否存在”，
    /// 不负责判断它是否也是最佳模型。
    pub fn get_last_checkpoint(&self) -> Option<PathBuf> {
        let last_path = self.checkpoint_dir.join("checkpoint_last.bin");
        if last_path.exists() {
            Some(last_path)
        } else {
            None
        }
    }

    /// 列出所有检查点
    pub fn list_checkpoints(&self) -> Result<Vec<(PathBuf, CheckpointMetadata)>, String> {
        let mut checkpoints = Vec::new();

        let entries =
            fs::read_dir(&self.checkpoint_dir).map_err(|e| format!("读取检查点目录失败: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("读取目录项失败: {}", e))?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                // 读取对应的JSON元数据
                let metadata_path = path.with_extension("json");
                if metadata_path.exists() {
                    let metadata_json = fs::read_to_string(&metadata_path)
                        .map_err(|e| format!("读取元数据失败: {}", e))?;
                    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
                        .map_err(|e| format!("解析元数据失败: {}", e))?;
                    checkpoints.push((path, metadata));
                }
            }
        }

        // 按epoch排序
        checkpoints.sort_by_key(|(_, metadata)| metadata.epoch);

        Ok(checkpoints)
    }

    /// 清理多余的 best checkpoint，仅保留 loss 最优的前 `keep_best_n` 个。
    ///
    /// 教学说明：这里按 metadata 中的 `loss` 排序，而不是按文件时间排序，
    /// 因为“最新保存”不一定等于“损失最优”。
    fn cleanup_old_checkpoints(&self) -> Result<(), String> {
        let mut best_checkpoints: Vec<_> = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| format!("读取检查点目录失败: {}", e))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("checkpoint_best") && name_str.ends_with(".bin")
            })
            .filter_map(|entry| {
                let path = entry.path();
                let metadata_path = path.with_extension("json");
                if metadata_path.exists() {
                    let metadata_json = fs::read_to_string(&metadata_path).ok()?;
                    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json).ok()?;
                    Some((path, metadata))
                } else {
                    None
                }
            })
            .collect();

        // 按loss排序（最好的在前）
        best_checkpoints.sort_by(|(_, a), (_, b)| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 删除多余的检查点
        for (path, metadata) in best_checkpoints.iter().skip(self.keep_best_n) {
            log::info!(
                "🗑️  删除旧检查点: {} (epoch={}, loss={:.4})",
                path.display(),
                metadata.epoch,
                metadata.loss
            );
            fs::remove_file(path).map_err(|e| format!("删除检查点失败: {}", e))?;

            // 同时删除元数据文件
            let metadata_path = path.with_extension("json");
            if metadata_path.exists() {
                fs::remove_file(metadata_path).ok();
            }
        }

        Ok(())
    }

    /// 更新最佳loss（用于EarlyStopping集成）
    pub fn update_best_loss(&mut self, loss: f32, epoch: usize) {
        if loss < self.best_loss {
            self.best_loss = loss;
            self.best_epoch = epoch;
        }
    }

    /// 获取最佳loss
    pub fn get_best_loss(&self) -> f32 {
        self.best_loss
    }

    /// 获取最佳epoch
    pub fn get_best_epoch(&self) -> usize {
        self.best_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_strategy_should_save() {
        let manager =
            CheckpointManager::new("test_checkpoints", CheckpointStrategy::Periodic(10), 3)
                .unwrap();

        assert!(manager.should_save(10, 1.0));
        assert!(!manager.should_save(11, 1.0));
        assert!(manager.should_save(20, 1.0));
    }

    #[test]
    fn test_best_checkpoint_update() {
        let mut manager =
            CheckpointManager::new("test_checkpoints", CheckpointStrategy::Best, 3).unwrap();

        manager.update_best_loss(2.0, 10);
        assert_eq!(manager.get_best_loss(), 2.0);
        assert_eq!(manager.get_best_epoch(), 10);

        manager.update_best_loss(1.5, 20);
        assert_eq!(manager.get_best_loss(), 1.5);
        assert_eq!(manager.get_best_epoch(), 20);

        // 不应该更新（loss更高）
        manager.update_best_loss(2.5, 30);
        assert_eq!(manager.get_best_loss(), 1.5);
        assert_eq!(manager.get_best_epoch(), 20);
    }
}
