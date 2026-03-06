# Repository Guidelines

## Project Structure & Module Organization

- `src/`: 核心库与二进制（`lib.rs`, `main.rs`）；模块按单文件 `snake_case.rs` 组织（如 `self_attention.rs`）。
- `tests/`: 集成测试，文件命名 `*_test.rs`，支持 `cargo test --test <name>` 定向运行。
- `benches/`: 基准测试（见 `Cargo.toml` 的 `[[bench]]` 条目）。
- `examples/`: 可运行示例代码。
- `data/`: 训练语料（JSON 字符串数组），按用途分 `pretraining/` 与 `chat/`；传入目录路径时会自动加载目录下所有 `.json` 并合并。
- `docs/`: 设计与性能优化说明（中文）。

## Architecture Notes

- 核心链路通常从 `src/main.rs`（训练/CLI）进入，模型主逻辑在 `src/llm.rs`，Transformer 组件在 `src/transformer.rs` / `src/self_attention.rs`。
- 设计目标是“纯 Rust + `ndarray` 从零实现”；新增依赖请保持轻量，并在 PR 中说明必要性与替代方案。

## Build, Test, and Development Commands

```bash
cargo run                    # 运行训练/交互主流程（默认会读取 data/ 并产出 checkpoints/）
cargo build --release         # 发布构建（更快的推理/训练性能）
cargo build --features blas   # 启用 BLAS（需要系统安装 OpenBLAS）
cargo bench                  # 运行基准测试（可选，通常更耗时）
```

```bash
cargo test                   # 运行全部测试（CI 会执行 cargo build + cargo test）
cargo test --test llm_test    # 运行单个集成测试目标（tests/llm_test.rs）
cargo test -- --nocapture     # 显示测试输出（便于定位数值问题）
```

```bash
cargo fmt                    # 格式化（遵循 rustfmt.toml）
cargo clippy                 # 静态检查（建议在提交前运行）
```

## Coding Style & Naming Conventions

- 以 `cargo fmt` 作为唯一格式标准；本仓库 `rustfmt.toml` 采用 **4 空格缩进**，且不强制重排 `use` 导入顺序。
- 命名遵循 Rust 约定：类型/trait 用 `CamelCase`，函数/变量用 `snake_case`，模块/文件用 `snake_case.rs`。
- 复杂数学/算法优先用**中文注释**说明意图与推导；保持“从零实现”的教学取向，避免引入重量级 ML 依赖。

## Testing Guidelines

- 新增/修复功能必须补齐测试，优先覆盖：梯度/掩码（mask）、序列化与检查点恢复、中文分词与词表行为。
- 测试文件放在 `tests/`，命名 `*_test.rs`；修复回归时建议先添加失败用例再改实现。

## Data, Checkpoints & Artifacts

- 训练语料位于 `data/pretraining/` 与 `data/chat/`：保持为 **JSON 字符串数组**；新增分片建议沿用 `datasetN.json`/`setN.json` 的递增命名，避免破坏加载顺序。
- 不要提交运行产物：`target/`、`checkpoints/`、`exports/`、`logs/` 等目录已在 `.gitignore` 中忽略（PR 中出现这些文件通常意味着误提交）。

## Training API & Teaching Notes

- 训练公开入口应尽量保持最小表面积；当前推荐保留：`train(...)`、`train_monitored(...)`、`train_bucketed_sequential(...)`、`train_with_checkpointing(...)`。新增训练入口前先确认是否真的需要对外公开。
- 训练相关指标统一使用 **token-weighted mean loss**；不要在不同公开训练 API 中混用 sample-mean 与 token-mean。
- `train_bucketed_sequential(...)` 的语义是“按批次组织样本 + 批内逐样本顺序更新”，**不是**严格的 batch 梯度平均更新；后续文档与命名必须保持这一点。
- 教学主线优先围绕 `PreparedTrainingStep`、`prepare_training_step(...)`、`backward_with_ctx(...)` 组织，避免重新复制单样本训练逻辑。
- 删除无实际调用且教学价值不明确的训练变体 API，避免“功能开关式”入口持续膨胀。
- 注释必须描述当前真实行为，不要保留过时能力说明（如未启用的并行 tokenization）或夸张性能承诺。

## Commit & Pull Request Guidelines

- Git 历史同时存在命令式提交与 Conventional Commits；新提交建议统一使用下列格式（scope 可选）：

```text
feat(scope): 简要说明
fix: 修复 XXX
refactor: ...
docs: ...
test: ...
```

- PR 最小要求：说明动机与影响范围、关联 Issue（如有）、给出验证步骤（至少 `cargo test`），涉及语料请注明变更的 `data/...` 文件与数据格式约束（JSON 字符串数组）。
- CI 约束：`.github/workflows/rust.yml` 会在 Ubuntu 上运行 `cargo build` 与 `cargo test`；请确保本地通过 `cargo fmt && cargo clippy && cargo test` 再提 PR。
