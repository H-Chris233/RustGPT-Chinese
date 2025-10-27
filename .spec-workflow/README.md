# RustGPT-Chinese 项目规范文档

## 📋 文档概览

本目录包含 RustGPT-Chinese 项目的完整规范和开发指南文档。作为一个教育性质的深度学习项目，我们致力于保持代码简洁、注释充分、依赖最小化。

---

## 📚 核心文档

### 1. [SPEC_WORKFLOW.md](./SPEC_WORKFLOW.md) - 完整规范与工作流程 ⭐
**最重要的文档**，涵盖：
- 📋 项目概述和核心原则
- 🏗️ 架构规范（模块层次、Layer Trait、数据流）
- 📝 代码规范（命名、注释、错误处理、测试）
- 🔄 开发工作流程（分支策略、提交规范、代码审查）
- 📚 依赖管理原则（最小依赖哲学）
- 🧪 测试策略（测试金字塔、运行方法）
- 📖 文档维护规范
- 🚀 发布流程
- 🎯 贡献指南

**适用人群**: 所有开发者、贡献者、代码审查员

---

### 2. [TECH_STACK.md](./TECH_STACK.md) - 技术栈详细说明
深入介绍技术选型和架构：
- 🦀 Rust 2024 Edition 选择理由
- 📦 核心依赖详解（ndarray, jieba-rs, LRU 等）
- 🏗️ 分层神经网络架构
- 💾 数据存储方式（JSON、二进制、缓存）
- 🔧 开发工具链（Cargo, Rustfmt, Clippy）
- 🚀 部署和分发策略
- 📊 性能基准数据
- 🔒 安全和合规考虑
- 🤔 技术决策理由（为什么选择 Rust、ndarray、Pre-LN 等）

**适用人群**: 架构师、技术决策者、新加入的开发者

---

### 3. [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) - 项目结构文档
详细的代码组织和模块说明：
- 📁 完整的目录结构树
- 📄 核心文件详解（Cargo.toml, lib.rs, main.rs）
- 🧠 神经网络层详解（每个模块的职责和实现）
- 🏗️ 模型编排（LLM 类的前向/反向传播）
- 🎓 训练基础设施（Adam, CheckpointManager, BatchLoader）
- 📊 数据处理（Vocab, DatasetLoader）
- ⚡ 性能优化（FusedOps, KV-Cache）
- 🧪 测试文件组织
- 📚 文档层次结构
- 🔍 快速导航指南

**适用人群**: 想要理解代码库整体结构的开发者

---

### 4. [EDUCATIONAL_GUIDELINES.md](./EDUCATIONAL_GUIDELINES.md) - 教育性代码指南
教学优先的代码编写原则：
- 🎓 项目教育目标阐述
- 📝 注释原则（必须注释的内容、注释风格）
- 🏗️ 代码结构原则（单一职责、显式优于隐式）
- 🧮 数学实现规范（公式注释、梯度推导）
- 🧪 测试规范（测试驱动的教学、测试作为文档）
- 📚 依赖管理哲学（只添加无法简单实现的依赖）
- 🎨 代码美学（对齐、分隔、命名一致性）
- 🚫 反模式（应避免的做法）
- 📖 学习路径建议
- ✅ 代码审查 Checklist

**适用人群**: 想要贡献代码的开发者、代码审查员、学习者

---

## 🎯 快速导航

### 我想要...

| 目标 | 查看文档 | 章节 |
|------|---------|------|
| **了解项目整体架构** | SPEC_WORKFLOW.md | 架构规范 |
| **理解技术选型理由** | TECH_STACK.md | 技术决策和理由 |
| **找到某个功能的代码** | PROJECT_STRUCTURE.md | 文件快速导航 |
| **开始贡献代码** | EDUCATIONAL_GUIDELINES.md | 代码规范 + Checklist |
| **添加新的神经网络层** | SPEC_WORKFLOW.md | 开发工作流程 |
| **运行测试** | SPEC_WORKFLOW.md | 测试策略 |
| **理解依赖管理原则** | TECH_STACK.md | 依赖管理 + EDUCATIONAL_GUIDELINES.md |
| **学习如何注释代码** | EDUCATIONAL_GUIDELINES.md | 注释原则 |
| **查看发布流程** | SPEC_WORKFLOW.md | 发布流程 |
| **理解数据流** | TECH_STACK.md | 数据流向 + PROJECT_STRUCTURE.md |

---

## 🚀 快速开始

### 新开发者上手步骤

1. **阅读核心概念**（15-20 分钟）
   - 阅读 [SPEC_WORKFLOW.md](./SPEC_WORKFLOW.md) 的"项目概述"和"架构规范"
   - 理解 Layer Trait 和数据流

2. **理解代码结构**（20-30 分钟）
   - 阅读 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) 的目录结构和核心模块
   - 浏览 `src/` 目录，了解各文件职责

3. **学习代码规范**（15-20 分钟）
   - 阅读 [EDUCATIONAL_GUIDELINES.md](./EDUCATIONAL_GUIDELINES.md) 的注释和代码结构原则
   - 查看代码示例

4. **实践运行**（30 分钟）
   ```bash
   # 克隆项目
   git clone <repository_url>
   cd RustGPT-Chinese
   
   # 运行测试
   cargo test
   
   # 运行主程序（训练 + 推理）
   cargo run --release
   ```

5. **尝试小修改**（1-2 小时）
   - 修改 `lib.rs` 中的超参数（如 DROPOUT_RATE）
   - 观察训练效果变化
   - 参考 EDUCATIONAL_GUIDELINES.md 添加注释

---

## 📖 文档维护

### 更新文档的时机

| 变更类型 | 需要更新的文档 |
|---------|--------------|
| 添加新的神经网络层 | SPEC_WORKFLOW.md, PROJECT_STRUCTURE.md |
| 修改架构（如改变 Transformer 层数） | SPEC_WORKFLOW.md, TECH_STACK.md |
| 添加新依赖 | TECH_STACK.md, SPEC_WORKFLOW.md（依赖管理） |
| 修改训练流程 | TECH_STACK.md（数据流向） |
| 修改超参数 | PROJECT_STRUCTURE.md（配置规范） |
| 添加性能优化 | TECH_STACK.md（性能要求） |
| 修改代码规范 | EDUCATIONAL_GUIDELINES.md |

### 文档风格指南

- ✅ 使用 Markdown 格式
- ✅ 代码块标注语言（\`\`\`rust, \`\`\`bash）
- ✅ 使用 Emoji 增强可读性（📋 🏗️ 🧪 📚）
- ✅ 表格对齐和格式化
- ✅ 章节编号和层次清晰
- ✅ 提供示例代码和命令
- ✅ 中文为主，专业术语保留英文

---

## 🔗 相关文档

### 项目根目录文档
- **README.md / README_zh.md** - 用户快速开始指南
- **CLAUDE.md** - AI 辅助开发指南
- **IMPLEMENTATION_v0.4.0.md** - 当前版本实现笔记
- **PERFORMANCE_OPTIMIZATIONS.md** - 性能优化特性说明
- **BATCH_TRAINING.md** - 批量训练使用教程

### Spec Workflow 模板
- **templates/** - 默认模板（requirements, design, tasks 等）
- **user-templates/** - 自定义模板（可覆盖默认模板）

---

## 🤝 贡献指南

### 提交文档修改

1. **检查相关文档是否需要同步更新**
   - 例如：添加新层需要更新 SPEC_WORKFLOW.md 和 PROJECT_STRUCTURE.md

2. **遵循文档风格指南**
   - 保持格式一致
   - 使用清晰的标题和章节

3. **提交前检查**
   ```bash
   # 检查 Markdown 格式（可选）
   markdownlint *.md
   
   # 确保链接有效
   # 检查文档中的相对链接是否正确
   ```

4. **提交信息**
   ```bash
   git commit -m "docs(spec): 更新架构规范文档

   - 添加新增层的说明
   - 更新数据流图
   - 同步 PROJECT_STRUCTURE.md"
   ```

---

## 📞 反馈和建议

如果您发现文档有：
- ❓ 不清楚或难以理解的部分
- 🐛 错误或过时的信息
- 💡 改进建议

请：
1. 提交 GitHub Issue（标签：`documentation`）
2. 或直接提交 Pull Request 修改文档

---

## 📅 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v0.4.0 | 2024-10-25 | 创建完整的 Spec Workflow 文档集 |
|  |  | - SPEC_WORKFLOW.md（规范和工作流程） |
|  |  | - TECH_STACK.md（技术栈详解） |
|  |  | - PROJECT_STRUCTURE.md（项目结构） |
|  |  | - EDUCATIONAL_GUIDELINES.md（教育指南） |

---

## 📄 许可证

本文档集与 RustGPT-Chinese 项目使用相同的 MIT 许可证。详见 [LICENSE.txt](../LICENSE.txt)。

---

**文档维护者**: RustGPT-Chinese 项目组  
**最后更新**: 2024-10-25  
**对应项目版本**: v0.4.0

---

*感谢您对 RustGPT-Chinese 项目的关注！希望这些文档能帮助您更好地理解和贡献代码。* 🚀
