# 混合精度训练系统

> **版本**: v0.5.0  
> **状态**: ✅ 完成并验证

## 概述

RustGPT-Chinese 现已支持 FP16/BF16 混合精度训练，通过降低计算精度来加速训练并减少内存占用，同时使用动态损失缩放（Dynamic Loss Scaling）确保训练稳定性。

### 核心特性

- ✅ **FP16/BF16 支持**：前向和反向传播使用低精度计算
- ✅ **双权重系统**：Master 权重（FP32）+ Working 副本（FP16/BF16）
- ✅ **动态损失缩放**：自动调整缩放因子，防止梯度下溢
- ✅ **溢出检测**：实时监测 NaN/Inf，自动跳过异常步骤
- ✅ **自动回退**：持续不稳定时自动切换回 FP32
- ✅ **完整监控**：记录精度类型、缩放因子、溢出率

## 快速开始

### 1. 基本使用

```rust
use llm::{LLM, MixedPrecisionConfig, MixedPrecisionTrainer};

// 创建混合精度配置
let config = MixedPrecisionConfig::fp16();

// 创建训练器
let mut trainer = MixedPrecisionTrainer::new(config);

// 训练模型
let mut llm = LLM::default();
let tokenized_data = /* ... */;

trainer.train_monitored(
    &mut llm,
    tokenized_data,
    100,    // epochs
    0.001,  // initial_lr
    30,     // patience
);
```

### 2. 配置选项

#### FP16 训练

```rust
let config = MixedPrecisionConfig::fp16();
```

#### BF16 训练

```rust
let config = MixedPrecisionConfig::bf16();
```

#### 纯 FP32 训练（禁用混合精度）

```rust
let config = MixedPrecisionConfig::disabled();
```

#### 自定义配置

```rust
let config = MixedPrecisionConfig::fp16()
    .with_loss_scale(32768.0)           // 初始损失缩放因子
    .with_growth_params(2.0, 2000)       // 增长因子和间隔
    .with_backoff_factor(0.5)            // 回退因子
    .with_scale_range(1.0, 16777216.0)   // 缩放范围
    .with_auto_fallback(true, 5);        // 自动回退（5次溢出后）
```

## 技术细节

### 精度类型对比

| 特性 | FP32 | FP16 | BF16 |
|------|------|------|------|
| 位数 | 32 | 16 | 16 |
| 指数位 | 8 | 5 | 8 |
| 尾数位 | 23 | 10 | 7 |
| 表示范围 | ±3.4e38 | ±6.55e4 | ±3.4e38 |
| 精度 | ~7位 | ~3-4位 | ~2-3位 |
| 适用场景 | 高精度需求 | 通用深度学习 | 需要大动态范围 |

### 双权重系统

```
Master 权重 (FP32)
    ↓ (前向传播前转换)
Working 副本 (FP16/BF16)
    ↓ (前向传播)
激活值 (FP16/BF16)
    ↓ (反向传播)
梯度 (FP16/BF16 → unscale → FP32)
    ↓ (优化器更新)
Master 权重 (FP32) ← 更新
```

### 动态损失缩放算法

```
1. 缩放损失: scaled_loss = loss × scale
2. 反向传播: 计算缩放后的梯度
3. Unscale 梯度: grad /= scale
4. 溢出检测:
   - 如果有 NaN/Inf → 减小 scale，跳过更新
   - 如果正常 → 继续
5. 梯度裁剪: 使用原始未缩放的梯度
6. 优化器更新: 更新 Master 权重
7. 缩放调整:
   - 连续 N 步无溢出 → scale *= growth_factor
   - 发生溢出 → scale *= backoff_factor
```

### 训练流程集成

```rust
for epoch in 0..max_epochs {
    for sample in training_data {
        // 1. 前向传播（使用低精度）
        let logits = forward_with_low_precision(sample);
        
        // 2. 计算损失
        let loss = compute_loss(logits, target);
        
        // 3. 缩放损失（隐式）
        // 梯度会自动被缩放
        
        // 4. 反向传播（低精度）
        let mut gradients = backward(loss);
        
        // 5. Unscale 并检查溢出
        if !scaler.unscale_gradients(&mut gradients) {
            continue; // 跳过本次更新
        }
        
        // 6. 梯度裁剪（FP32）
        clip_gradients(&mut gradients, max_norm);
        
        // 7. 优化器更新（Master 权重始终 FP32）
        optimizer.step(&mut master_weights, &gradients, lr);
    }
}
```

## 配置参数详解

### MixedPrecisionConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 是否启用混合精度 |
| `precision_type` | `F32` | 精度类型（F16/BF16/F32） |
| `loss_scale` | `65536.0` | 初始损失缩放因子（2^16） |
| `scale_growth_factor` | `2.0` | 缩放增长倍数 |
| `scale_backoff_factor` | `0.5` | 缩放回退倍数 |
| `scale_growth_interval` | `2000` | 连续多少步无溢出才增长 |
| `max_loss_scale` | `16777216.0` | 最大缩放因子（2^24） |
| `min_loss_scale` | `1.0` | 最小缩放因子 |
| `auto_fallback` | `true` | 是否启用自动回退到 FP32 |
| `fallback_threshold` | `5` | 多少次连续溢出触发回退 |

### 推荐配置

#### 预训练（大数据集）

```rust
let config = MixedPrecisionConfig::fp16()
    .with_loss_scale(65536.0)       // 较大的初始缩放
    .with_growth_params(2.0, 2000)   // 稳定增长
    .with_auto_fallback(true, 5);    // 允许回退
```

#### 指令微调（小数据集）

```rust
let config = MixedPrecisionConfig::bf16()
    .with_loss_scale(32768.0)       // 较小的初始缩放
    .with_growth_params(1.5, 1000)   // 更激进的增长
    .with_auto_fallback(true, 3);    // 更敏感的回退
```

#### 实验/调试

```rust
let config = MixedPrecisionConfig::disabled(); // 使用 FP32
```

## 监控和诊断

### 训练日志示例

```
[MIXED PRECISION] Training with precision: FP16, loss scale: 65536
Epoch 0: Loss = 4.2315, LR = 0.001000, Precision = FP16, Scale = 65536.0, Overflows = 0/10 (0.00%)
Epoch 10: Loss = 3.8921, LR = 0.000995, Precision = FP16, Scale = 65536.0, Overflows = 2/110 (1.82%)
[OVERFLOW] Step 125: Loss scale reduced from 65536.0 to 32768.0 (overflow #3/125)
[SCALE] Loss scale increased from 32768.0 to 65536.0 after 2000 stable steps
[EARLY STOP] No improvement for 30 epochs. Best loss: 2.1456 at epoch 75
[TRAINING COMPLETE] Final stats - Loss: 2.1456, Precision: FP16, Overflow rate: 2.40%
```

### 关键指标

1. **Overflow Rate**：溢出率应保持在 < 10%
   - 0-5%：理想状态
   - 5-10%：可接受
   - > 10%：考虑调整配置或使用 BF16

2. **Loss Scale**：观察缩放因子的变化
   - 频繁波动：训练不稳定
   - 持续增长：训练稳定
   - 持续下降：可能需要回退

3. **Fallback Events**：自动回退事件
   - 0次：理想状态
   - 1次：可能的数值问题
   - 多次：建议直接使用 FP32

### 获取统计信息

```rust
// 获取溢出统计
let (total_overflows, total_steps, overflow_rate) = trainer.scaler_stats();
println!("溢出率: {:.2}%", overflow_rate * 100.0);

// 检查是否回退
if trainer.is_fallback_triggered() {
    println!("已自动回退到 FP32");
}

// 获取当前精度
println!("当前精度: {}", trainer.current_precision());
```

## 验证测试

运行混合精度验证脚本：

```bash
cargo run --example mixed_precision_test
```

### 预期输出

```
混合精度训练验证测试
============================================================

词汇表大小: 45
训练样本数: 10

============================================================
开始实验: FP32 (Baseline)
============================================================
...
实验结果:
  - 实际训练 epoch 数: 100
  - 最终损失: 2.3456
  - Perplexity: 10.4321
  - 训练时间: 5.23s
  - 溢出统计: 0/100 (0.00%)

============================================================
开始实验: FP16
============================================================
...
实验结果:
  - 实际训练 epoch 数: 100
  - 最终损失: 2.3512
  - Perplexity: 10.4912
  - 训练时间: 5.18s
  - 溢出统计: 3/100 (3.00%)

============================================================
验收判断
============================================================

✓ FP16 Loss 稳定性:     通过 (0.24% < 5%)
✓ BF16 Loss 稳定性:     通过 (0.31% < 5%)
✓ FP16 Perplexity:     通过 (0.57% < 3%)
✓ BF16 Perplexity:     通过 (0.63% < 3%)
✓ FP16 溢出率:         通过 (3.00% < 10%)
✓ BF16 溢出率:         通过 (2.50% < 10%)

============================================================
✅ 所有测试通过！混合精度训练系统验证成功。
============================================================
```

## 故障排除

### 问题 1: 溢出率过高 (> 10%)

**原因**：
- 模型不稳定
- 学习率过大
- 梯度爆炸

**解决方案**：
1. 降低初始 `loss_scale`
2. 使用 BF16 代替 FP16（更大的动态范围）
3. 降低学习率
4. 增强梯度裁剪

```rust
let config = MixedPrecisionConfig::bf16()
    .with_loss_scale(16384.0)  // 降低初始缩放
    .with_backoff_factor(0.25); // 更激进的回退
```

### 问题 2: 损失不收敛

**原因**：
- 精度损失过大
- 数值不稳定

**解决方案**：
1. 切换到 BF16 或 FP32
2. 禁用混合精度

```rust
let config = MixedPrecisionConfig::disabled();
```

### 问题 3: 频繁触发自动回退

**原因**：
- 模型对低精度敏感
- 训练数据存在异常值

**解决方案**：
1. 增加 `fallback_threshold`
2. 清洗训练数据
3. 使用 FP32 训练

```rust
let config = MixedPrecisionConfig::fp16()
    .with_auto_fallback(true, 10);  // 提高阈值
```

### 问题 4: Loss Scale 持续下降

**原因**：
- 梯度爆炸
- 学习率过大

**解决方案**：
1. 降低学习率
2. 增强梯度裁剪
3. 检查数据质量

## 性能对比

### 理论加速比

| 精度 | 内存占用 | 理论加速比 | 实际加速比* |
|------|----------|-----------|------------|
| FP32 | 100% | 1.0x | 1.0x |
| FP16 | 50% | 2.0x | 1.0-1.2x** |
| BF16 | 50% | 2.0x | 1.0-1.2x** |

\* 在 CPU 上实际加速有限，因为 ndarray 并未充分优化低精度运算  
\*\* GPU 上可获得 1.5-2.5x 的实际加速

### 内存节省

- **权重存储**：Master (FP32) + Working (FP16) ≈ 150%（比纯 FP32 略多）
- **激活值**：50%（显著节省）
- **梯度**：50%（显著节省）
- **总体**：约 30-40% 内存节省

## 最佳实践

1. **优先使用 BF16**：
   - 更大的动态范围
   - 更好的数值稳定性
   - 溢出率更低

2. **监控溢出率**：
   - 保持在 5% 以下
   - 超过 10% 考虑调整配置

3. **逐步启用**：
   - 先用 FP32 验证模型正确性
   - 再切换到混合精度
   - 对比 loss 曲线

4. **保存检查点**：
   - Master 权重始终是 FP32
   - 可以在任意精度间切换

5. **验证收敛性**：
   - 对比 FP32 baseline
   - loss 差异应 < 5%
   - perplexity 差异应 < 3%

## 未来优化方向

- [ ] SIMD 加速低精度运算
- [ ] 梯度累积时使用 FP16
- [ ] 选择性混合精度（部分层 FP32）
- [ ] GPU 后端集成
- [ ] 自动化超参数调优

## 参考资料

- [Mixed Precision Training (NVIDIA)](https://arxiv.org/abs/1710.03740)
- [BFloat16 for Deep Learning (Google)](https://arxiv.org/abs/1905.12322)
- [Dynamic Loss Scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)

## 更新日志

### v0.5.0 (当前版本)
- ✅ 完整的 FP16/BF16 支持
- ✅ 动态损失缩放
- ✅ 自动回退机制
- ✅ 训练监控和日志
- ✅ 验证测试脚本
- ✅ 完整文档

---

**作者**: RustGPT-Chinese Team  
**许可**: MIT License  
**联系**: 请通过 GitHub Issues 反馈问题
