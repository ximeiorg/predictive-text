# Small 模型优化指南

## 问题分析

### 当前状态
```
Loss: 4.6 (困惑度 ~100)
Epoch: 10
问题: Loss 难以下降
```

### 困惑度对比
| 模型 | 困惑度 | 评分 |
|-----|--------|------|
| **Small (当前)** | 100 | 22/100 |
| Base | 45 | 36/100 |
| 目标 | < 60 | > 30/100 |

### 可能原因

1. **学习率衰减过快**
   - Cosine decay 可能在 epoch 5-6 就降到很低
   - 解决：训练脚本已自动优化

2. **模型容量限制**
   - Hidden=256, Layers=4 是较小的配置
   - 解决：可调整模型配置

3. **优化策略问题**
   - 梯度裁剪、学习率等参数不适合小模型
   - 解决：训练脚本已自动优化

---

## 优化方案

### ✅ 方案1: 使用自动优化的训练参数 (推荐)

训练脚本已自动优化，直接使用即可：

```bash
# 训练 small 模型 - 自动使用优化参数
uv run src/train.py --model-size small --use-prepared-data
```

**自动应用的优化**:
- 学习率: **5e-4** (比原来 3e-4 更大)
- 训练轮数: **25** (比原来 10 更多)
- 预热步数: **300** (更长的预热)
- 梯度裁剪: **0.5** (比原来 1.0 更保守)
- 标签平滑: **0.1** (正则化)
- 最小学习率: **1e-6** (允许更低)

**预期效果**:
```
困惑度: 83 → 55-65
评分: 22 → 30-35
```

### 方案2: 自定义参数覆盖

如果需要进一步调整：

```bash
# 更大的学习率
uv run src/train.py \
    --model-size small \
    --lr 8e-4 \
    --epochs 30 \
    --use-prepared-data

# 更大的 batch size (梯度累积效果)
uv run src/train.py \
    --model-size small \
    --batch-size 512 \
    --epochs 25 \
    --use-prepared-data
```

### 方案3: 调整模型配置

修改 `src/config.py` 中的 small 配置:

```python
@classmethod
def small(cls) -> "ModelConfig":
    """Small 模型 - 优化版"""
    return cls(
        vocab_size=8000,
        hidden_dim=320,      # 256 → 320 (+25%)
        num_heads=4,
        num_layers=5,        # 4 → 5 (+1层)
        ffn_dim=640,         # 512 → 640 (+25%)
        max_seq_len=64,
        dropout=0.1,
    )
```

参数量: 4.2M → ~8M (仍在 small 范围)

然后训练:
```bash
uv run src/train.py --model-size small --use-prepared-data
```

### 方案4: 使用 Medium 模型

如果 small 模型效果不够:

```bash
uv run src/train.py --model-size medium --use-prepared-data
```

预期效果: 困惑度 ~50

---

## 各模型尺寸的自动优化参数

| 参数 | Tiny | Small | Medium | Base | Large |
|-----|------|-------|--------|------|-------|
| **学习率** | 6e-4 | 5e-4 | 4e-4 | 3e-4 | 2e-4 |
| **Epochs** | 20 | 25 | 18 | 15 | 15 |
| **Batch Size** | 512 | 384 | 320 | 256 | 192 |
| **梯度裁剪** | 0.5 | 0.5 | 0.8 | 1.0 | 1.0 |
| **标签平滑** | 0.1 | 0.1 | 0.05 | 0.0 | 0.05 |
| **最小学习率** | 1e-6 | 1e-6 | 1e-6 | 1e-5 | 1e-6 |

---

## 推荐训练流程

### 步骤1: 训练 small 模型

```bash
# 使用自动优化参数
uv run src/train.py --model-size small --use-prepared-data
```

### 步骤2: 监控训练

```bash
# 打开 TensorBoard
tensorboard --logdir output/small/logs

# 观察:
# - train_loss 和 val_loss 是否持续下降
# - 是否有过拟合迹象 (val_loss 开始上升)
```

### 步骤3: 评估结果

```bash
# 训练完成后评估
uv run src/evaluate.py --model-size small
```

### 步骤4: 如果效果不够好

**选项A**: 增加训练轮数
```bash
uv run src/train.py --model-size small --epochs 30 --use-prepared-data
```

**选项B**: 尝试更大的学习率
```bash
uv run src/train.py --model-size small --lr 8e-4 --use-prepared-data
```

**选项C**: 使用 medium 模型
```bash
uv run src/train.py --model-size medium --use-prepared-data
```

---

## 预期改进效果

### 目标指标

| 指标 | 当前 | 目标 | 优秀 |
|-----|------|------|------|
| 困惑度 | 83 | < 60 | < 40 |
| Top-1 | 26% | > 30% | > 35% |
| Top-5 | 46% | > 50% | > 55% |
| 评分 | 22 | > 30 | > 40 |

### 改进幅度预估

**使用自动优化参数**: 30-40% 改进
```
困惑度: 83 → 55-65
评分: 22 → 30-35
```

**增加模型容量**: 额外 10-15% 改进
```
困惑度: 55-65 → 48-55
评分: 30-35 → 35-40
```

---

## 常见问题

### Q1: 训练多久能看到效果?

**A**: 
- 使用优化参数后，5-8 个 epoch 就能看到明显改善
- 预计在 epoch 15-20 达到较好效果

### Q2: 如何判断是否过拟合?

**A**: 
- Train loss 下降，val loss 上升
- 观察 TensorBoard 中的曲线
- 训练脚本会自动保存最佳模型

### Q3: Loss 还是下降很慢怎么办?

**A**: 
1. 增大学习率: `--lr 8e-4`
2. 减小 weight_decay: 修改 `src/config.py` 中的 `weight_decay=0.001`
3. 增加模型容量
4. 使用 medium 模型

### Q4: 小模型能达到什么效果?

**A**: 
- Small (优化后): 困惑度 50-65
- Medium: 困惑度 45-55
- Base: 困惑度 35-45

---

## 快速测试命令

```bash
# 完整训练 (推荐)
uv run src/train.py --model-size small --use-prepared-data

# 训练后自动评估
uv run src/train.py --model-size small --use-prepared-data --eval-after-train

# 对比不同模型
uv run src/train.py --model-size tiny --use-prepared-data
uv run src/train.py --model-size small --use-prepared-data
uv run scripts/compare_models.py
```

---

## 总结

| 方案 | 难度 | 效果 | 推荐度 |
|-----|------|------|--------|
| 自动优化参数 | ⭐ | 好 | ⭐⭐⭐⭐⭐ 直接用 |
| 增加容量 | ⭐⭐ | 较好 | ⭐⭐⭐⭐ |
| 自定义参数 | ⭐⭐ | 好 | ⭐⭐⭐ |
| Medium 模型 | ⭐ | 很好 | ⭐⭐⭐⭐ |

**最佳实践**: 直接使用 `--model-size small`，训练脚本已自动应用优化参数！