# 模型评估指南

## 快速开始

### 评估单个模型

```bash
# 评估 base 模型 (默认)
uv run src/evaluate.py --model-size base

# 评估 small 模型
uv run src/evaluate.py --model-size small

# 指定参数
uv run src/evaluate.py \
    --model-size base \
    --max-samples 5000 \
    --batch-size 128
```

### 对比多个模型

```bash
# 对比 tiny, small, medium, base 模型
uv run scripts/compare_models.py

# 指定要对比的模型
uv run scripts/compare_models.py --model-sizes tiny small base
```

---

## 评估指标说明

### 1. 困惑度 (Perplexity)

**定义**: 衡量模型预测下一个词的不确定性

**计算公式**: `PPL = exp(average_loss)`

**解读**:
- 越低越好
- PPL = 1: 完美预测
- PPL < 10: 优秀
- PPL < 20: 良好
- PPL < 50: 可接受
- PPL > 100: 需要改进

**影响因素**:
- 词表大小 (词表越大，PPL 越高)
- 模型尺寸
- 训练数据质量
- 训练轮数

### 2. Top-K 准确率

**定义**: 正确词在模型预测的前 K 个词中的比例

**常用 K 值**:
- **Top-1**: 最严格，正确词必须是第一个预测
- **Top-3**: 实际应用中最常用
- **Top-5**: 输入法联想常用
- **Top-10**: 较宽松

**解读**:
| Top-1 | Top-5 | 等级 |
|-------|-------|------|
| > 40% | > 60% | 优秀 |
| 30-40% | 50-60% | 良好 |
| 20-30% | 40-50% | 中等 |
| < 20% | < 40% | 需改进 |

### 3. 平均排名

**定义**: 正确词在所有词表中的排名

**解读**:
- 越低越好
- 平均排名 < 5: 优秀
- 平均排名 < 20: 良好
- 平均排名 < 50: 可接受
- 平均排名 > 100: 需要改进

### 4. 综合评分

**计算方式** (满分 100):
```
困惑度得分: 0-40 分
Top-1 准确率: 0-30 分
Top-5 准确率: 0-20 分
Top-10 准确率: 0-10 分
```

**等级划分**:
- 80-100 分: 优秀 ⭐⭐⭐⭐⭐
- 60-79 分: 良好 ⭐⭐⭐⭐
- 40-59 分: 中等 ⭐⭐⭐
- 20-39 分: 一般 ⭐⭐
- 0-19 分: 需改进 ⭐

---

## 评估输出示例

```
============================================================
评估摘要
============================================================
困惑度:       45.92
Top-1 准确率:  33.10%
Top-5 准确率:  52.00%
平均排名:     106.96
综合评分:     36.4/100 (一般 ⭐⭐)
============================================================
```

---

## 改进建议

### 如果困惑度高 (> 50)

**可能原因**:
1. 训练不充分
   ```bash
   # 增加训练轮数
   uv run src/train.py --model-size base --epochs 20 --use-prepared-data
   ```

2. 学习率不合适
   ```bash
   # 尝试不同学习率
   uv run src/train.py --model-size base --lr 1e-4 --use-prepared-data
   ```

3. 模型容量不足
   ```bash
   # 使用更大的模型
   uv run src/train.py --model-size large --use-prepared-data
   ```

4. 数据质量问题
   - 检查训练数据是否有噪声
   - 增加训练数据量

### 如果准确率低

**可能原因**:
1. 词表太大
   ```bash
   # 减小词表大小
   # 重新准备数据
   ```

2. 训练数据不足
   - 增加训练数据量
   - 数据增强

3. 模型容量不足
   - 使用更大的模型

### 如果预测不合理

**检查**:
1. 分词是否正确
2. 词表是否合理
3. 训练数据是否代表性

---

## 评估工作流

### 训练后立即评估

```bash
# 训练 + 自动评估
uv run src/train.py --model-size small --use-prepared-data --eval-after-train
```

### 完整评估流程

```bash
# 1. 训练模型
uv run src/train.py --model-size small --use-prepared-data

# 2. 评估模型
uv run src/evaluate.py --model-size small

# 3. 对比不同模型
uv run scripts/compare_models.py

# 4. 查看评估报告
cat output/small/evaluation_report.json
```

---

## 评估报告

评估完成后会生成 JSON 格式的报告:

```json
{
  "model_size": "base",
  "perplexity": 45.92,
  "avg_loss": 3.82,
  "accuracy": {
    "top_1": 0.331,
    "top_3": 0.452,
    "top_5": 0.520,
    "top_10": 0.598
  },
  "detailed_metrics": {
    "mean_rank": 106.96,
    "median_rank": 4.0,
    "mean_prob": 0.249
  },
  "score": 36.4,
  "grade": "一般 ⭐⭐"
}
```

### 保存位置

```
output/
├── base/
│   ├── best_model.pt
│   └── evaluation_report.json
├── small/
│   ├── best_model.pt
│   └── evaluation_report.json
└── model_comparison.json
```

---

## 性能基准

### 不同模型尺寸的预期性能

| 模型 | 参数量 | 困惑度 | Top-1 | Top-5 | 评分 |
|-----|-------|--------|-------|-------|------|
| tiny | 2M | 60-80 | 20-30% | 35-45% | 25-35 |
| small | 6M | 50-60 | 25-35% | 40-50% | 30-40 |
| medium | 12M | 40-50 | 30-40% | 45-55% | 35-45 |
| base | 20M | 35-45 | 35-45% | 50-60% | 40-50 |
| large | 40M | 30-40 | 40-50% | 55-65% | 45-55 |

### 参考基准 (训练良好)

**小型模型 (tiny/small)**:
- 困惑度: < 60
- Top-1: > 25%
- 评分: > 30

**中型模型 (medium)**:
- 困惑度: < 50
- Top-1: > 30%
- 评分: > 35

**大型模型 (base/large)**:
- 困惑度: < 40
- Top-1: > 35%
- 评分: > 40

---

## 常见问题

### Q1: 评估结果不稳定怎么办?

A: 增加评估样本数:
```bash
uv run src/evaluate.py --model-size base --max-samples 10000
```

### Q2: 如何评估量化后的模型?

A: 目前评估脚本评估的是 PyTorch 模型。量化后的 MNN 模型需要集成到应用中测试。

### Q3: 训练多个 epoch 后性能反而下降?

A: 可能过拟合，尝试:
- 减少训练轮数
- 增加数据量
- 使用正则化 (dropout)

### Q4: 如何加速评估?

A: 
```bash
# 使用更大的 batch size
uv run src/evaluate.py --batch-size 256

# 减少样本数
uv run src/evaluate.py --max-samples 1000
```

---

## 总结

| 指标 | 优秀 | 良好 | 中等 | 需改进 |
|-----|------|------|------|--------|
| 困惑度 | < 10 | 10-20 | 20-50 | > 50 |
| Top-1 | > 40% | 30-40% | 20-30% | < 20% |
| Top-5 | > 60% | 50-60% | 40-50% | < 40% |
| 评分 | > 80 | 60-80 | 40-60 | < 40 |

**建议**: 定期评估模型性能，对比不同配置，选择最适合的模型尺寸和训练参数。