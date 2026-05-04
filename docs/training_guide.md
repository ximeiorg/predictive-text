# 训练配置指南

## 1. 数据准备

### 推荐：词表剪枝流程（基于 label.txt）

```bash
uv run python scripts/analyze_and_prune_vocab.py
```

流程：
1. 训练 BPE（12000）
2. 采样 30% 语料统计 token 频次
3. 用 `label.txt` 过滤 + 频率排序 → 剪枝到 5000 词表
4. 输出 `data/vocab.json`（SimpleTokenizer 格式）、`data/train.bin`、`data/val.bin`

可选参数：
```bash
# 只看统计不 tokenize
uv run python scripts/analyze_and_prune_vocab.py --dry-run

# 自定义目标词表大小
uv run python scripts/analyze_and_prune_vocab.py --target-vocab-size 4000
```

### 备选：直接训练 BPE

```bash
uv run src/train.py \
    --data-path data/cleaned/all_cleaned.txt \
    --model-size tiny
```

或使用 `prepare_from_cleaned.py`：
```bash
uv run python scripts/prepare_from_cleaned.py --char-vocab label.txt
```

## 2. 模型配置 (src/config.py - ModelConfig)

```python
@dataclass
class ModelConfig:
    vocab_size: int = 5000       # 词表大小（必须与 vocab.json 匹配）
    hidden_dim: int = 256        # 隐藏层维度
    num_heads: int = 4           # 注意力头数
    num_layers: int = 4          # Transformer层数
    ffn_dim: int = 512           # FFN中间层维度
    max_seq_len: int = 32        # 最大序列长度
    dropout: float = 0.1
```

**参数量计算**（vocab_size=5000）:
- hidden_dim=128, num_layers=2: ~1.5M（tiny）
- hidden_dim=320, num_layers=6: ~5M（small）
- hidden_dim=384, num_layers=6: ~9M（medium）
- hidden_dim=512, num_layers=6: ~16M（base）

## 3. 训练配置 (TrainingConfig)

```python
@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
```

## 4. 训练命令示例

```bash
# 使用已准备好的数据训练
uv run src/train.py --model-size small --use-prepared-data

# 自定义参数
uv run src/train.py --model-size small --use-prepared-data \
    --epochs 20 \
    --batch-size 64 \
    --lr 5e-4 \
    --max-seq-len 64

# 从头训练（新数据）
uv run src/train.py \
    --data-path data/cache/corpus.txt \
    --model-size tiny \
    --char-vocab label.txt
```

## 5. 标点 Loss 掩码

`WikipediaDataset` 在 `__getitem__` 中会自动检测词表中的句中停顿标点
（`，、；：`），将其 label 替换为 `pad_id`（ID=0）。

`cross_entropy(ignore_index=0)` 会忽略这些位置，模型不会学习输出逗号/顿号。
句末标点 `。！？` 保留在 loss 中，模型可正常学习。

无需手动配置，数据集按 `id2token` 自动识别。

## 6. 性能指标参考

| 模型 | 词表 | hidden_dim | layers | 参数量 | 大小 |
|-----|------|-----------|--------|-------|------|
| tiny | 5000 | 128       | 2      | ~1.5M | ~6MB |
| small | 5000 | 320      | 6      | ~5M   | ~20MB |
| medium | 5000 | 384     | 6      | ~9M   | ~36MB |
| base | 5000 | 512       | 6      | ~16M  | ~64MB |
| large | 5000 | 768      | 8      | ~35M  | ~140MB |

## 7. 常见问题

**Q: UNK 率 0.77% 是否正常？**
A: 正常。UNK 主要来自语料中的英文/URL，输入法使用场景下中文输入几乎无 UNK。

**Q: 模型总预测逗号怎么办？**
A: 数据集会自动掩码逗号/顿号的 loss。如果预测结果仍有逗号，说明模型在生成阶段选择
了逗号——可以在 `inference_candidate.py` 中加候选过滤，排除逗号顿号。

**Q: 词表太大导致模型不收敛？**
A: 使用 `analyze_and_prune_vocab.py --target-vocab-size 4000` 剪枝到更小词表。

**Q: 训练速度慢？**
A: 减小 batch_size 或 hidden_dim。词表 5000 比 14939 的 embedding 矩阵小 3 倍。

**Q: 如何添加新语料？**
A: 将新文本追加到 `data/cache/corpus.txt`，然后重新运行
`analyze_and_prune_vocab.py`（会自动加载已有 BPE 缓存，仅需 ~10 分钟）。