# 训练配置指南

## 1. 模型配置 (src/config.py - ModelConfig)

```python
@dataclass
class ModelConfig:
    vocab_size: int = 8192       # 词表大小（必须与 vocab.json 匹配）
    hidden_dim: int = 256        # 隐藏层维度（增大→更好效果，更多参数）
    num_heads: int = 4           # 注意力头数（hidden_dim必须能被num_heads整除）
    num_layers: int = 4          # Transformer层数（4-6层适合小模型）
    ffn_dim: int = 512           # FFN中间层维度（通常是hidden_dim的2倍）
    max_seq_len: int = 32        # 最大序列长度（输入法预测建议16-64）
    dropout: float = 0.1         # Dropout率（0.1-0.3）
```

**参数量计算**:
- vocab_size=8192, hidden_dim=256: ~6M参数
- vocab_size=8192, hidden_dim=512: ~12M参数
- vocab_size=8192, hidden_dim=512, num_layers=6: ~20M参数

## 2. 训练配置 (src/config.py - TrainingConfig)

```python
@dataclass
class TrainingConfig:
    batch_size: int = 128        # 批次大小（32-256，越大越稳定但越慢）
    learning_rate: float = 1e-3  # 学习率（1e-4 到 1e-3）
    num_epochs: int = 10         # 训练轮数（3-20轮）
    warmup_steps: int = 500      # 学习率预热步数
    max_grad_norm: float = 1.0   # 梯度裁剪
    label_smoothing: float = 0.1 # 标签平滑（0.0-0.2）
```

**调参建议**:
- Loss下降慢 → 增大 learning_rate 或 num_epochs
- Loss震荡 → 减小 learning_rate 或增大 batch_size
- 过拟合 → 减小 num_epochs 或增大 dropout

## 3. 数据配置 (src/config.py - DataConfig)

```python
@dataclass
class DataConfig:
    vocab_path: str = "data/vocab.json"
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    vocab_size: int = 8192       # 目标词表大小
    val_ratio: float = 0.05      # 验证集比例
```

## 4. 训练命令示例

```bash
# 使用默认配置训练
uv run src/train.py --use-prepared-data

# 自定义参数训练
uv run src/train.py --use-prepared-data \
    --epochs 20 \
    --batch-size 64 \
    --lr 5e-4 \
    --max-seq-len 64

# 从头训练（新数据）
uv run src/train.py \
    --data-path data/wiki_cleaned.txt \
    --vocab-size 8192 \
    --epochs 10
```

## 5. 推荐训练流程

### 小模型 (< 10MB，适合移动端)
```bash
# 1. 准备数据（8192词表）
uv run scripts/prepare_small_vocab.py --vocab-size 8192

# 2. 训练（6M参数）
uv run src/train.py --use-prepared-data --epochs 15 --batch-size 128

# 3. 导出
python -m src.export --checkpoint output/best_model.pt --format quantized
```

### 中等模型 (20-50MB，更好效果)
修改 `src/config.py`:
```python
hidden_dim: int = 512
num_layers: int = 6
```

然后:
```bash
uv run src/train.py --use-prepared-data --epochs 20 --batch-size 64
```

## 6. 性能指标参考

| 模型大小 | 词表 | hidden_dim | layers | Loss范围 | 适用场景 |
|---------|------|-----------|--------|---------|---------|
| 6M      | 8192 | 256       | 4      | 3-4     | 移动端 |
| 12M     | 8192 | 512       | 4      | 2.5-3.5 | PC端 |
| 20M     | 8192 | 512       | 6      | 2-3     | 高质量 |

## 7. 常见问题

**Q: Loss从6降到5就不降了？**
A: 词表太大(23k)导致参数分布不合理。重新训练8192词表。

**Q: 训练速度慢？**
A: 减小batch_size，或增大max_seq_len会导致变慢。

**Q: 内存不足？**
A: 减小batch_size或hidden_dim。

**Q: 如何监控训练？**
A: 查看进度条的loss值，loss应该稳定下降。