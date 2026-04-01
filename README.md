# 轻量级中文输入法预测模型

基于 Decoder-Only Transformer 的极致轻量级中文输入法预测模型，专为移动端设计。

## 设计目标

| 指标 | 目标值 |
|------|--------|
| 模型体积 | < 10 MB (INT8量化后) |
| 运行内存 | < 50 MB |
| 推理延迟 | < 30ms (单次预测) |
| 词表大小 | 5120 |
| 上下文长度 | 32 |

## 模型架构

```
Decoder-Only Transformer
├── Token Embedding: 5120 × 256
├── Positional Embedding: 32 × 256
├── Transformer Blocks × 4
│   ├── Multi-Head Attention (4 heads)
│   ├── Feed-Forward (256 → 512 → 256)
│   └── LayerNorm + Residual
├── LayerNorm
└── Output Projection: 256 → 5120

总参数量: ~8.2M
```

## 安装

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

## 快速开始

### 1. 准备数据

**方式一：使用配置文件 (推荐)**

编辑 `data_config.json`：

```json
{
  "sources": [
    {"path": "大王饶命.txt", "weight": 1.0},
    {"path": "other_novel.txt", "weight": 1.0}
  ],
  "max_samples": 100000000,
  "min_length": 5,
  "max_length": 512
}
```

准备数据：

```bash
uv run python -m src.prepare_data --config data_config.json
```

**方式二：命令行添加**

```bash
# 添加多个文件
uv run python -m src.prepare_data \
  --add-file novel1.txt \
  --add-file novel2.txt \
  --vocab-size 5120

# 添加整个目录
uv run python -m src.prepare_data \
  --add-dir data/corpus \
  --pattern "*.txt"

# 使用缓存增量添加
uv run python -m src.prepare_data \
  --use-cache \
  --add-file new_data.txt
```

### 2. 训练模型

```bash
# 使用配置文件训练
uv run python -m src.train --data-config data_config.json --epochs 3

# 使用已准备的数据训练
uv run python -m src.train --use-prepared-data --epochs 3

# 自定义参数
uv run python -m src.train \
  --use-prepared-data \
  --batch-size 32 \
  --epochs 3 \
  --lr 5e-4 \
  --device auto
```

### 3. 测试模型

```bash
# 使用默认测试文本
uv run python -m src.inference --checkpoint output/best_model.pt

# 交互模式
uv run python -m src.inference --checkpoint output/best_model.pt --interactive

# 自定义输入
uv run python -m src.inference --checkpoint output/best_model.pt --text "今天天气"
```

### 4. 导出模型

```bash
# 导出 ONNX 格式
python -m src.export --checkpoint output/best_model.pt --format onnx

# 导出量化模型
python -m src.export --checkpoint output/best_model.pt --format quantized

# 导出所有格式
python -m src.export --checkpoint output/best_model.pt --format both
```

## 数据加载器功能

### 多数据源支持

支持多种格式：
- **TXT**: 每行一段文本
- **JSON**: 数组或对象格式
- **CSV**: 自动合并所有列

### 数据清洗

自动过滤：
- 长度过短/过长的文本
- 中文占比低于30%的文本
- 纯数字/符号文本
- 广告和版权声明
- 特殊字符过多的文本

### 数据权重

不同数据源可设置不同权重：

```json
{
  "sources": [
    {"path": "novel.txt", "weight": 1.0},
    {"path": "news.txt", "weight": 2.0}  // 重复2次，增加权重
  ]
}
```

### 缓存机制

首次处理数据后会自动缓存：

```
data/cache/
├── data_cache.txt       # 处理后的文本
└── data_cache_stats.json # 统计信息
```

使用缓存：

```bash
uv run python -m src.prepare_data --use-cache
```

### 增量添加数据

```python
from src.data.loader import DataLoader, DataLoaderConfig

# 加载已有数据
loader = DataLoader(DataLoaderConfig())
loader.load_cache("data_cache")

# 添加新数据
loader.add_file("new_novel.txt")
texts = loader.load_all()

# 更新缓存
loader.save_cache("data_cache")
```

详细使用说明见 `docs/data_loader_guide.md`。

## 项目结构

```
wubi-lianxiang/
├── src/
│   ├── __init__.py
│   ├── config.py              # 配置文件
│   ├── train.py               # 训练脚本
│   ├── inference.py           # 推理脚本
│   ├── export.py              # 模型导出
│   ├── prepare_data.py        # 数据准备脚本
│   ├── build_vocab.py         # 词表构建
│   ├── model/
│   │   ├── __init__.py
│   │   └── transformer.py     # Transformer 模型
│   └── data/
│       ├── __init__.py
│       ├── dataset.py         # 数据集
│       └── loader.py          # 灵活数据加载器
├── data/                       # 训练数据
│   ├── cache/                 # 数据缓存
│   ├── vocab.model            # 词表
│   ├── train.bin              # 训练数据
│   └── val.bin                # 验证数据
├── output/                     # 模型输出
├── mobile/                     # 移动端模型
├── docs/                       # 文档
├── examples/                   # 示例脚本
├── data_config.json           # 数据配置
└── README.md
```

## 训练配置

```python
ModelConfig:
  vocab_size: 5120
  hidden_dim: 256
  num_heads: 4
  num_layers: 4
  ffn_dim: 512
  max_seq_len: 32
  dropout: 0.1

TrainingConfig:
  batch_size: 32
  learning_rate: 5e-4
  num_epochs: 3
  warmup_steps: 1000
  max_grad_norm: 1.0
  label_smoothing: 0.1
```

## 性能优化

### 量化
- INT8 动态量化可减小模型体积约 4 倍
- 推理速度提升约 2-3 倍

### ONNX 导出
- 支持 ONNX Runtime 部署
- 兼容移动端推理框架

### 移动端优化建议
1. 使用 CoreML (iOS) 或 NNAPI (Android)
2. 启用硬件加速
3. 使用更小的 batch size

## 训练数据格式

支持纯文本格式，每行一段文本：

```
这是一个例子。
我们今天去公园玩。
```

## 许可证

MIT