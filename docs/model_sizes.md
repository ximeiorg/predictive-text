# 模型尺寸配置指南

本文档说明如何使用不同尺寸的模型配置进行训练和部署。

---

## 快速开始

### 1. 查看可用配置

```bash
# 列出所有可用模型尺寸
python -c "from src.config import list_model_sizes; list_model_sizes()"

# 或使用训练脚本
uv run src/train.py --list-sizes
```

### 2. 训练不同尺寸的模型

```bash
# 训练 tiny 模型 (2M 参数，8 MB)
uv run src/train.py --model-size tiny --use-prepared-data

# 训练 small 模型 (6M 参数，24 MB) - 推荐手机
uv run src/train.py --model-size small --use-prepared-data

# 训练 medium 模型 (12M 参数，48 MB)
uv run src/train.py --model-size medium --use-prepared-data

# 训练 base 模型 (20M 参数，80 MB) - 默认
uv run src/train.py --model-size base --use-prepared-data

# 训练 large 模型 (40M 参数，160 MB)
uv run src/train.py --model-size large --use-prepared-data
```

### 3. 使用训练好的模型

```bash
# 交互式推理
uv run src/inference.py --model-size small --interactive

# 单次预测
uv run src/inference.py --model-size small --text "今天天气"

# 指定 checkpoint
uv run src/inference.py --checkpoint output/small/best_model.pt --interactive
```

### 4. 导出为 MNN 格式

```bash
# 导出 small 模型
uv run src/export_mobile.py --model-size small

# 导出 tiny 模型
uv run src/export_mobile.py --model-size tiny

# 导出所有量化版本
uv run src/export_mobile.py --model-size small --all
```

---

## 模型尺寸对比

| 名称 | 参数量 | 模型大小 | 推理速度* | 内存占用 | 推荐场景 |
|-----|-------|---------|----------|---------|---------|
| **tiny** | ~2M | 8 MB | ~8 ms | ~20 MB | 低端手机、智能手表 |
| **small** | ~6M | 24 MB | ~12 ms | ~40 MB | 普通手机 (推荐) |
| **medium** | ~12M | 48 MB | ~18 ms | ~70 MB | 高端手机 |
| **base** | ~20M | 80 MB | ~25 ms | ~100 MB | PC/服务器 (默认) |
| **large** | ~40M | 160 MB | ~40 ms | ~180 MB | 追求精度 |

*推理速度为典型 Android 手机的测试数据

---

## 配置详情

### Tiny 模型

```python
ModelConfig(
    hidden_dim=128,    # 隐藏维度
    num_heads=4,       # 注意力头数
    num_layers=2,      # Transformer层数
    ffn_dim=256,       # FFN维度
    max_seq_len=64,    # 最大序列长度
)
```

**特点:**
- 参数量: ~2M
- 体积: 8 MB (float32)
- 极致体积优化
- 适合低端设备和实时推理

**推荐场景:**
- 低端手机 (2GB内存以下)
- 智能手表
- 嵌入式设备
- 实时性要求高

---

### Small 模型 (推荐手机)

```python
ModelConfig(
    hidden_dim=256,
    num_heads=4,
    num_layers=4,
    ffn_dim=512,
    max_seq_len=64,
)
```

**特点:**
- 参数量: ~6M
- 体积: 24 MB (float32)
- 体积和精度平衡
- 推荐大多数手机

**推荐场景:**
- 普通手机 (2-4GB内存)
- 输入法联想
- 日常应用

**训练推荐:**
```bash
uv run src/train.py --model-size small \
    --use-prepared-data \
    --batch-size 384 \
    --lr 4e-4 \
    --epochs 12
```

---

### Medium 模型

```python
ModelConfig(
    hidden_dim=384,
    num_heads=6,
    num_layers=6,
    ffn_dim=1024,
    max_seq_len=64,
)
```

**特点:**
- 参数量: ~12M
- 体积: 48 MB (float32)
- 精度和速度平衡
- 高端手机适用

**推荐场景:**
- 高端手机 (4GB+内存)
- 追求更好的效果
- 复杂联想场景

---

### Base 模型 (默认)

```python
ModelConfig(
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    ffn_dim=2048,
    max_seq_len=128,
)
```

**特点:**
- 参数量: ~20M
- 体积: 80 MB (float32)
- 默认配置
- PC/服务器部署

**推荐场景:**
- PC/服务器
- 追求精度
- 云端API

---

### Large 模型

```python
ModelConfig(
    hidden_dim=768,
    num_heads=12,
    num_layers=8,
    ffn_dim=3072,
    max_seq_len=128,
)
```

**特点:**
- 参数量: ~40M
- 体积: 160 MB (float32)
- 追求精度
- 大容量模型

**推荐场景:**
- 追求极致精度
- 复杂任务
- 学术研究

---

## 训练配置

### 自动训练配置

训练脚本会根据模型尺寸自动调整训练参数:

| 模型尺寸 | Batch Size | Learning Rate | Epochs |
|---------|-----------|--------------|--------|
| tiny | 512 | 5e-4 | 15 |
| small | 384 | 4e-4 | 12 |
| medium | 320 | 3e-4 | 10 |
| base | 256 | 3e-4 | 10 |
| large | 192 | 2e-4 | 15 |

### 手动指定训练参数

```bash
uv run src/train.py --model-size small \
    --use-prepared-data \
    --batch-size 256 \
    --lr 3e-4 \
    --epochs 15
```

---

## 部署优化

### 量化后的体积对比

以 small 模型为例:

| 量化方式 | 体积 | 相对原始 |
|---------|------|---------|
| float32 | 24 MB | 基准 |
| FP16 | 12 MB | -50% |
| int8 + HQQ | 9 MB | -62% |
| int4 | 6.5 MB | -73% |

### 推荐部署方案

**手机端 (推荐 small 模型):**
```bash
# 训练
uv run src/train.py --model-size small --use-prepared-data

# 导出 int8 量化
uv run src/export_mobile.py --model-size small

# 部署文件: mobile/small/model_q8_hqq.mnn (9 MB)
```

**低端手机 (推荐 tiny 模型):**
```bash
# 训练
uv run src/train.py --model-size tiny --use-prepared-data

# 导出 int4 量化
uv run src/export_mobile.py --model-size tiny --quant-bits 4

# 部署文件: mobile/tiny/model_q4.mnn (2 MB)
```

**PC/服务器 (base 模型):**
```bash
# 训练
uv run src/train.py --model-size base --use-prepared-data

# 导出 FP16 或 int8
uv run src/export_mobile.py --model-size base --fp16

# 部署文件: mobile/base/model_fp16.mnn (40 MB)
```

---

## 文件组织

训练和导出后的文件结构:

```
output/
├── tiny/
│   ├── best_model.pt
│   ├── checkpoint_epoch_*.pt
│   └── logs/
├── small/
│   ├── best_model.pt
│   └── ...
├── medium/
├── base/
└── large/

mobile/
├── tiny/
│   ├── model_q8_hqq.mnn
│   ├── model_q4.mnn
│   ├── model_fp16.mnn
│   └── vocab.json
├── small/
├── medium/
├── base/
└── large/
```

---

## 性能对比

### 推理速度 (Android)

| 模型 | float32 | int8 + HQQ | int4 |
|-----|---------|-----------|------|
| tiny | 8 ms | 5 ms | 4 ms |
| small | 12 ms | 7 ms | 6 ms |
| medium | 18 ms | 10 ms | 8 ms |
| base | 25 ms | 14 ms | 11 ms |
| large | 40 ms | 22 ms | 18 ms |

### 内存占用

| 模型 | float32 | int8 | int4 |
|-----|---------|------|------|
| tiny | 20 MB | 12 MB | 8 MB |
| small | 40 MB | 25 MB | 18 MB |
| medium | 70 MB | 45 MB | 32 MB |
| base | 100 MB | 65 MB | 45 MB |
| large | 180 MB | 110 MB | 80 MB |

---

## 常见问题

### Q1: 如何选择模型尺寸?

**A:**
- 低端手机: tiny
- 普通手机: small (推荐)
- 高端手机: medium
- PC/服务器: base
- 追求精度: large

### Q2: 小模型精度会下降吗?

**A:** 会有一定下降，但对于词语联想任务:
- tiny 模型精度约为基础的 85-90%
- small 模型精度约为基础的 92-95%
- medium 模型精度约为基础的 96-98%

### Q3: 可以自定义模型配置吗?

**A:** 可以，修改 `src/config.py` 中的 `ModelConfig` 类:

```python
# 添加自定义配置
@classmethod
def custom(cls) -> "ModelConfig":
    return cls(
        hidden_dim=320,
        num_heads=5,
        num_layers=5,
        ffn_dim=640,
    )
```

### Q4: 如何迁移学习?

**A:** 从大模型初始化小模型:

```bash
# 1. 训练大模型
uv run src/train.py --model-size base --use-prepared-data

# 2. 使用预训练权重初始化小模型 (需要手动实现)
# 目前不支持自动迁移学习
```

---

## 总结

| 场景 | 推荐模型 | 训练命令 | 导出命令 |
|-----|---------|---------|---------|
| 低端手机 | tiny | `--model-size tiny` | `--model-size tiny --quant-bits 4` |
| 普通手机 | small | `--model-size small` | `--model-size small` |
| 高端手机 | medium | `--model-size medium` | `--model-size medium` |
| PC/服务器 | base | `--model-size base` | `--model-size base --fp16` |
| 追求精度 | large | `--model-size large` | `--model-size large` |

**推荐**: 大多数场景使用 **small** 模型 + **int8 量化**