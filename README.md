# Predictive Text

轻量级中文预测性文本输入模型，基于 Transformer 架构，支持 ONNX 部署。

> 这是为 [https://github.com/ximeiorg/Kime](https://github.com/ximeiorg/Kime) 项目准备的联想词预测模型。

## 功能特性

- 支持多种模型尺寸（tiny/small/medium/base/large）
- 基于 `label.txt` 常用汉字表的词表剪枝，小参数模型也能有效收敛
- 训练时自动忽略句中逗号/顿号的 loss，避免模型偏好预测标点
- ONNX 导出与量化（INT8）
- 动态序列长度，无需填充

## 快速开始

```bash
# 安装依赖
uv sync

# 1. 准备数据：BPE 训练 → 频率分析 → label.txt 过滤剪枝 → tokenize
uv run python scripts/analyze_and_prune_vocab.py

# 2. 训练模型
uv run src/train.py --model-size small --use-prepared-data

# 3. 交互式测试
uv run src/inference_candidate.py --model-size small

# 4. 导出 ONNX
uv run src/export_mobile.py --model-size small
```

## 词表剪枝流程

BPE 词表（~12000）对中文会产生大量低效的字节级碎片 token。
使用 `label.txt` 常见汉字表 + 频率分析做剪枝：

1. 训练标准 BPE tokenizer（12000 词表）
2. 采样 30% 语料统计每个 token 频次
3. 分类：
   - 单字在 `label.txt` 中 → 保留
   - 单字不在 `label.txt` 中 → 丢弃（覆盖率 < 0.2%）
   - 多字组合 → 按频率排序保留 top-N
4. 输出 `data/vocab.json`（~5000 词表）+ `data/train.bin` + `data/val.bin`

## 标点 loss 掩码

训练时 `WikipediaDataset` 自动检测词表中的 `，、；：` 等句中停顿标点，
将其 label 替换为 `pad_id`（ID=0），让 `cross_entropy(ignore_index=0)` 忽略这些位置。

效果：模型不会倾向预测逗号/顿号，句末标点 `。！？` 仍正常学习。

## 模型尺寸

| 名称 | vocabsize | 参数量 | 大小 | 说明 |
|------|-----------|--------|------|------|
| tiny | 5000      | ~1.5M  | ~6 MB | 低端手机 |
| small | 5000     | ~5M    | ~20 MB | 推荐手机 |
| medium | 5000    | ~9M    | ~36 MB | 高端手机 |
| base | 5000      | ~16M   | ~64 MB | PC/服务器 |
| large | 5000     | ~35M   | ~140 MB | 追求精度 |

## 目录结构

```
├── src/
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 推理脚本
│   ├── inference_candidate.py  # 联想词推理
│   ├── export_mobile.py  # ONNX 导出
│   ├── config.py         # 配置定义
│   └── model/            # 模型实现
├── scripts/
│   ├── analyze_and_prune_vocab.py  # 词表分析+剪枝 (核心)
│   ├── prepare_from_cleaned.py     # 数据准备 (备选)
│   ├── export_onnx_quantized.py    # ONNX 导出与量化
│   ├── test_onnx_inference.py      # ONNX 测试
│   └── verify_model.py             # 模型验证
├── data/                 # 数据目录 (vocab.json / train.bin / val.bin)
├── output/               # 训练输出
├── label.txt             # 常用汉字表 (用于词表剪枝)
└── mobile/               # ONNX 模型
```

## 文档

- [训练指南](docs/training_guide.md)
- [模型尺寸配置](docs/model_sizes.md)
- [移动端部署](docs/mobile_deployment.md)
- [优化指南](docs/optimization_guide.md)
- [评估指南](docs/evaluation_guide.md)

## License

CC BY-NC-SA