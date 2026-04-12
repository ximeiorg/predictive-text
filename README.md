# Predictive Text

轻量级中文预测性文本输入模型，基于 Transformer 架构，支持 ONNX 部署。

> 这是为 [https://github.com/ximeiorg/Kime](https://github.com/ximeiorg/Kime) 项目准备的联想词预测模型。

> 目前效果还没达到满意，后续有时间再慢慢优化。

## 功能特性

- 支持多种模型尺寸（tiny/small/medium/base/large）
- BPE Tokenizer，支持多字词联想
- ONNX 导出与量化（INT8）
- 动态序列长度，无需填充

## 快速开始

```bash
# 安装依赖
uv sync

# 训练模型
uv run src/train.py --model-size small --use-prepared-data

# 交互式测试
uv run src/inference_candidate.py --model-size small

# 导出 ONNX
uv run src/export_mobile.py --model-size small
```

## 模型尺寸

| 名称 | 参数量 | 大小 | 说明 |
|------|--------|------|------|
| tiny | ~2M | ~8 MB | 低端手机 |
| small | ~6M | ~24 MB | 推荐手机 |
| medium | ~12M | ~48 MB | 高端手机 |
| base | ~20M | ~80 MB | PC/服务器 |
| large | ~40M | ~160 MB | 追求精度 |

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
│   ├── export_onnx_quantized.py  # ONNX 导出与量化
│   ├── test_onnx_inference.py    # ONNX 测试
│   └── verify_model.py           # 模型验证
├── data/                 # 数据目录
├── output/               # 训练输出
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