# 项目进度

## 已完成

### 数据准备
- [x] 语料清洗流程 (data/cleaned/all_cleaned.txt)
- [x] BPE 词表训练脚本 (scripts/analyze_and_prune_vocab.py)
- [x] 词表剪枝: 12000 → 5000, 过滤纯标点多字组合
- [x] 最长匹配优先编码: 多字 token 优先匹配, ASCII 逐字符编码
- [x] 跳过不在词表的字符, 训练数据零 UNK
- [x] 生成数据: vocab.json (5002), train.bin (2.6B tokens), val.bin (141M tokens)

### 训练
- [x] 训练脚本: src/train.py, 支持 --prepare-vocab 和 --use-prepared-data
- [x] Decoder-Only Transformer 模型 (tiny/small/medium/base/large)
- [x] GPU 支持 (PyTorch CUDA 13.0 + UV_NO_SYNC)
- [x] small 模型训练完成 (10M 参数, Top-3: 48.68%)

### 推理
- [x] 交互式推理: src/inference.py
- [x] IME 评估脚本: scripts/evaluate_ime.py

### 导出
- [x] ONNX 导出脚本: scripts/export_onnx.py
- [x] INT8 动态量化 (52% 压缩: 39MB → 19MB)
- [x] 修复导出验证逻辑: 移除 EOS token 后再预测

## 下一步
- 优化多字 token 的剪枝策略（目前仅保留 930 个）
- 测试 mobile 端推理性能
