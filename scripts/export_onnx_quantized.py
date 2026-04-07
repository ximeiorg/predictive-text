#!/usr/bin/env python3
"""导出 ONNX 模型并进行量化"""

import torch
import torch.onnx
import torch.serialization
import numpy as np
import json
import shutil
from pathlib import Path
import argparse
from tokenizers import Tokenizer

from src.config import ModelConfig, MODEL_SIZES
from src.model.transformer import create_model

torch.serialization.add_safe_globals([ModelConfig])

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        CalibrationDataReader,
    )
except ImportError:
    print("请安装 onnxruntime: pip install onnxruntime")
    exit(1)


def export_onnx(checkpoint_path, output_path, max_seq_len=32):
    """导出 PyTorch 模型到 ONNX（支持动态序列长度）"""
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids)["logits"]

    wrapper = Wrapper(model)
    wrapper.eval()

    dummy_input = torch.randint(
        0, config.vocab_size, (1, max_seq_len), dtype=torch.long
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"导出 ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"✓ ONNX 模型: {size_mb:.1f} MB")
    return output_path, config


def quantize_onnx_dynamic(onnx_path, output_path, quant_type="int8"):
    """动态量化 ONNX 模型"""
    print(f"\n动态量化 ({quant_type})...")

    quant_type_map = {
        "int8": QuantType.QInt8,
        "uint8": QuantType.QUInt8,
        "qint8": QuantType.QInt8,
    }

    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=quant_type_map.get(quant_type, QuantType.QInt8),
        op_types_to_quantize=["MatMul", "Gemm", "Attention", "LSTM", "Conv"],
    )

    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"✓ 量化模型: {output_path} ({size_mb:.1f} MB)")
    return output_path


class TextCalibrationDataReader(CalibrationDataReader):
    """用于静态量化的校准数据读取器"""

    def __init__(self, tokenizer, sample_texts, max_seq_len=64):
        self.tokenizer = tokenizer
        self.sample_texts = sample_texts
        self.max_seq_len = max_seq_len
        self.index = 0

    def get_next(self):
        if self.index >= len(self.sample_texts):
            return None

        text = self.sample_texts[self.index]
        tokens = self.tokenizer.encode(text).ids
        tokens = tokens[: self.max_seq_len]

        self.index += 1

        return {"input_ids": np.array([tokens], dtype=np.int64)}

    def rewind(self):
        self.index = 0

    def get_next(self):
        if self.index >= len(self.sample_texts):
            return None

        text = self.sample_texts[self.index]
        tokens = self.tokenizer.encode(text)
        tokens = tokens[: self.max_seq_len]

        self.index += 1

        return {"input_ids": np.array([tokens], dtype=np.int64)}

    def rewind(self):
        self.index = 0


def quantize_onnx_static(
    onnx_path, output_path, tokenizer_path, sample_texts=None, quant_type="int8"
):
    """静态量化 ONNX 模型（需要校准数据）"""
    print(f"\n静态量化 ({quant_type})...")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    if sample_texts is None:
        sample_texts = [
            "你好世界",
            "今天天气很好",
            "中华人民共和国",
            "这是一个测试句子",
            "输入法联想功能",
            "人工智能技术",
            "机器学习算法",
            "深度神经网络",
            "自然语言处理",
            "计算机视觉应用",
        ]

    print(f"校准样本数: {len(sample_texts)}")

    calibration_reader = TextCalibrationDataReader(tokenizer, sample_texts)

    quant_type_map = {
        "int8": QuantType.QInt8,
        "uint8": QuantType.QUInt8,
        "qint8": QuantType.QInt8,
    }

    quantize_static(
        onnx_path,
        output_path,
        calibration_data_reader=calibration_reader,
        weight_type=quant_type_map.get(quant_type, QuantType.QInt8),
        op_types_to_quantize=["MatMul", "Gemm", "Attention", "LSTM", "Conv"],
    )

    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"✓ 静态量化模型: {output_path} ({size_mb:.1f} MB)")
    return output_path


def verify_onnx(onnx_path, tokenizer_path, test_texts=None):
    """验证 ONNX 模型"""
    print(f"\n验证模型: {onnx_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    session = ort.InferenceSession(onnx_path)

    print(f"\n模型信息:")
    for inp in session.get_inputs():
        print(f"  输入: {inp.name}, shape: {inp.shape}")
    for out in session.get_outputs():
        print(f"  输出: {out.name}, shape: {out.shape}")

    if test_texts is None:
        test_texts = ["你好", "今天天气", "中华人民共和国"]

    print(f"\n测试推理:")
    input_name = session.get_inputs()[0].name

    for text in test_texts:
        tokens = tokenizer.encode(text).ids
        input_data = np.array([tokens], dtype=np.int64)

        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0]

        last_logits = logits[0, -1, :]
        top5_indices = np.argsort(last_logits)[-5:][::-1]
        top5_tokens = [tokenizer.decode([int(idx)]) for idx in top5_indices]

        print(f"  「{text}」 (seq={len(tokens)}) -> Top-5: {top5_tokens}")

    print(f"✓ 验证通过")
    return True


def compare_models(original_path, quantized_path, tokenizer_path, test_texts=None):
    """对比原始模型和量化模型的输出"""
    print(f"\n对比原始模型和量化模型:")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    original_session = ort.InferenceSession(original_path)
    quantized_session = ort.InferenceSession(quantized_path)

    if test_texts is None:
        test_texts = ["你好", "今天天气很好", "中华人民共和国", "输入法联想"]

    input_name = original_session.get_inputs()[0].name

    total_diff = 0
    max_diff = 0

    for text in test_texts:
        tokens = tokenizer.encode(text).ids
        input_data = np.array([tokens], dtype=np.int64)

        orig_outputs = original_session.run(None, {input_name: input_data})[0]
        quant_outputs = quantized_session.run(None, {input_name: input_data})[0]

        diff = np.abs(orig_outputs - quant_outputs)

        orig_top5 = np.argsort(orig_outputs[0, -1, :])[-5:][::-1]
        quant_top5 = np.argsort(quant_outputs[0, -1, :])[-5:][::-1]

        match = np.sum(orig_top5 == quant_top5)

        print(f"\n  「{text}」:")
        print(f"    最大差异: {diff.max():.4f}")
        print(f"    平均差异: {diff.mean():.4f}")
        print(f"    Top-5 匹配: {match}/5")
        print(f"    原始: {[tokenizer.decode([int(i)]) for i in orig_top5]}")
        print(f"    量化: {[tokenizer.decode([int(i)]) for i in quant_top5]}")

        total_diff += diff.mean()
        max_diff = max(max_diff, diff.max())

    print(f"\n总体对比:")
    print(f"  平均差异: {total_diff / len(test_texts):.4f}")
    print(f"  最大差异: {max_diff:.4f}")


def main():
    parser = argparse.ArgumentParser(description="导出并量化 ONNX 模型")
    parser.add_argument("--checkpoint", type=str, default="output/small/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="onnx")
    parser.add_argument(
        "--model-size", type=str, default="small", choices=list(MODEL_SIZES.keys())
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="dynamic",
        choices=["none", "dynamic", "static"],
    )
    parser.add_argument(
        "--quant-type", type=str, default="int8", choices=["int8", "uint8"]
    )
    parser.add_argument("--compare", action="store_true", help="对比原始和量化模型")
    parser.add_argument("--verify", action="store_true", default=True)

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        checkpoint_path = f"output/{args.model_size}/best_model.pt"
        if Path(checkpoint_path).exists():
            args.checkpoint = checkpoint_path
        else:
            print(f"❌ 模型不存在: {args.checkpoint}")
            print(f"请先训练模型: uv run src/train.py --model-size {args.model_size}")
            return 1

    if not Path(args.tokenizer).exists():
        print(f"❌ Tokenizer 不存在: {args.tokenizer}")
        return 1

    output_dir = Path(args.output_dir) / args.model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ONNX 模型导出与量化")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Tokenizer:   {args.tokenizer}")
    print(f"输出目录:    {output_dir}")
    print(f"量化模式:    {args.quant_mode}")
    print(f"量化类型:    {args.quant_type}")
    print("=" * 60)

    onnx_path = str(output_dir / "model.onnx")
    export_onnx(args.checkpoint, onnx_path)

    if args.verify:
        verify_onnx(onnx_path, args.tokenizer)

    if args.quant_mode == "dynamic":
        quantized_path = str(output_dir / f"model_{args.quant_type}_dynamic.onnx")
        quantize_onnx_dynamic(onnx_path, quantized_path, args.quant_type)

        if args.verify:
            verify_onnx(quantized_path, args.tokenizer)

        if args.compare:
            compare_models(onnx_path, quantized_path, args.tokenizer)

    elif args.quant_mode == "static":
        quantized_path = str(output_dir / f"model_{args.quant_type}_static.onnx")
        quantize_onnx_static(
            onnx_path, quantized_path, args.tokenizer, quant_type=args.quant_type
        )

        if args.verify:
            verify_onnx(quantized_path, args.tokenizer)

        if args.compare:
            compare_models(onnx_path, quantized_path, args.tokenizer)

    vocab_src = Path(args.tokenizer)
    if vocab_src.exists():
        vocab_dst = output_dir / "vocab.json"
        shutil.copy(vocab_src, vocab_dst)

        tokenizer = Tokenizer.from_file(args.tokenizer)
        vocab = tokenizer.get_vocab()
        id2word = {v: k for k, v in vocab.items()}

        with open(output_dir / "vocab.txt", "w") as f:
            for i in range(tokenizer.get_vocab_size()):
                f.write(id2word.get(i, f"[UNK_{i}]") + "\n")

        print(f"\n✓ 词表: {tokenizer.get_vocab_size()} 词")

    manifest = {
        "model": args.model_size,
        "vocab_size": tokenizer.get_vocab_size(),
        "quant_mode": args.quant_mode,
        "quant_type": args.quant_type if args.quant_mode != "none" else None,
        "files": {},
    }

    for f in output_dir.glob("*.onnx"):
        manifest["files"][f.name] = f.stat().st_size / (1024**2)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ 导出完成")
    print("=" * 60)
    print(f"\n输出: {output_dir}/")

    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size / (1024**2)
            print(f"  {f.name:<30} {size:>8.1f} MB")

    if args.quant_mode != "none":
        original_size = Path(onnx_path).stat().st_size / (1024**2)
        data_path = Path(onnx_path + ".data")
        if data_path.exists():
            original_size += data_path.stat().st_size / (1024**2)
        quantized_size = Path(quantized_path).stat().st_size / (1024**2)
        compression_ratio = (1 - quantized_size / original_size) * 100
        print(f"\n原始大小: {original_size:.1f} MB")
        print(f"量化大小: {quantized_size:.1f} MB")
        print(f"压缩比:   {compression_ratio:.1f}%")

    print("\n使用方法:")
    print(
        f"  Python: import onnxruntime as ort; session = ort.InferenceSession('{output_dir}/model.onnx')"
    )
    print(
        f"  验证:   uv run scripts/test_onnx_inference.py --onnx {output_dir}/model.onnx"
    )

    return 0


if __name__ == "__main__":
    exit(main())
