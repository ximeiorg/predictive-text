#!/usr/bin/env python3
"""导出 ONNX 模型 (float32 + INT8 动态量化)"""

import json
import shutil
from pathlib import Path
import argparse

import torch
import torch.onnx
import numpy as np

from src.config import MODEL_SIZES
from src.model.lightning_module import DecoderTransformerLightningModule
from src.data.dataset import SimpleTokenizer

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return SimpleTokenizer(vocab)


def export_onnx_float32(checkpoint_path, output_path, max_seq_len=64):
    pl_module = DecoderTransformerLightningModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model = pl_module.model
    config = pl_module.model_config
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids)["logits"]

    wrapper = Wrapper(model).eval()

    dummy = torch.randint(0, config.vocab_size, (1, max_seq_len), dtype=torch.long)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n导出 ONNX float32: {output_path}")
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )
    return _model_size(output_path)


def quantize_int8_dynamic(onnx_path, output_path, quant_type="int8"):
    print(f"\nINT8 动态量化: {output_path}")
    quant_map = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
    quantize_dynamic(
        onnx_path, output_path,
        weight_type=quant_map.get(quant_type, QuantType.QInt8),
        op_types_to_quantize=["MatMul", "Gemm", "Attention", "LSTM", "Conv"],
    )
    return _model_size(output_path)


def verify(onnx_path, tokenizer):
    if not HAS_ORT:
        print("  [skip] onnxruntime 未安装")
        return

    session = ort.InferenceSession(onnx_path)
    for inp in session.get_inputs():
        print(f"  输入: {inp.name} {inp.shape}")
    for out in session.get_outputs():
        print(f"  输出: {out.name} {out.shape}")

    input_name = session.get_inputs()[0].name
    for text in ["你好", "今天天气", "我觉得", "我们一起去"]:
        ids = tokenizer.encode(text).ids
        inp = np.array([ids], dtype=np.int64)
        logits = session.run(None, {input_name: inp})[0]
        top5 = np.argsort(logits[0, -1, :])[-5:][::-1]
        tokens = [tokenizer.id2token.get(int(i), "?") for i in top5]
        print(f"  [{text}] -> {tokens}")
    print("  [OK]")


def _model_size(path):
    p = Path(path)
    s = p.stat().st_size / (1024**2)
    data = Path(str(p) + ".data")
    if data.exists():
        s += data.stat().st_size / (1024**2)
    print(f"  size: {s:.1f} MB")
    return s


def save_vocab(vocab_path, output_dir):
    shutil.copy(vocab_path, output_dir / "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    id2word = {v: k for k, v in vocab.items()}
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for i in range(len(vocab)):
            f.write(id2word.get(i, f"[UNK_{i}]") + "\n")
    return len(vocab)


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX + INT8 量化")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-size", default="small", choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quant", default="int8", choices=["int8", "uint8", "none"])
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint or f"output/{args.model_size}/best_model.pt"
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint 不存在: {checkpoint}")
        return 1

    output_dir = Path(args.output_dir or f"mobile/{args.model_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = "data/vocab.json"
    tokenizer = load_vocab(vocab_path)

    print("=" * 60)
    print("ONNX 导出")
    print("=" * 60)
    print(f"Checkpoint:  {checkpoint}")
    print(f"输出目录:    {output_dir}")
    print("=" * 60)

    float_onnx = str(output_dir / "model.onnx")
    float_size = export_onnx_float32(checkpoint, float_onnx)

    if args.verify:
        print("\n--- 验证 float32 ---")
        verify(float_onnx, tokenizer)

    int8_size = None
    if args.quant != "none":
        int8_onnx = str(output_dir / f"model_{args.quant}_dynamic.onnx")
        int8_size = quantize_int8_dynamic(float_onnx, int8_onnx, args.quant)

        if args.verify:
            print(f"\n--- 验证 {args.quant} ---")
            verify(int8_onnx, tokenizer)

    vocab_size = save_vocab(vocab_path, output_dir)

    manifest = {
        "model": args.model_size,
        "vocab_size": vocab_size,
        "float32_mb": round(float_size, 1),
        "int8_mb": round(int8_size, 1) if int8_size else None,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("[OK] 导出完成")
    print("=" * 60)
    print(f"\n输出: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            sz = f.stat().st_size / 1024
            unit = "KB" if sz < 1024 else "MB"
            val = sz if sz < 1024 else sz / 1024
            print(f"  {f.name:<30} {val:>8.1f} {unit}")

    if int8_size:
        ratio = (1 - int8_size / float_size) * 100
        print(f"\nfloat32: {float_size:.1f} MB -> INT8: {int8_size:.1f} MB ({ratio:.0f}% 压缩)")
    return 0


if __name__ == "__main__":
    exit(main())
