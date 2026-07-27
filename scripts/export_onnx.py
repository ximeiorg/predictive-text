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


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    elif "state_dict" in checkpoint:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
        hp = checkpoint.get("hyper_parameters", {})
        config = hp.get("model_config", {})
    else:
        raise KeyError("Unknown checkpoint format")

    from src.config import ModelConfig
    from src.model.transformer import DecoderTransformer
    mc = ModelConfig.from_dict(config)
    model = DecoderTransformer(mc)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, mc


def export_onnx_float32(model, config, output_path, max_seq_len=64):

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
    skip_ids = {tokenizer.vocab.get(k, -1) for k in ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]}
    
    for text in ["你好", "今天天气", "我觉得", "我们一起去"]:
        ids = tokenizer.encode(text).ids
        if len(ids) > 1:
            ids = ids[:-1]
        inp = np.array([ids], dtype=np.int64)
        logits = session.run(None, {input_name: inp})[0]
        
        last_logits = logits[0, -1, :]
        top_k = 20
        top_indices = np.argsort(last_logits)[-top_k:][::-1]
        
        tokens = []
        for i in top_indices:
            if int(i) not in skip_ids:
                tokens.append(tokenizer.id2token.get(int(i), "?"))
            if len(tokens) >= 5:
                break
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


def _find_checkpoint(model_size):
    paths = [
        f"output/{model_size}/best_model.pt",
        f"output/{model_size}/logs/version_1/checkpoints/last.ckpt",
        f"output/{model_size}/logs/version_0/checkpoints/last.ckpt",
    ]
    for p in paths:
        if Path(p).exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX + INT8 量化")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-size", default="small", choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None, help="序列长度 (默认从模型配置读取)")
    parser.add_argument("--quant", default="int8", choices=["int8", "uint8", "none"])
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint or _find_checkpoint(args.model_size)
    if not checkpoint or not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint 不存在: {checkpoint}")
        return 1

    output_dir = Path(args.output_dir or f"mobile/{args.model_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用临时模型读取 config 中的 max_seq_len
    from src.config import ModelConfig, get_config_manager
    from src.data.dataset import SimpleTokenizer

    vocab_path = "data/vocab.json"
    tokenizer = load_vocab(vocab_path)

    print("=" * 60)
    print("ONNX 导出")
    print("=" * 60)
    print(f"Checkpoint:  {checkpoint}")
    print(f"输出目录:    {output_dir}")
    print("=" * 60)

    # 载入模型获取 config
    model, config = load_model(checkpoint)
    seq_len = args.seq_len or config.max_seq_len or 64
    float_onnx = str(output_dir / "model.onnx")
    float_size = export_onnx_float32(model, config, float_onnx, seq_len)

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
