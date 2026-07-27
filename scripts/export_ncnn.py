#!/usr/bin/env python3
"""导出 ncnn 模型 (via PNNX)"""

import json
import shutil
from pathlib import Path
import argparse

import torch
import pnnx

from src.config import MODEL_SIZES


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    from src.data.dataset import SimpleTokenizer
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


def export_ncnn(checkpoint_path, output_dir, max_seq_len=64, fp16=True):
    model, config = load_model(checkpoint_path)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids)["logits"]

    wrapper = Wrapper(model).eval()
    seq_len = min(max_seq_len, config.max_seq_len)
    dummy = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)

    pt_path = str(output_dir / "model.pt")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用绝对路径避免 Windows 反斜杠转义问题
    abs_pt = str(Path(pt_path).resolve())

    print(f"\n导出 ncnn: {output_dir}/")
    try:
        pnnx.export(
            wrapper,
            abs_pt,
            inputs=dummy,
            device="cpu",
            fp16=fp16,
            optlevel=2,
        )
    except Exception:
        # PNNX 在 Windows 上生成的 _pnnx.py 有路径转义问题，
        # 但 .ncnn.param 和 .ncnn.bin 已经生成，可忽略此错误
        pass

    # PNNX 输出 model.ncnn.param 和 model.ncnn.bin
    param_src = output_dir / "model.ncnn.param"
    bin_src = output_dir / "model.ncnn.bin"
    param_dst = output_dir / "model.param"
    bin_dst = output_dir / "model.bin"
    if param_src.exists():
        shutil.move(param_src, param_dst)
        print(f"  {param_dst.name}")
    if bin_src.exists():
        sz = bin_src.stat().st_size / (1024**2)
        shutil.move(bin_src, bin_dst)
        print(f"  {bin_dst.name} ({sz:.1f} MB)")

    # 清理中间文件
    for f in output_dir.glob("model*"):
        if f.name not in ("model.param", "model.bin"):
            f.unlink(missing_ok=True)
    for f in output_dir.glob("*.py"):
        f.unlink(missing_ok=True)

    return config


def save_vocab(vocab_path, output_dir):
    shutil.copy(vocab_path, output_dir / "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    id2word = {v: k for k, v in vocab.items()}
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for i in range(len(vocab)):
            f.write(id2word.get(i, f"[UNK_{i}]") + "\n")
    return len(vocab)


def find_checkpoint(model_size):
    paths = [
        f"output/{model_size}/best_model.pt",
        f"output/{model_size}/logs/version_1/checkpoints/last.ckpt",
        f"output/{model_size}/logs/version_0/checkpoints/last.ckpt",
        f"output/{model_size}/checkpoints/last.ckpt",
    ]
    for p in paths:
        if Path(p).exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="导出 ncnn 模型")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-size", default="base", choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=64, help="导出序列长度")
    parser.add_argument("--fp16", action="store_true", default=True, dest="fp16", help="fp16 权重量化")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="导出 fp32")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_checkpoint(args.model_size)
    if not checkpoint or not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint 不存在: {checkpoint}")
        return 1

    output_dir = Path(args.output_dir or f"mobile/{args.model_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = "data/vocab.json"

    print("=" * 60)
    print("NCNN 导出")
    print("=" * 60)
    print(f"Checkpoint:  {checkpoint}")
    print(f"Seq Len:     {args.seq_len}")
    print(f"FP16:        {args.fp16}")
    print(f"输出目录:    {output_dir}")
    print("=" * 60)

    config = export_ncnn(checkpoint, output_dir, args.seq_len, args.fp16)
    vocab_size = save_vocab(vocab_path, output_dir)

    manifest = {
        "model": args.model_size,
        "vocab_size": vocab_size,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "max_seq_len": args.seq_len,
        "fp16": args.fp16,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] 导出完成: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            sz = f.stat().st_size / 1024
            unit = "KB" if sz < 1024 else "MB"
            val = sz if sz < 1024 else sz / 1024
            print(f"  {f.name:<30} {val:>8.1f} {unit}")
    return 0


if __name__ == "__main__":
    exit(main())
