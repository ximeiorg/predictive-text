"""Export model to ONNX format for mobile deployment."""

import torch
import torch.onnx
import torch.serialization
import numpy as np
import json
import shutil
from pathlib import Path
import argparse

from src.config import ModelConfig, list_model_sizes, MODEL_SIZES
from src.model.transformer import create_model

torch.serialization.add_safe_globals([ModelConfig])


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
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        data_size = data_path.stat().st_size / (1024**2)
        print(f"ONNX: {size_mb:.1f} MB + 数据: {data_size:.1f} MB")
    else:
        print(f"ONNX: {size_mb:.1f} MB")

    return output_path


def verify_onnx(onnx_path):
    """验证 ONNX 模型"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime 未安装，跳过验证")
        return None

    print("\n验证 ONNX...")
    session = ort.InferenceSession(onnx_path)

    for inp in session.get_inputs():
        print(f"  输入: {inp.name} {inp.shape}")
    for out in session.get_outputs():
        print(f"  输出: {out.name} {out.shape}")

    input_name = session.get_inputs()[0].name

    test_input = np.array([[1, 100, 200, 300]], dtype=np.int64)
    outputs = session.run(None, {input_name: test_input})
    print(f"  测试推理 (seq=4): {test_input.shape} → {outputs[0].shape}")

    test_input2 = np.array([[1, 100]], dtype=np.int64)
    outputs2 = session.run(None, {input_name: test_input2})
    print(f"  测试推理 (seq=2): {test_input2.shape} → {outputs2[0].shape}")

    last_logits = outputs[0][0, -1, :]
    top5 = np.argsort(last_logits)[-5:][::-1]
    print(f"  Top-5: {top5.tolist()}")

    print("✓ ONNX 验证通过（支持动态序列长度）")
    return True


def main():
    parser = argparse.ArgumentParser(description="导出模型到 ONNX")
    parser.add_argument("--checkpoint", type=str, help="模型路径")
    parser.add_argument(
        "--model-size", default="base", choices=list(MODEL_SIZES.keys())
    )
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--verify", action="store_true", default=True, help="验证模型")

    args = parser.parse_args()

    # 确定 checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = f"output/{args.model_size}/best_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        print(f"\n训练模型:")
        print(
            f"  uv run src/train.py --model-size {args.model_size} --use-prepared-data"
        )
        return 1

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"mobile/{args.model_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("模型导出")
    print("=" * 60)
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"输出目录:    {output_dir}")
    print("=" * 60)

    # 导出 ONNX
    onnx_path = str(output_dir / "model.onnx")
    export_onnx(checkpoint_path, onnx_path)

    if args.verify:
        verify_onnx(onnx_path)

    # 导出词表
    vocab_src = Path("data/vocab.json")
    vocab = None
    if vocab_src.exists():
        vocab_dst = output_dir / "vocab.json"
        shutil.copy(vocab_src, vocab_dst)

        # 创建 vocab.txt
        with open(vocab_src) as f:
            vocab = json.load(f)
        id2word = {v: k for k, v in vocab.items()}

        with open(output_dir / "vocab.txt", "w") as f:
            for i in range(len(vocab)):
                f.write(id2word.get(i, f"[UNK_{i}]") + "\n")

        print(f"\n✓ 词表: {len(vocab)} 词")

    # 创建 manifest
    manifest = {
        "model": args.model_size,
        "vocab_size": len(vocab) if vocab else 8000,
        "onnx_size_mb": Path(onnx_path).stat().st_size / (1024**2),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 总结
    print("\n" + "=" * 60)
    print("✅ 导出完成")
    print("=" * 60)
    print(f"\n输出: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1024
            unit = "KB" if size < 1024 else "MB"
            size_str = f"{size:.1f} {unit}" if size < 1024 else f"{size / 1024:.1f} MB"
            print(f"  {f.name:<20} {size_str:>10}")

    print("\n" + "-" * 60)
    print("\n验证命令:")
    print(f"  uv run scripts/verify_model.py --onnx {onnx_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
