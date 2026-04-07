"""Export model to MNN format for mobile deployment."""

import torch
import torch.onnx
import torch.serialization
import numpy as np
import json
import subprocess
import shutil
from pathlib import Path
import argparse

from src.config import ModelConfig, list_model_sizes, MODEL_SIZES
from src.model.transformer import create_model

torch.serialization.add_safe_globals([ModelConfig])


def export_onnx(checkpoint_path, output_path, max_seq_len=32):
    """导出 PyTorch 模型到 ONNX"""
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

    # 检查文件大小
    onnx_size = Path(output_path).stat().st_size / (1024**2)
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        data_size = data_path.stat().st_size / (1024**2)
        print(f"ONNX: {onnx_size:.1f} MB + 数据: {data_size:.1f} MB")
    else:
        print(f"ONNX: {onnx_size:.1f} MB")

    return output_path


def convert_to_mnn(onnx_path, mnn_path, quant_bits=8, use_hqq=False):
    """转换 ONNX 到 MNN"""
    # 查找 MNNConvert
    mnnconvert = shutil.which("MNNConvert")
    if not mnnconvert:
        mnnconvert = str(Path.home() / ".local/bin/MNNConvert")
        if not Path(mnnconvert).exists():
            print("❌ MNNConvert 未找到")
            print(
                "安装: git clone https://github.com/alibaba/MNN.git && cd MNN && cmake -B build -DMNN_BUILD_CONVERTER=ON && cmake --build build --target MNNConvert"
            )
            return False

    cmd = [
        mnnconvert,
        "-f",
        "ONNX",
        "--modelFile",
        onnx_path,
        "--MNNModel",
        mnn_path,
        "--bizCode",
        "input_method",
    ]

    if quant_bits in [4, 8]:
        cmd.extend(["--weightQuantBits", str(quant_bits)])
        if use_hqq:
            cmd.append("--hqq")
        print(f"转换 MNN int{quant_bits}...")
    else:
        print("转换 MNN float32...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size_mb = Path(mnn_path).stat().st_size / (1024**2)
        print(f"✓ MNN: {mnn_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ 转换失败: {result.stderr}")
        return False


def verify_onnx(onnx_path):
    """验证 ONNX 模型"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime 未安装，跳过验证")
        return None

    print("\n验证 ONNX...")
    session = ort.InferenceSession(onnx_path)

    # 打印信息
    for inp in session.get_inputs():
        print(f"  输入: {inp.name} {inp.shape}")
    for out in session.get_outputs():
        print(f"  输出: {out.name} {out.shape}")

    # 测试推理
    input_name = session.get_inputs()[0].name
    test_input = np.array([[1, 100, 200, 300]], dtype=np.int64)
    outputs = session.run(None, {input_name: test_input})

    print(f"  测试推理: {test_input.shape} → {outputs[0].shape}")

    # 检查输出
    last_logits = outputs[0][0, -1, :]
    top5 = np.argsort(last_logits)[-5:][::-1]
    print(f"  Top-5: {top5.tolist()}")

    print("✓ ONNX 验证通过")
    return True


def main():
    parser = argparse.ArgumentParser(description="导出模型到 MNN")
    parser.add_argument("--checkpoint", type=str, help="模型路径")
    parser.add_argument(
        "--model-size", default="base", choices=list(MODEL_SIZES.keys())
    )
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--quant-bits", type=int, default=8, choices=[4, 8, 32])
    parser.add_argument("--hqq", action="store_true", default=True)
    parser.add_argument("--no-hqq", action="store_true")
    parser.add_argument("--all", action="store_true", help="导出所有量化版本")
    parser.add_argument("--verify", action="store_true", default=True, help="验证模型")

    args = parser.parse_args()

    if args.no_hqq:
        args.hqq = False

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
    print(f"量化位数:    {args.quant_bits}")
    print("=" * 60)

    # 确定要导出的配置
    if args.all:
        configs = [
            (4, False, "model_q4.mnn"),
            (8, True, "model_q8_hqq.mnn"),
            (32, False, "model.mnn"),
        ]
    else:
        if args.quant_bits == 32:
            filename = "model.mnn"
        elif args.quant_bits == 8 and args.hqq:
            filename = "model_q8_hqq.mnn"
        else:
            filename = f"model_q{args.quant_bits}.mnn"
        configs = [(args.quant_bits, args.hqq, filename)]

    # 导出 ONNX
    onnx_path = str(output_dir / "model.onnx")
    export_onnx(checkpoint_path, onnx_path)

    if args.verify:
        verify_onnx(onnx_path)

    # 转换为 MNN
    models = {}
    for bits, hqq, filename in configs:
        mnn_path = str(output_dir / filename)
        if convert_to_mnn(onnx_path, mnn_path, bits, hqq):
            models[filename.replace(".mnn", "")] = mnn_path

    # 清理 ONNX
    Path(onnx_path).unlink(missing_ok=True)
    Path(onnx_path + ".data").unlink(missing_ok=True)

    # 导出词表
    vocab_src = Path("data/vocab.json")
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
        "vocab_size": len(vocab) if vocab_src.exists() else 8000,
        "models": {k: Path(v).stat().st_size / (1024**2) for k, v in models.items()},
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
    print("Android 集成:")
    print("  dependencies { implementation 'com.github.alibaba:MNN:3.4.1' }")
    print("\n验证命令:")
    print(f"  uv run scripts/verify_model.py --all")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
