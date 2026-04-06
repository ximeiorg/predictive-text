"""Quantize model with calibration data for better accuracy."""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from src.config import ModelConfig
from src.model.transformer import create_model


class CalibratedQuantizer:
    """Quantize model with calibration dataset."""

    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.scale_dict = {}
        self.zero_point_dict = {}

    def collect_stats(self):
        """Collect activation statistics from calibration data."""
        print("Collecting activation statistics...")

        activation_stats = {}

        def hook_fn(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = {"min": [], "max": []}
                if isinstance(output, torch.Tensor):
                    activation_stats[name]["min"].append(output.min().item())
                    activation_stats[name]["max"].append(output.max().item())

            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.calibration_data, desc="Calibrating"):
                input_ids = torch.tensor(batch, dtype=torch.long).unsqueeze(0)
                self.model(input_ids)

        for hook in hooks:
            hook.remove()

        return activation_stats

    def quantize_weights_int8(self):
        """Quantize weights to int8 with computed scales."""
        print("Quantizing weights to int8...")

        quantized_weights = {}

        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                weight = param.data

                w_min = weight.min().item()
                w_max = weight.max().item()

                scale = (w_max - w_min) / 255.0
                zero_point = round(-w_min / scale)
                zero_point = max(0, min(255, zero_point))

                quantized = (
                    ((weight / scale) + zero_point)
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                )

                quantized_weights[name] = {
                    "data": quantized,
                    "scale": scale,
                    "zero_point": zero_point,
                }
            else:
                quantized_weights[name] = {
                    "data": param.data,
                    "scale": None,
                    "zero_point": None,
                }

        return quantized_weights

    def quantize_weights_int4(self):
        """Quantize weights to int4 (NF4 or symmetric)."""
        print("Quantizing weights to int4...")

        quantized_weights = {}

        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                weight = param.data.float()

                w_max = weight.abs().max().item()
                scale = w_max / 7.0

                quantized = (weight / scale).round().clamp(-8, 7).to(torch.int8)

                quantized_weights[name] = {
                    "data": quantized,
                    "scale": scale,
                    "zero_point": 0,
                    "bits": 4,
                }
            else:
                quantized_weights[name] = {
                    "data": param.data,
                    "scale": None,
                    "zero_point": None,
                }

        return quantized_weights

    def save_quantized_model(self, output_path, quant_bits=8):
        """Save quantized model with metadata."""
        checkpoint = torch.load(
            self.model_checkpoint, map_location="cpu", weights_only=False
        )
        config = checkpoint.get("config", ModelConfig())

        if quant_bits == 4:
            quantized_weights = self.quantize_weights_int4()
        else:
            quantized_weights = self.quantize_weights_int8()

        quant_state_dict = {}
        scales = {}

        for name, info in quantized_weights.items():
            quant_state_dict[name] = info["data"]
            if info["scale"] is not None:
                scales[name] = {
                    "scale": info["scale"],
                    "zero_point": info["zero_point"],
                }

        torch.save(
            {
                "model_state_dict": quant_state_dict,
                "quant_scales": scales,
                "config": config,
                "quantized": True,
                "quant_bits": quant_bits,
            },
            output_path,
        )

        self._report_size(output_path)

    def _report_size(self, output_path):
        original_size = Path(self.model_checkpoint).stat().st_size / (1024 * 1024)
        quant_size = Path(output_path).stat().st_size / (1024 * 1024)

        print(f"\nOriginal model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quant_size:.2f} MB")
        print(f"Size reduction: {(1 - quant_size / original_size) * 100:.1f}%")
        print(f"Saved to: {output_path}")


def load_calibration_data(data_path: str, vocab_path: str, num_samples: int = 1000):
    """Load calibration data from training data."""
    print(f"Loading {num_samples} calibration samples...")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    data = np.fromfile(data_path, dtype=np.uint16)

    samples = []
    stride = max(1, len(data) // num_samples)

    for i in range(0, len(data) - 32, stride):
        if i >= num_samples:
            break
        sample = data[i : i + 32].tolist()
        if len(sample) == 32:
            samples.append(sample)

    print(f"Loaded {len(samples)} calibration samples")
    return samples


def quantize_pytorch_native(
    checkpoint_path: str, output_path: str, quant_bits: int = 8
):
    """Use PyTorch's native quantization."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class QuantizableWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)["logits"]

    wrapper = QuantizableWrapper(model)
    wrapper.eval()

    if quant_bits == 8:
        quantized = torch.quantization.quantize_dynamic(
            wrapper, {nn.Linear, nn.Embedding}, dtype=torch.qint8
        )
    else:
        quantized = torch.quantization.quantize_dynamic(
            wrapper, {nn.Linear}, dtype=torch.qint8
        )

    torch.save(
        {
            "model_state_dict": quantized.state_dict(),
            "config": config,
            "quantized": True,
            "quant_bits": quant_bits,
            "method": "pytorch_dynamic",
        },
        output_path,
    )

    original_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"PyTorch quantized: {original_size:.2f} MB → {quant_size:.2f} MB")


def quantize_to_onnx_int8(
    checkpoint_path: str, output_path: str, calibration_data=None
):
    """Export to quantized ONNX format."""
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            quantize_dynamic,
            quantize_static,
            QuantType,
        )
    except ImportError:
        print("Install onnx and onnxruntime for ONNX quantization:")
        print("  pip install onnx onnxruntime")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class QuantizableWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)["logits"]

    wrapper = QuantizableWrapper(model)
    wrapper.eval()

    float_onnx_path = Path(output_path).parent / "model_float.onnx"
    dummy_input = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(float_onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    if calibration_data:
        print("Performing static quantization with calibration...")

        def preprocess_input(data):
            return {"input_ids": np.array(data, dtype=np.int64).reshape(1, -1)}

        quantize_static(
            str(float_onnx_path),
            output_path,
            calibration_data,
            preprocess_input,
            weight_type=QuantType.QInt8,
        )
    else:
        print("Performing dynamic quantization...")
        quantize_dynamic(
            str(float_onnx_path),
            output_path,
            weight_type=QuantType.QInt8,
        )

    float_size = float_onnx_path.stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"ONNX quantized: {float_size:.2f} MB → {quant_size:.2f} MB")

    float_onnx_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Quantize model with calibration")
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pt")
    parser.add_argument("--vocab", type=str, default="data/vocab.json")
    parser.add_argument("--train-data", type=str, default="data/train.bin")
    parser.add_argument("--output-dir", type=str, default="mobile")
    parser.add_argument("--quant-bits", type=int, choices=[4, 8], default=8)
    parser.add_argument(
        "--method",
        type=str,
        choices=["pytorch", "onnx_dynamic", "onnx_static", "custom"],
        default="onnx_dynamic",
    )
    parser.add_argument("--calibration-samples", type=int, default=100)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Model Quantization (int{args.quant_bits})")
    print("=" * 60)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Method:      {args.method}")
    print(f"Output:      {args.output_dir}")
    print("=" * 60 + "\n")

    calibration_data = None
    if args.method == "onnx_static":
        calibration_data = load_calibration_data(
            args.train_data, args.vocab, args.calibration_samples
        )

    output_name = f"model_q{args.quant_bits}"

    if args.method == "pytorch":
        output_path = output_dir / f"{output_name}.pt"
        quantize_pytorch_native(args.checkpoint, str(output_path), args.quant_bits)

    elif args.method.startswith("onnx"):
        output_path = output_dir / f"{output_name}.onnx"
        quantize_to_onnx_int8(
            args.checkpoint,
            str(output_path),
            calibration_data if args.method == "onnx_static" else None,
        )

    elif args.method == "custom":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", ModelConfig())

        model = create_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        calibration_data = load_calibration_data(
            args.train_data, args.vocab, args.calibration_samples
        )

        quantizer = CalibratedQuantizer(model, calibration_data)
        quantizer.model_checkpoint = args.checkpoint

        output_path = output_dir / f"{output_name}_calibrated.pt"
        quantizer.save_quantized_model(str(output_path), args.quant_bits)

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
