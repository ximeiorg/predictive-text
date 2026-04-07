#!/usr/bin/env python3
"""验证 ONNX 模型能否正确加载和推理"""

import numpy as np
import sys
from pathlib import Path


def verify_onnx(model_path: str):
    """使用 ONNX Runtime 验证模型"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ 请安装 onnxruntime: pip install onnxruntime")
        return False

    print("\n" + "=" * 60)
    print("ONNX 模型验证")
    print("=" * 60)

    print(f"\n1. 加载模型: {model_path}")
    try:
        session = ort.InferenceSession(model_path)
        print("   ✓ 加载成功")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False

    print("\n2. 模型信息:")
    for inp in session.get_inputs():
        print(f"   输入: {inp.name}")
        print(f"      类型: {inp.type}")
        print(f"      形状: {inp.shape}")

    for out in session.get_outputs():
        print(f"   输出: {out.name}")
        print(f"      类型: {out.type}")
        print(f"      形状: {out.shape}")

    print("\n3. 测试推理...")
    input_name = session.get_inputs()[0].name

    test_inputs = [
        np.array([[1, 100, 200]], dtype=np.int64),
        np.array([[1, 100, 200, 300, 400]], dtype=np.int64),
        np.array([[1, 100, 200, 300, 400, 500]], dtype=np.int64),
    ]

    for i, input_data in enumerate(test_inputs):
        try:
            outputs = session.run(None, {input_name: input_data})
            print(
                f"   测试 {i + 1}: 输入 {input_data.shape} → 输出 {outputs[0].shape} ✓"
            )
        except Exception as e:
            print(f"   测试 {i + 1}: ❌ {e}")
            return False

    print("\n4. 验证输出...")
    input_data = np.array([[1, 100, 200, 300, 400]], dtype=np.int64)
    outputs = session.run(None, {input_name: input_data})

    logits = outputs[0]
    print(f"   Logits 形状: {logits.shape}")
    print(f"   Logits 类型: {logits.dtype}")
    print(f"   Logits 范围: [{logits.min():.2f}, {logits.max():.2f}]")

    last_logits = logits[0, -1, :]
    top5_idx = np.argsort(last_logits)[-5:][::-1]
    print(f"\n   Top-5 预测 (最后位置):")
    for i, idx in enumerate(top5_idx):
        print(f"     {i + 1}. Token {idx}: {last_logits[idx]:.2f}")

    print("\n✅ ONNX 模型验证通过!")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证导出的模型")
    parser.add_argument("--onnx", type=str, help="ONNX 模型路径")
    parser.add_argument("--all", action="store_true", help="验证 mobile/ 下所有模型")

    args = parser.parse_args()

    if args.all:
        mobile_dir = Path("mobile")
        if not mobile_dir.exists():
            print("❌ mobile/ 目录不存在")
            return 1

        for size_dir in mobile_dir.iterdir():
            if size_dir.is_dir():
                print(f"\n{'=' * 60}")
                print(f"验证 {size_dir.name} 模型")
                print("=" * 60)

                onnx_path = size_dir / "model.onnx"

                if onnx_path.exists():
                    verify_onnx(str(onnx_path))

        return 0

    if args.onnx:
        verify_onnx(args.onnx)

    if not (args.onnx or args.all):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
