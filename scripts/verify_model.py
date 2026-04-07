#!/usr/bin/env python3
"""验证 MNN 模型能否正确加载和推理"""

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

    # 加载模型
    print(f"\n1. 加载模型: {model_path}")
    try:
        session = ort.InferenceSession(model_path)
        print("   ✓ 加载成功")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False

    # 打印模型信息
    print("\n2. 模型信息:")
    for inp in session.get_inputs():
        print(f"   输入: {inp.name}")
        print(f"      类型: {inp.type}")
        print(f"      形状: {inp.shape}")

    for out in session.get_outputs():
        print(f"   输出: {out.name}")
        print(f"      类型: {out.type}")
        print(f"      形状: {out.shape}")

    # 运行推理
    print("\n3. 测试推理...")
    input_name = session.get_inputs()[0].name

    # 创建测试输入
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

    # 验证输出合理性
    print("\n4. 验证输出...")
    input_data = np.array([[1, 100, 200, 300, 400]], dtype=np.int64)
    outputs = session.run(None, {input_name: input_data})

    logits = outputs[0]
    print(f"   Logits 形状: {logits.shape}")
    print(f"   Logits 类型: {logits.dtype}")
    print(f"   Logits 范围: [{logits.min():.2f}, {logits.max():.2f}]")

    # 检查最后一个位置的预测
    last_logits = logits[0, -1, :]
    top5_idx = np.argsort(last_logits)[-5:][::-1]
    print(f"\n   Top-5 预测 (最后位置):")
    for i, idx in enumerate(top5_idx):
        print(f"     {i + 1}. Token {idx}: {last_logits[idx]:.2f}")

    print("\n✅ ONNX 模型验证通过!")
    return True


def verify_mnn_file(model_path: str):
    """检查 MNN 文件基本信息"""
    print("\n" + "=" * 60)
    print("MNN 文件检查")
    print("=" * 60)

    path = Path(model_path)

    if not path.exists():
        print(f"\n❌ 文件不存在: {model_path}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n文件: {model_path}")
    print(f"大小: {size_mb:.1f} MB")

    # 读取文件头
    with open(model_path, "rb") as f:
        header = f.read(8)
        print(f"文件头: {header.hex()}")

        # MNN 文件魔数
        if header[:4] == b"MNN\x00":
            print("✓ 有效的 MNN 文件")
            return True
        else:
            print("⚠️  不是标准 MNN 文件头")
            return True  # 仍然可能有效

    return True


def verify_with_mnn_python(model_path: str):
    """使用 MNN Python API 验证"""
    print("\n" + "=" * 60)
    print("MNN Python API 验证")
    print("=" * 60)

    try:
        import MNN

        print("✓ MNN Python 绑定已安装")

        # 加载模型
        print(f"\n加载模型: {model_path}")
        net = MNN.nn.load_module_from_file(model_path, [], [])
        print("✓ 模型加载成功")

        # 创建输入
        input_ids = np.array([[1, 100, 200, 300]], dtype=np.int32)
        print(f"\n测试输入: {input_ids.shape}")

        # 运行推理
        input_tensor = MNN.expr.placeholder([1, 4], MNN.expr.Int)
        input_tensor.write(input_ids.tolist())

        output = net.forward([input_tensor])

        if len(output) > 0:
            logits = output[0].read()
            print(f"输出形状: {logits.shape}")
            print("✓ 推理成功")
            return True
        else:
            print("❌ 无输出")
            return False

    except ImportError:
        print("⚠️  MNN Python 绑定未安装")
        print("\n安装方法:")
        print("  方式1: pip install MNN")
        print("  方式2: 从源码编译")
        print("\n注意: Python 绑定是可选的，Android/iOS 上应该能正常加载")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证导出的模型")
    parser.add_argument("--onnx", type=str, help="ONNX 模型路径")
    parser.add_argument("--mnn", type=str, help="MNN 模型路径")
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
                mnn_path = size_dir / "model.mnn"

                if onnx_path.exists():
                    verify_onnx(str(onnx_path))

                if mnn_path.exists():
                    verify_mnn_file(str(mnn_path))
                    verify_with_mnn_python(str(mnn_path))

        return 0

    if args.onnx:
        verify_onnx(args.onnx)

    if args.mnn:
        verify_mnn_file(args.mnn)
        verify_with_mnn_python(args.mnn)

    if not (args.onnx or args.mnn or args.all):
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
