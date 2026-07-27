#!/usr/bin/env python3
"""ncnn int8 量化 (需要 ncnn2table + ncnn2int8 工具)"""

import subprocess
import sys
from pathlib import Path
import argparse

NCNN_BIN = Path(r"C:\Users\kingz\Downloads\ncnn-20260526-windows-vs2022\ncnn-20260526-windows-vs2022\x86\bin")


def run(cmd, desc):
    print(f"\n[{desc}]")
    print(f"  {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout[-500:]}")
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"{desc} 失败 (code={result.returncode})")
    # print last few lines of output
    out = (result.stdout or "") + (result.stderr or "")
    for line in out.strip().split("\n")[-5:]:
        print(f"  {line}")
    print("  [OK]")


def main():
    parser = argparse.ArgumentParser(description="ncnn int8 量化")
    parser.add_argument("--model-size", default="base", help="模型尺寸")
    parser.add_argument("--table", default=None, help="已有 table 文件路径（跳过 ncnn2table）")
    args = parser.parse_args()

    model_dir = Path(f"mobile/{args.model_size}")
    param = model_dir / "model.param"
    bin_fp32 = model_dir / "model.bin"

    if not param.exists() or not bin_fp32.exists():
        print(f"[ERROR] 请先导出 fp32 模型: uv run scripts/export_ncnn.py --model-size {args.model_size} --no-fp16")
        return 1

    # 检查 ncnn2table.exe / ncnn2int8.exe
    ncnn2table = NCNN_BIN / "ncnn2table.exe"
    ncnn2int8 = NCNN_BIN / "ncnn2int8.exe"
    if not ncnn2table.exists():
        print(f"[ERROR] 未找到 ncnn2table.exe (expected: {ncnn2table})")
        return 1
    if not ncnn2int8.exists():
        print(f"[ERROR] 未找到 ncnn2int8.exe (expected: {ncnn2int8})")
        return 1

    print("=" * 60)
    print("ncnn int8 量化")
    print("=" * 60)
    print(f"模型: {param.parent}")

    # Step 1: 生成 calibration table
    # 模型包含 MultiHeadAttention + Embed，支持无校准数据生成静态 scale
    table_path = args.table or (model_dir / "model.table")
    if not table_path.exists():
        run(
            [str(ncnn2table), str(param), str(bin_fp32), str(table_path), "method=kl"],
            "ncnn2table (静态量化, 无需校准数据)",
        )
    else:
        print(f"\n[跳过 ncnn2table] 使用已有 table: {table_path}")

    # Step 2: int8 量化
    param_int8 = model_dir / "model-int8.param"
    bin_int8 = model_dir / "model-int8.bin"
    run(
        [str(ncnn2int8), str(param), str(bin_fp32), str(param_int8), str(bin_int8), str(table_path)],
        "ncnn2int8",
    )

    # 结果
    fp32_mb = bin_fp32.stat().st_size / (1024**2)
    int8_mb = bin_int8.stat().st_size / (1024**2)
    ratio = (1 - int8_mb / fp32_mb) * 100

    print("\n" + "=" * 60)
    print("[OK] int8 量化完成")
    print("=" * 60)
    print(f"  fp32: {fp32_mb:.1f} MB")
    print(f"  int8: {int8_mb:.1f} MB ({ratio:.0f}% 压缩)")
    print(f"  输出: {model_dir}/model-int8.param + model-int8.bin")

    return 0


if __name__ == "__main__":
    exit(main())
