"""对比评估不同模型尺寸的性能"""

import json
import subprocess
from pathlib import Path
import argparse

from src.config import list_model_sizes, MODEL_SIZES


def evaluate_model(model_size, data_path, vocab_path, max_samples=3000):
    """评估单个模型"""
    checkpoint_path = f"output/{model_size}/best_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"⚠️  {model_size} 模型未训练，跳过")
        return None

    print(f"\n评估 {model_size} 模型...")

    cmd = [
        "uv",
        "run",
        "src/evaluate.py",
        "--model-size",
        model_size,
        "--data-path",
        data_path,
        "--vocab-path",
        vocab_path,
        "--max-samples",
        str(max_samples),
        "--output",
        f"output/{model_size}/eval_report.json",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ 评估失败: {result.stderr}")
        return None

    # 读取结果
    report_path = Path(f"output/{model_size}/eval_report.json")
    if report_path.exists():
        with open(report_path, "r") as f:
            return json.load(f)

    return None


def compare_models(results):
    """对比不同模型的结果"""
    print("\n" + "=" * 80)
    print("模型性能对比")
    print("=" * 80)

    # 表头
    print(
        f"{'模型':<10} {'参数':<10} {'困惑度':<10} {'Top-1':<10} {'Top-5':<10} {'评分':<10} {'等级'}"
    )
    print("-" * 80)

    for model_size, result in results.items():
        if result is None:
            continue

        config = result.get("config", {})
        params = config.get("hidden_dim", 0) * (
            config.get("num_layers", 0) * 4 + 2
        ) + result.get("config", {}).get("vocab_size", 8000) * config.get(
            "hidden_dim", 0
        )

        perplexity = result.get("perplexity", 0)
        accuracy = result.get("accuracy", {})
        top1 = accuracy.get("top_1", 0) * 100
        top5 = accuracy.get("top_5", 0) * 100
        score = result.get("score", 0)
        grade = result.get("grade", "-")

        print(
            f"{model_size:<10} "
            f"{params / 1e6:>5.1f}M    "
            f"{perplexity:>8.2f}  "
            f"{top1:>7.2f}%  "
            f"{top5:>7.2f}%  "
            f"{score:>7.1f}   "
            f"{grade}"
        )

    print("=" * 80)

    # 找出最佳模型
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1].get("score", 0))
        print(
            f"\n🏆 最佳模型: {best_model[0]} (评分: {best_model[1].get('score', 0):.1f})"
        )

        # 推荐建议
        print("\n推荐建议:")

        # 找出性价比最高的
        small_models = {
            k: v for k, v in valid_results.items() if k in ["tiny", "small", "medium"]
        }
        if small_models:
            best_small = max(small_models.items(), key=lambda x: x[1].get("score", 0))
            print(
                f"  手机部署: {best_small[0]} (体积小、速度快、精度{best_small[1].get('score', 0):.0f}分)"
            )

        # 找出精度最高的
        large_models = {
            k: v for k, v in valid_results.items() if k in ["base", "large"]
        }
        if large_models:
            best_large = max(large_models.items(), key=lambda x: x[1].get("score", 0))
            print(
                f"  PC/服务器: {best_large[0]} (精度{best_large[1].get('score', 0):.0f}分)"
            )


def main():
    parser = argparse.ArgumentParser(description="对比评估不同模型尺寸")
    parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["tiny", "small", "medium", "base"],
        help="要评估的模型尺寸",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/val.bin",
        help="验证数据路径",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/vocab.json",
        help="词表路径",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="每个模型最大评估样本数",
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="列出所有可用的模型尺寸",
    )

    args = parser.parse_args()

    if args.list_sizes:
        list_model_sizes()
        return

    print("\n" + "=" * 80)
    print("批量模型评估")
    print("=" * 80)
    print(f"模型尺寸: {', '.join(args.model_sizes)}")
    print(f"验证数据: {args.data_path}")
    print(f"样本数量: {args.max_samples}")
    print("=" * 80)

    # 评估每个模型
    results = {}
    for model_size in args.model_sizes:
        result = evaluate_model(
            model_size, args.data_path, args.vocab_path, args.max_samples
        )
        results[model_size] = result

    # 对比结果
    compare_models(results)

    # 保存对比结果
    output_path = Path("output/model_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n对比结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
