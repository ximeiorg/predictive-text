"""模型评估脚本 - 全面评估模型性能"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict

from src.config import ModelConfig, list_model_sizes, MODEL_SIZES
from src.model.transformer import create_model
from src.data.dataset import WikipediaDataset


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, device, vocab_path):
        self.model = model
        self.device = device
        self.model.eval()

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def compute_perplexity(self, dataloader):
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="计算困惑度"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]

                # 计算非 pad token 数量
                non_pad_tokens = (labels != 0).sum().item()

                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = np.exp(avg_loss)

        return perplexity, avg_loss

    def compute_topk_accuracy(self, dataloader, k_list=[1, 3, 5, 10]):
        """计算 Top-K 准确率"""
        correct = defaultdict(int)
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="计算准确率"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                logits = outputs["logits"]  # [batch, seq_len, vocab_size]

                # 只评估最后一个位置
                last_logits = logits[:, -1, :]  # [batch, vocab_size]
                last_labels = labels[:, -1]  # [batch]

                # 只统计非 pad token
                mask = last_labels != 0
                if mask.sum() == 0:
                    continue

                last_logits = last_logits[mask]
                last_labels = last_labels[mask]

                for k in k_list:
                    _, topk_indices = torch.topk(last_logits, k, dim=1)
                    # topk_indices: [batch, k]

                    for i, label in enumerate(last_labels):
                        if label in topk_indices[i]:
                            correct[k] += 1

                total += len(last_labels)

        accuracy = {f"top_{k}": correct[k] / total if total > 0 else 0 for k in k_list}
        accuracy["total_samples"] = total

        return accuracy

    def compute_next_token_metrics(self, dataloader, num_samples=1000):
        """计算下一个词预测的详细指标"""
        all_ranks = []
        all_probs = []

        with torch.no_grad():
            sample_count = 0
            for batch in tqdm(dataloader, desc="计算详细指标"):
                if sample_count >= num_samples:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                logits = outputs["logits"]

                # 评估最后一个位置
                last_logits = logits[:, -1, :]
                last_labels = labels[:, -1]

                # 过滤 pad
                mask = last_labels != 0
                if mask.sum() == 0:
                    continue

                last_logits = last_logits[mask]
                last_labels = last_labels[mask]

                probs = F.softmax(last_logits, dim=1)

                for i, label in enumerate(last_labels):
                    prob = probs[i, label].item()
                    all_probs.append(prob)

                    # 计算正确词的排名
                    sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
                    rank = (sorted_indices == label).nonzero()
                    if len(rank) > 0:
                        all_ranks.append(rank[0].item() + 1)  # 排名从1开始

                sample_count += len(last_labels)

        metrics = {
            "mean_rank": np.mean(all_ranks) if all_ranks else float("inf"),
            "median_rank": np.median(all_ranks) if all_ranks else float("inf"),
            "mean_prob": np.mean(all_probs) if all_probs else 0,
            "median_prob": np.median(all_probs) if all_probs else 0,
            "num_samples": len(all_ranks),
        }

        # 计算排名分布
        rank_distribution = {
            "top_1": sum(1 for r in all_ranks if r == 1) / len(all_ranks)
            if all_ranks
            else 0,
            "top_3": sum(1 for r in all_ranks if r <= 3) / len(all_ranks)
            if all_ranks
            else 0,
            "top_5": sum(1 for r in all_ranks if r <= 5) / len(all_ranks)
            if all_ranks
            else 0,
            "top_10": sum(1 for r in all_ranks if r <= 10) / len(all_ranks)
            if all_ranks
            else 0,
        }
        metrics["rank_distribution"] = rank_distribution

        return metrics

    def evaluate_sample_predictions(self, test_cases, max_new_tokens=5):
        """评估示例预测"""
        results = []

        for text in test_cases:
            # 使用实际的分词器
            words = list(text)
            input_ids = [self.vocab.get(w, 3) for w in words]  # 3 is UNK

            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_k=5,
                    deterministic=False,
                )

            generated_ids = output_ids[0].tolist()[len(input_ids) :]
            generated_words = [
                self.id2word.get(i, "[UNK]")
                for i in generated_ids
                if i not in [0, 1, 2]
            ]

            results.append(
                {
                    "input": text,
                    "predictions": generated_words,
                }
            )

        return results


def load_model(checkpoint_path, device="auto"):
    """加载模型"""
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 兼容 Lightning checkpoint 和自定义 checkpoint 格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", ModelConfig())
    elif "state_dict" in checkpoint:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
        hp = checkpoint.get("hyper_parameters", {})
        config = ModelConfig.from_dict(hp.get("model_config", {}))
    else:
        raise KeyError("Unrecognized checkpoint format: no 'model_state_dict' or 'state_dict' found")

    model = create_model(config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, device, config


def create_test_cases():
    """创建测试用例"""
    return [
        # 日常对话
        "今天天气",
        "我们一起去",
        "这个事情",
        "我觉得",
        "时间过得",
        # 时间相关
        "明天早上",
        "昨天晚上",
        "下个星期",
        # 地点相关
        "我在北京",
        "去上海",
        # 情感表达
        "非常高兴",
        "有点难过",
        "特别开心",
        # 动作
        "正在吃饭",
        "准备睡觉",
        "开始工作",
        # 常用词组
        "无论如何",
        "总的来说",
        "因为所以",
    ]


def main():
    parser = argparse.ArgumentParser(description="评估模型性能")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=list(MODEL_SIZES.keys()),
        help="模型尺寸",
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
        "--batch-size",
        type=int,
        default=64,
        help="评估 batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="最大评估样本数",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="评估结果输出路径",
    )

    args = parser.parse_args()

    # 确定 checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = f"output/{args.model_size}/best_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        return 1

    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"output/{args.model_size}/evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    print(f"模型尺寸:     {args.model_size}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"验证数据:     {args.data_path}")
    print(f"最大样本数:   {args.max_samples}")
    print("=" * 60 + "\n")

    # 加载模型
    print("加载模型...")
    model, device, config = load_model(checkpoint_path)
    print(f"设备: {device}")
    print(f"模型配置: {config}")

    # 加载数据
    print("\n加载验证数据...")
    dataset = WikipediaDataset(
        args.data_path,
        args.vocab_path,
        max_seq_len=config.max_seq_len if hasattr(config, "max_seq_len") else 32,
    )

    # 限制样本数
    if len(dataset) > args.max_samples:
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        from torch.utils.data import Subset

        dataset = Subset(dataset, indices)

    from torch.utils.data import DataLoader as TorchDataLoader

    dataloader = TorchDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 创建评估器
    evaluator = ModelEvaluator(model, device, args.vocab_path)

    # 评估结果
    results = {
        "model_size": args.model_size,
        "checkpoint": checkpoint_path,
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "ffn_dim": config.ffn_dim,
            "vocab_size": config.vocab_size,
        },
        "num_samples": len(dataset),
    }

    # 1. 困惑度
    print("\n" + "=" * 60)
    print("1. 计算困惑度")
    print("=" * 60)
    perplexity, avg_loss = evaluator.compute_perplexity(dataloader)
    print(f"平均 Loss: {avg_loss:.4f}")
    print(f"困惑度: {perplexity:.2f}")
    results["perplexity"] = perplexity
    results["avg_loss"] = avg_loss

    # 2. Top-K 准确率
    print("\n" + "=" * 60)
    print("2. 计算 Top-K 准确率")
    print("=" * 60)
    accuracy = evaluator.compute_topk_accuracy(dataloader, k_list=[1, 3, 5, 10])
    for k, acc in accuracy.items():
        if k.startswith("top_"):
            print(f"Top-{k[4:]} 准确率: {acc * 100:.2f}%")
    results["accuracy"] = accuracy

    # 3. 详细指标
    print("\n" + "=" * 60)
    print("3. 计算详细指标")
    print("=" * 60)
    metrics = evaluator.compute_next_token_metrics(dataloader, num_samples=2000)
    print(f"平均排名: {metrics['mean_rank']:.2f}")
    print(f"中位数排名: {metrics['median_rank']:.1f}")
    print(f"平均概率: {metrics['mean_prob']:.4f}")
    print(f"\n排名分布:")
    for k, v in metrics["rank_distribution"].items():
        print(f"  {k}: {v * 100:.2f}%")
    results["detailed_metrics"] = metrics

    # 4. 示例预测
    print("\n" + "=" * 60)
    print("4. 示例预测")
    print("=" * 60)
    test_cases = create_test_cases()
    predictions = evaluator.evaluate_sample_predictions(test_cases, max_new_tokens=5)

    for pred in predictions[:10]:  # 只显示前10个
        print(f"输入: {pred['input']}")
        print(f"预测: {', '.join(pred['predictions'][:5])}")
        print()

    results["sample_predictions"] = predictions

    # 5. 性能评分
    print("\n" + "=" * 60)
    print("5. 性能评分")
    print("=" * 60)

    # 综合评分 (0-100)
    score = 0

    # 困惑度评分 (越低越好)
    if perplexity < 5:
        score += 40
    elif perplexity < 10:
        score += 30
    elif perplexity < 20:
        score += 20
    elif perplexity < 50:
        score += 10

    # 准确率评分
    score += accuracy["top_1"] * 30  # 0-30分
    score += accuracy["top_5"] * 20  # 0-20分
    score += accuracy["top_10"] * 10  # 0-10分

    print(f"综合评分: {score:.1f}/100")

    if score >= 80:
        grade = "优秀 ⭐⭐⭐⭐⭐"
    elif score >= 60:
        grade = "良好 ⭐⭐⭐⭐"
    elif score >= 40:
        grade = "中等 ⭐⭐⭐"
    elif score >= 20:
        grade = "一般 ⭐⭐"
    else:
        grade = "需改进 ⭐"

    print(f"等级: {grade}")

    results["score"] = score
    results["grade"] = grade

    # 保存结果
    print(f"\n保存评估结果到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    print(f"困惑度:       {perplexity:.2f}")
    print(f"Top-1 准确率:  {accuracy['top_1'] * 100:.2f}%")
    print(f"Top-5 准确率:  {accuracy['top_5'] * 100:.2f}%")
    print(f"平均排名:     {metrics['mean_rank']:.2f}")
    print(f"综合评分:     {score:.1f}/100 ({grade})")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
