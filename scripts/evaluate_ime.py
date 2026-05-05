#!/usr/bin/env python3
"""评估 IME 联想模型：top-k 准确率 + 示例预测"""

import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import WikipediaDataset, load_vocab
from src.model.transformer import DecoderTransformer
from src.config import ModelConfig


def load_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model_dict = state["model_state_dict"]
        config_data = state.get("config", {})
    else:
        model_dict = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
        config_data = state.get("hyper_parameters", {}).get("model_config", {})
    if isinstance(config_data, dict):
        cfg = ModelConfig(**{k: v for k, v in config_data.items() if k in ModelConfig.__dataclass_fields__})
    else:
        cfg = config_data
    model = DecoderTransformer(cfg)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate(model, dataloader, device, max_samples=5000):
    model.eval()
    total, correct_1, correct_3, correct_5, reciprocal_ranks = 0, 0, 0, 0, []
    losses = []

    for batch in tqdm(dataloader, desc="评估"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        losses.append(outputs["loss"].item())

        logits = outputs["logits"]
        last_logits = logits[:, -1, :]
        last_labels = labels[:, -1]

        mask = last_labels != 0
        if mask.sum() == 0:
            continue

        last_logits = last_logits[mask]
        last_labels = last_labels[mask]

        probs = F.softmax(last_logits, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)

        for i, label in enumerate(last_labels):
            rank = (sorted_ids[i] == label).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                r = rank[0].item() + 1
                reciprocal_ranks.append(1.0 / r)
                if r == 1:
                    correct_1 += 1
                if r <= 3:
                    correct_3 += 1
                if r <= 5:
                    correct_5 += 1

        total += len(last_labels)
        if total >= max_samples:
            break

    n = total
    return {
        "samples": n,
        "loss": float(np.mean(losses)),
        "top1_acc": correct_1 / n if n else 0,
        "top3_acc": correct_3 / n if n else 0,
        "top5_acc": correct_5 / n if n else 0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0,
    }


@torch.no_grad()
def sample_predictions(model, tokenizer, device, test_cases, top_k=5):
    model.eval()
    results = []
    for text in test_cases:
        ids = tokenizer.encode(text).ids
        if len(ids) > 1:
            ids = ids[:-1]
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_tensor)["logits"][:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k * 3, dim=-1)

        candidates = []
        skip_ids = {tokenizer.vocab.get(k, -1) for k in ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] if hasattr(tokenizer, "vocab")}
        if not skip_ids:
            skip_ids = {0, 1, 2, 3}
        for i in range(top_ids.size(1)):
            tid = top_ids[0, i].item()
            if tid in skip_ids:
                continue
            decoded = tokenizer.decode([tid])
            if decoded.strip():
                candidates.append(decoded.strip())
            if len(candidates) >= top_k:
                break

        results.append({"input": text, "candidates": candidates})
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估 IME 联想模型")
    parser.add_argument("--checkpoint", default=None, help="模型 checkpoint 路径")
    parser.add_argument("--model-size", default="small", help="模型尺寸")
    parser.add_argument("--data-path", default="data/val.bin", help="验证数据")
    parser.add_argument("--vocab-path", default="data/vocab.json", help="词表")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = f"output/{args.model_size}/best_model.pt"

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint 不存在: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"设备: {device}")
    print(f"检查点: {args.checkpoint}")

    model, config = load_model(args.checkpoint, device)
    print(f"模型参数: {model.count_parameters():,}")
    print(f"词表大小: {config.vocab_size}")

    dataset = WikipediaDataset(args.data_path, args.vocab_path, max_seq_len=config.max_seq_len)
    tokenizer = load_vocab(args.vocab_path)
    actual_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else tokenizer.get_vocab_size()
    print(f"实际词表: {actual_vocab_size}")

    indices = np.random.choice(len(dataset), min(args.max_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size)

    print("\n评估 Top-K 准确率...")
    metrics = evaluate(model, loader, device, args.max_samples)
    print(f"\n验证集 ({metrics['samples']} 样本):")
    print(f"  Loss:       {metrics['loss']:.4f}")
    print(f"  Top-1:      {metrics['top1_acc']*100:.2f}%")
    print(f"  Top-3:      {metrics['top3_acc']*100:.2f}%")
    print(f"  Top-5:      {metrics['top5_acc']*100:.2f}%")
    print(f"  MRR:        {metrics['mrr']:.4f}")

    if metrics["top3_acc"] >= 0.30:
        print("\n结论: 达到可用标准 (Top-3 >= 30%)")
    elif metrics["top3_acc"] >= 0.20:
        print("\n结论: 基本可用 (Top-3 >= 20%)")
    else:
        print("\n结论: 未达到可用标准 (Top-3 < 20%)")

    test_cases = [
        "今天天气", "我们一起去", "我觉得", "明天早上",
        "我在北京", "正在吃饭", "无论如何", "你",
        "我", "好", "是", "不",
    ]
    predictions = sample_predictions(model, tokenizer, device, test_cases, top_k=5)

    print("\n联想预测示例:")
    for p in predictions:
        print(f"  \"{p['input']}\" -> {' | '.join(p['candidates'][:5])}")

    out_path = Path(f"output/{args.model_size}/ime_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": predictions}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
