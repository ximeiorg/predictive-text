#!/usr/bin/env python3
"""诊断训练问题：梯度分布、per-token loss、label smoothing 影响"""
import sys, torch, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import WikipediaDataset, load_vocab
from src.model.transformer import DecoderTransformer
from src.config import ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型（优先 checkpoint，否则新建随机模型）
ckpt_path = "output/small/best_model.pt"
if Path(ckpt_path).exists():
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" in state:
        hparams = state["hyper_parameters"].get("model_config", {})
        if isinstance(hparams, dict):
            cfg = ModelConfig(**{k: v for k, v in hparams.items() if k in ModelConfig.__dataclass_fields__})
    model = DecoderTransformer(cfg)
    model_dict = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(model_dict, strict=False)
    print(f"加载 checkpoint: {ckpt_path}")
    print(f"  参数: {model.count_parameters():,}")
    print(f"  词表: {cfg.vocab_size}")
else:
    cfg = ModelConfig(vocab_size=5000, hidden_dim=384, num_heads=4, num_layers=4, ffn_dim=1536)
    model = DecoderTransformer(cfg)
    print("新建随机模型")
model = model.to(device)
model.train()

# 加载一个 batch
dataset = WikipediaDataset("data/val.bin", "data/vocab.json", max_seq_len=128)
loader = torch.utils.data.DataLoader(dataset, batch_size=256)
batch = next(iter(loader))
input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

# 1. 不同 label_smoothing 下的 loss
print("=" * 60)
print("1. Label Smoothing 影响")
print("=" * 60)
for smoothing in [0.0, 0.05, 0.1, 0.15]:
    outputs = model(input_ids)
    logits = outputs["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        labels.view(-1),
        ignore_index=0,
        label_smoothing=smoothing,
    )
    print(f"  smoothing={smoothing:.2f}  loss={loss.item():.4f}  (ceil={math.log(cfg.vocab_size):.2f})")

# 2. Per-token 梯度 norm
print("\n" + "=" * 60)
print("2. 梯度分布")
print("=" * 60)
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]
loss.backward()

total_norm = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        norm = p.grad.norm().item()
        total_norm += norm ** 2
        if norm > 0.01:
            print(f"  {name:40s}  grad_norm={norm:.6f}")
print(f"\n  Total grad norm: {math.sqrt(total_norm):.4f}")
model.zero_grad()

# 3. Per-token 预测统计
print("\n" + "=" * 60)
print("3. Per-token 预测难度")
print("=" * 60)
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs["logits"]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs, preds = probs.max(dim=-1)
    correct = (preds == labels) | (labels == 0)

    # 按 token ID 统计
    token_acc = {}
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            tid = labels[i, j].item()
            if tid == 0:
                continue
            if tid not in token_acc:
                token_acc[tid] = {"correct": 0, "total": 0}
            token_acc[tid]["total"] += 1
            if correct[i, j]:
                token_acc[tid]["correct"] += 1

    # 高频错误 token
    sorted_tokens = sorted(
        [(tid, info["correct"]/info["total"], info["total"])
         for tid, info in token_acc.items() if info["total"] >= 5],
        key=lambda x: x[1]
    )

    tokenizer = load_vocab("data/vocab.json")
    print(f"  {'Token':10s} {'Acc':>8s} {'Count':>8s}  {'Decoded'}")
    for tid, acc, cnt in sorted_tokens[:15]:
        decoded = tokenizer.decode([tid]).strip()[:10] if hasattr(tokenizer, 'decode') else str(tid)
        print(f"  ID={tid:<5d} {acc:7.1%} {cnt:>8,}  '{decoded}'")

print(f"\n  总体准确率: {sum(correct).item()}/{correct.numel()} = {(sum(correct).item()/correct.numel()):.2%}")
