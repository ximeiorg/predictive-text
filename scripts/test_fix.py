#!/usr/bin/env python3
"""验证修复效果 - 快速训练测试"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig
from src.model.transformer import create_model
from src.data.dataset import WikipediaDataset, DataLoader


def test_model():
    """快速测试模型和数据加载"""
    print("=" * 60)
    print("验证修复效果")
    print("=" * 60)

    vocab_path = Path("data/vocab.json")
    train_data_path = Path("data/train.bin")

    if not vocab_path.exists() or not train_data_path.exists():
        print("❌ 数据文件不存在，请先运行数据准备脚本")
        return

    import json

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    actual_vocab_size = len(vocab)
    print(f"✓ 词汇表大小: {actual_vocab_size}")

    config = ModelConfig(vocab_size=actual_vocab_size, max_seq_len=128)
    print(f"✓ 序列长度: {config.max_seq_len}")

    dataset = WikipediaDataset(
        str(train_data_path), str(vocab_path), max_seq_len=config.max_seq_len
    )
    print(f"✓ 数据集样本数: {len(dataset):,}")

    sample = dataset[0]
    print(f"✓ 输入形状: {sample['input_ids'].shape}")
    print(f"✓ 标签形状: {sample['labels'].shape}")

    model = create_model(config)
    print(f"✓ 模型参数量: {model.count_parameters():,}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"\n开始快速训练测试 (10步)...")
    losses = []

    for i, batch in enumerate(dataloader):
        if i >= 10:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        losses.append(loss.item())
        print(f"  Step {i + 1}: loss = {loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    print(f"\n✓ 平均loss: {avg_loss:.4f}")

    if avg_loss < 8:
        print("✓ Loss正常，修复成功！")
    else:
        print("⚠ Loss仍然较高，可能需要更多训练或数据问题")

    print("\n建议运行完整训练:")
    print("  python src/train.py --use-prepared-data --epochs 3")


if __name__ == "__main__":
    test_model()
