#!/usr/bin/env python3
"""从 data 目录的 JSONL 文件提取中文内容并进行预处理"""

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


def extract_text_from_jsonl(jsonl_paths: list[str], output_path: str):
    """从 JSONL 文件提取中文文本"""
    print("从 JSONL 文件提取文本...")

    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for jsonl_path in jsonl_paths:
            print(f"处理: {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as in_f:
                for line in tqdm(in_f, desc="提取"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    texts = []

                    if "question" in data and "answer" in data:
                        texts.append(data["question"])
                        texts.append(data["answer"])
                    elif "input" in data and "content" in data:
                        texts.append(data["input"])
                        texts.append(data["content"])

                    for text in texts:
                        if text and text.strip():
                            out_f.write(text.strip() + "\n")
                            total_lines += 1

                    if total_lines % 10000 == 0:
                        gc.collect()

    print(f"提取了 {total_lines:,} 行文本")
    return total_lines


def stream_text(file_path: str) -> Iterator[str]:
    """流式读取文本，避免内存泄漏"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def train_tokenizer(text_path: str, vocab_size: int, output_path: str):
    """训练 BPE tokenizer"""
    print(f"\n训练词表 (目标大小: {vocab_size})...")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        show_progress=True,
    )

    tokenizer.train([text_path], trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.save(str(output_path))
    print(f"词表大小: {tokenizer.get_vocab_size()}")

    return tokenizer


def tokenize_and_save(
    text_path: str,
    tokenizer: Tokenizer,
    train_output: str,
    val_output: str,
    val_ratio: float = 0.05,
):
    """Tokenize 并保存为二进制文件"""
    print("\nTokenizing 数据...")

    print("统计非空行数...")
    total_non_empty = 0
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_non_empty += 1

    split_at = int(total_non_empty * (1 - val_ratio))
    print(f"总非空行数: {total_non_empty:,}, 分割点: {split_at:,}")
    print(f"训练集行数: {split_at:,}, 验证集行数: {total_non_empty - split_at:,}")

    train_count = 0
    val_count = 0
    line_idx = 0

    with (
        open(train_output + ".tmp", "wb") as train_f,
        open(val_output + ".tmp", "wb") as val_f,
    ):
        for line in tqdm(
            stream_text(text_path), total=total_non_empty, desc="Tokenizing"
        ):
            ids = tokenizer.encode(line).ids

            arr = np.array(ids, dtype=np.uint16)
            if line_idx < split_at:
                train_f.write(arr.tobytes())
                train_count += len(ids)
            else:
                val_f.write(arr.tobytes())
                val_count += len(ids)

            line_idx += 1

            if line_idx % 100000 == 0:
                print(
                    f"进度: {line_idx:,}/{total_non_empty:,}, 训练tokens: {train_count:,}, 验证tokens: {val_count:,}"
                )

            if line_idx % 10000 == 0:
                gc.collect()

    shutil.move(train_output + ".tmp", train_output)
    shutil.move(val_output + ".tmp", val_output)

    print(f"训练集: {train_count:,} tokens")
    print(f"验证集: {val_count:,} tokens")

    return train_count, val_count


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument("--input-dir", default="data", help="输入数据目录")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--vocab-size", type=int, default=8192, help="词表大小")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"未找到 JSONL 文件: {input_dir}")
        return

    print(f"找到 {len(jsonl_files)} 个 JSONL 文件:")
    for f in jsonl_files:
        print(f"  - {f}")

    extracted_path = output_dir / "extracted.txt"
    vocab_path = output_dir / "vocab.json"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.json"

    extract_text_from_jsonl([str(f) for f in jsonl_files], str(extracted_path))
    text_path = extracted_path

    tokenizer = train_tokenizer(str(text_path), args.vocab_size, str(vocab_path))
    vocab_size = tokenizer.get_vocab_size()

    del tokenizer
    gc.collect()

    tokenizer = Tokenizer.from_file(str(vocab_path))
    train_count, val_count = tokenize_and_save(
        str(text_path), tokenizer, str(train_path), str(val_path), args.val_ratio
    )

    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_count,
        "val_tokens": val_count,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n完成!")
    print(f"词表: {vocab_path}")
    print(f"训练数据: {train_path}")
    print(f"验证数据: {val_path}")


if __name__ == "__main__":
    main()
