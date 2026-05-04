#!/usr/bin/env python3
"""从已清洗的文本文件生成训练数据。支持 --char-vocab 使用 label.txt 做 Character-Seeded BPE"""

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


def stream_text(file_path: str) -> Iterator[str]:
    """流式读取文本，避免内存泄漏"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_chars_from_label(label_path: str) -> list[str]:
    """从 label.txt 加载字符列表（跳过 ID，只取 char 列）"""
    chars = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            char = line.split("\t")[0]
            if char:
                chars.append(char)
    return chars


def train_char_seeded_tokenizer(
    text_path: str, vocab_size: int, output_path: str, char_vocab_path: str
):
    """
    用常用汉字种子 + BPE 训练 tokenizer。
    label.txt 的字符作为 initial_alphabet 预置，BPE 只补充最常用多字组合。
    """
    chars = load_chars_from_label(char_vocab_path)
    print(f"\n训练字符种子 BPE 词表 (目标: {vocab_size})")
    print(f"种子字符: {len(chars)} 个 (来自 {char_vocab_path})")

    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    base_vocab = len(special_tokens) + len(chars)
    bpe_budget = vocab_size - base_vocab
    print(f"基础: {base_vocab} (特殊词: 4 + 字符: {len(chars)})")
    print(f"BPE 多字补充预算: {bpe_budget}")

    if bpe_budget < 50:
        print(f"WARNING: BPE 预算不足 ({bpe_budget})，建议增大 vocab_size")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r"([，。！？、])", behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=chars,
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

    vocab = tokenizer.get_vocab()
    multi_breakdown(vocab, special_tokens)
    return tokenizer


def train_tokenizer(text_path: str, vocab_size: int, output_path: str):
    """训练标准 BPE tokenizer (原始逻辑)"""
    print(f"\n训练词表 (目标大小: {vocab_size})...")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r"([，。！？、])", behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        min_frequency=1,
        show_progress=True,
        initial_alphabet=list(
            "，。！？、0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
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


def multi_breakdown(vocab: dict, special_tokens: list[str]):
    """打印词表组成分析"""
    multi_char = sum(
        1
        for w in vocab
        if len(w.replace("Ġ", "")) > 1 and w not in special_tokens
    )
    single_char = sum(
        1
        for w in vocab
        if len(w.replace("Ġ", "")) == 1 and w not in special_tokens
    )
    print(
        f"词表组成: {len(vocab)} 总, "
        f"{single_char} 单字, "
        f"{multi_char} 多字 ({multi_char / len(vocab) * 100:.1f}%)"
    )


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
    parser = argparse.ArgumentParser(description="从清洗后的数据准备训练数据")
    parser.add_argument("--cleaned-file", default="data/cleaned/all_cleaned.txt", help="清洗后的文本文件路径")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--vocab-size", type=int, default=5000, help="词表大小")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--char-vocab", default=None, help="常用汉字表 (如 label.txt)，启用字符种子 BPE")
    args = parser.parse_args()

    cleaned_path = Path(args.cleaned_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists():
        print(f"错误: 找不到文件 {cleaned_path}")
        return

    print(f"处理清洗后的数据: {cleaned_path}")
    print(f"文件大小: {cleaned_path.stat().st_size / (1024*1024):.1f} MB")

    vocab_path = output_dir / "vocab.json"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.json"

    # 训练tokenizer (字符种子 或 标准 BPE)
    if args.char_vocab:
        char_path = Path(args.char_vocab)
        if not char_path.exists():
            print(f"错误: 找不到字符文件 {char_path}")
            return
        tokenizer = train_char_seeded_tokenizer(
            str(cleaned_path), args.vocab_size, str(vocab_path), args.char_vocab
        )
    else:
        tokenizer = train_tokenizer(str(cleaned_path), args.vocab_size, str(vocab_path))

    vocab_size = tokenizer.get_vocab_size()

    del tokenizer
    gc.collect()

    # 重新加载tokenizer并tokenize数据
    tokenizer = Tokenizer.from_file(str(vocab_path))
    train_count, val_count = tokenize_and_save(
        str(cleaned_path), tokenizer, str(train_path), str(val_path), args.val_ratio
    )

    # 保存元数据
    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "source": str(cleaned_path),
        "char_vocab": args.char_vocab,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n完成!")
    print(f"词表: {vocab_path}")
    print(f"训练数据: {train_path}")
    print(f"验证数据: {val_path}")
    print(f"元数据: {meta_path}")


if __name__ == "__main__":
    main()
