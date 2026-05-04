#!/usr/bin/env python3
"""
分析 BPE 词表在语料中的分布与频率，用 label.txt 过滤不常用的字/词，剪枝到目标大小。

流程:
  1. 训练 BPE tokenizer (较大词表，如 12000) 或加载已有
  2. 扫描语料，统计每个 token 的出现频率
  3. 分类: 特殊词 / 单字(检查是否在 label.txt) / 多字组合
  4. 保留策略:
     - 特殊词: 全部保留
     - 单字: 在 label.txt 中 → 保留；不在 → 丢弃
     - 多字组合: 按频率排序，保留 top-N (填满剩余词表预算)
  5. 用剪枝后的词表重建 WordLevel tokenizer
  6. 重新 tokenize 语料并保存
"""

import argparse
import gc
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

CJK_PUNCT = set(
    "，。！？、；：""''（）【】《》——…··～·"
)


def is_pure_punct(text: str) -> bool:
    return all(c in CJK_PUNCT for c in text)


def stream_text(file_path: str) -> Iterator[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_chars_from_label(label_path: str) -> set[str]:
    """从 label.txt 加载所有字符，并补充常见中文全角标点（label.txt 只包含了 ASCII 对标）"""
    chars = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            char = line.split("\t")[0]
            if char:
                chars.add(char)

    # label.txt 有 ASCII 标点但缺少中文全角标点，补充进来
    cjk_punct = set(
        "，。！？、；：""''（）【】《》——…··～＋－＝＿｛｝［］＠＃＄％＾＆＊＼"
    )
    chars.update(cjk_punct)
    return chars


def count_lines_fast(path: str) -> int:
    """快速统计文件行数（分块读取避免大文件 OOM）"""
    buf_size = 1024 * 1024
    lines = 0
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            lines += chunk.count(b"\n")
    return lines


def train_bpe_tokenizer(
    corpus_path: str,
    output_path: str,
    vocab_size: int,
    max_lines: int = 1000000,
    char_vocab_path: str = None,
):
    """训练 BPE tokenizer（采样 + label 字符过滤）"""
    print(f"\n[1/4] 训练 BPE 词表 (初始大小: {vocab_size})...")

    total = count_lines_fast(corpus_path)

    label_chars = set()
    if char_vocab_path:
        label_chars = load_chars_from_label(char_vocab_path)
        print(f"  Label 字符数: {len(label_chars)}")

    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r"([，。！？、；：])", behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=list("，。！？、0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    )

    def line_iter():
        with open(corpus_path, "r", encoding="utf-8") as f:
            if total <= max_lines:
                for line in f:
                    if label_chars:
                        line = "".join(c for c in line if c in label_chars or c.isspace())
                    yield line.strip()
            else:
                sample_rate = max_lines / total
                print(f"  语料共 {total:,} 行，随机采样 {max_lines:,} 行 (采样率 {sample_rate:.1%})")
                for line in f:
                    if random.random() < sample_rate:
                        if label_chars:
                            line = "".join(c for c in line if c in label_chars or c.isspace())
                        yield line.strip()

    tokenizer.train_from_iterator(line_iter(), trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(output_path))
    print(f"BPE 词表已保存: {output_path} (大小: {tokenizer.get_vocab_size()})")
    return tokenizer


def analyze_frequency(
    tokenizer: Tokenizer, corpus_path: str, sample_ratio: float = 0.3
) -> Counter:
    """扫描语料采样，统计每个 token ID 出现的频次"""
    print(f"\n[2/4] 分析词表频率 (采样 {sample_ratio*100:.0f}%)...")

    total_lines = sum(1 for _ in open(corpus_path, "r", encoding="utf-8"))
    sample_lines = int(total_lines * sample_ratio)
    print(f"总行数: {total_lines:,}, 采样行数: {sample_lines:,}")

    freq = Counter()
    lines_processed = 0

    for line in tqdm(stream_text(corpus_path), total=total_lines, desc="频率统计"):
        ids = tokenizer.encode(line).ids
        freq.update(ids)
        lines_processed += 1
        if lines_processed >= sample_lines:
            break

    print(f"完成统计，共出现 {len(freq)} 个不同 token")
    return freq


def classify_and_prune(
    tokenizer: Tokenizer,
    freq: Counter,
    label_chars: set[str],
    target_vocab_size: int,
) -> dict[str, int]:
    """
    分类 + 剪枝:
      - 特殊词: 全保留
      - 单字 in label.txt: 保留（ByteLevel BPE 需 decode 获取实际字符）
      - 单字 NOT in label.txt: 丢弃（由 UNK 兜底）
      - 多字: 按频率从高到低排序，保留 top-N

    ByteLevel BPE 的 token 内部是 UTF-8 字节序列，
    需用 tokenizer.decode([tid]) 解码才能得到实际中文字符。
    """
    print(f"\n[3/4] 分类并剪枝词表 (目标: {target_vocab_size})...")

    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    special_ids = {tokenizer.token_to_id(s) for s in special_tokens}

    # 收集 token 信息: 用 decode 还原实际文本
    single_in_label = {}      # text -> (best_id, total_freq) 去重
    single_not_in_label = {}  # text -> (best_id, total_freq)
    multi_tokens = {}         # text -> (best_id, total_freq)
    punct_filtered = 0

    for tid in range(tokenizer.get_vocab_size()):
        if tid in special_ids:
            continue
        count = freq.get(tid, 0)
        decoded = tokenizer.decode([tid]).strip()

        if len(decoded) == 1:
            dst = single_in_label if decoded in label_chars else single_not_in_label
            if decoded not in dst or count > dst[decoded][1]:
                dst[decoded] = (tid, count)
        elif len(decoded) > 1:
            if is_pure_punct(decoded):
                punct_filtered += 1
                continue
            if decoded not in multi_tokens or count > multi_tokens[decoded][1]:
                multi_tokens[decoded] = (tid, count)

    single_in_label_list = [(t, c) for t, (_, c) in single_in_label.items()]
    single_not_in_label_list = [(t, c) for t, (_, c) in single_not_in_label.items()]
    multi_tokens_list = [(t, c) for t, (_, c) in multi_tokens.items()]

    single_in_label_list.sort(key=lambda x: x[1], reverse=True)
    single_not_in_label_list.sort(key=lambda x: x[1], reverse=True)
    multi_tokens_list.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  特殊词:       4")
    print(f"  单字(in label): {len(single_in_label_list):,}")
    print(f"  单字(not label): {len(single_not_in_label_list):,} ← 将被丢弃")
    print(f"  多字组合:     {len(multi_tokens_list):,}")
    if punct_filtered > 0:
        print(f"  纯标点丢弃:   {punct_filtered}")

    dumped_freq = sum(c for _, c in single_not_in_label_list)
    total_freq = sum(freq.values())
    if total_freq > 0:
        print(f"  丢弃单字覆盖率: {dumped_freq/total_freq*100:.4f}%")

    # 构建新词表
    new_vocab = {}

    # 1. 特殊词
    for i, tok in enumerate(special_tokens):
        new_vocab[tok] = i

    next_id = len(special_tokens)

    # 2. label.txt 中的单字（全部保留）
    #    再加回 BPE 没见到的 label 字（它们不在采样中出现，但保证手写/罕见字也能编码）
    added_fallback = 0
    for text, _ in single_in_label_list:
        new_vocab[text] = next_id
        next_id += 1
    # 如果 label.txt 中有字在 BPE 的 30% 采样里没出现过，也加进来
    for char in label_chars:
        if char not in new_vocab and len(char) == 1:
            new_vocab[char] = next_id
            next_id += 1
            added_fallback += 1
    if added_fallback > 0:
        print(f"    补充未出现的 label 单字: {added_fallback}")

    # 3. 高频率多字组合
    remaining = target_vocab_size - len(new_vocab)
    print(f"\n  词表剩余槽位: {remaining}")
    kept_multi = 0
    for text, count in multi_tokens_list:
        if remaining <= 0:
            break
        new_vocab[text] = next_id
        next_id += 1
        remaining -= 1
        kept_multi += 1

    final_size = len(new_vocab)
    base_size = len(special_tokens) + len(single_in_label_list)

    if final_size > target_vocab_size:
        print(
            f"\n  ⚠ 目标 {target_vocab_size} 小于基础字符数 ({base_size})。"
            f"最终词表 ({final_size}) 将超出目标。建议增大 target-vocab-size ≥ {base_size}"
        )

    print(f"\n  剪枝结果:")
    print(f"    特殊词:       {len(special_tokens)}")
    print(f"    保留单字:     {len(single_in_label_list)}")
    print(f"    保留多字:     {kept_multi}")
    print(f"    丢弃单字:     {len(single_not_in_label_list)}")
    print(f"    丢弃多字:     {len(multi_tokens_list) - kept_multi}")
    print(f"    最终词表:     {final_size}")

    return new_vocab


def add_ascii_chars(vocab: dict[str, int]) -> dict[str, int]:
    """添加所有可打印 ASCII 字符到词表（单字符编码，避免整词查表变 UNK）"""
    import string
    next_id = max(vocab.values()) + 1
    added = 0
    for c in string.printable:
        if c not in vocab and c not in ("\t", "\n", "\r", "\x0b", "\x0c"):
            vocab[c] = next_id
            next_id += 1
            added += 1
    if added:
        print(f"    补充 ASCII 字符: {added}")
    return vocab


def build_pruned_vocab(vocab: dict[str, int], output_path: str) -> dict[str, int]:
    """
    将剪枝后的词表保存为轻量 JSON 格式（非 tokenizers 格式）。

    训练时使用 SimpleTokenizer 加载此文件，
    只做字符级拆分 + vocab 查表，准确且零依赖。
    """
    print(f"\n[4/4] 保存剪枝后词表 (JSON)...")

    # 转成 {token: id} 格式
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"词表已保存: {output_path} ({len(vocab)} tokens)")

    # 验证编码
    multi_index, max_multi_len = build_multi_index(vocab)
    encode_tokens = simple_encode("你好世界，hello world!", vocab, multi_index, max_multi_len)
    print(f"  encode('你好世界，hello world!') -> tokens={encode_tokens[:10]}... ✓")

    return vocab


def build_multi_index(vocab: dict) -> tuple:
    """构建多字 token 索引：按首字母分组，用于最长匹配"""
    multi_tokens = {}
    max_len = 0
    for token in vocab:
        if len(token) > 1 and token not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]"):
            first = token[0]
            multi_tokens.setdefault(first, []).append(token)
            if len(token) > max_len:
                max_len = len(token)

    for first in multi_tokens:
        multi_tokens[first].sort(key=len, reverse=True)

    return multi_tokens, max_len


def simple_encode(text: str, vocab: dict[str, int], multi_index: dict = None, max_multi_len: int = 0) -> list[int]:
    """最长匹配优先编码：先尝试多字 token，不在词表的字符直接跳过"""
    ids = [vocab["[BOS]"]]
    if multi_index is None:
        multi_index, max_multi_len = build_multi_index(vocab)

    i, n = 0, len(text)
    while i < n:
        c = text[i]
        matched = False
        if c in multi_index and n - i >= 2:
            for token in multi_index[c]:
                tlen = len(token)
                if i + tlen <= n and text[i:i + tlen] == token:
                    ids.append(vocab[token])
                    i += tlen
                    matched = True
                    break

        if not matched:
            c = text[i]
            if c in vocab:
                ids.append(vocab[c])
            i += 1

    ids.append(vocab["[EOS]"])
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="分析 BPE 词表分布，用 label.txt 剪枝到目标大小"
    )
    parser.add_argument(
        "--corpus", default="data/cleaned/all_cleaned.txt", help="语料文件路径"
    )
    parser.add_argument(
        "--label", default="label.txt", help="常用汉字表路径 (label.txt)"
    )
    parser.add_argument(
        "--output-dir", default="data", help="输出目录"
    )
    parser.add_argument(
        "--initial-vocab-size", type=int, default=20000,
        help="初始 BPE 词表大小 (默认 12000)"
    )
    parser.add_argument(
        "--target-vocab-size", type=int, default=8000,
        help="剪枝目标大小 (默认 5000)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="验证集比例"
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=0.3,
        help="频率分析采样比例 (默认 0.3)"
    )
    parser.add_argument(
        "--max-bpe-lines", type=int, default=500000,
        help="BPE 训练最大行数 (默认 500K, 防 OOM)"
    )
    parser.add_argument(
        "--skip-bpe", action="store_true",
        help="跳过 BPE 训练，直接使用已有的 tokenizer.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只分析+打印统计信息，不执行 tokenize"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpe_tokenizer_path = output_dir / "tokenizer_bpe.json"
    pruned_tokenizer_path = output_dir / "vocab.json"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    # Step 1: Train or load BPE tokenizer
    if args.skip_bpe and bpe_tokenizer_path.exists():
        print(f"加载已有 BPE tokenizer: {bpe_tokenizer_path}")
        bpe_tokenizer = Tokenizer.from_file(str(bpe_tokenizer_path))
    else:
        bpe_tokenizer = train_bpe_tokenizer(
            args.corpus,
            str(bpe_tokenizer_path),
            args.initial_vocab_size,
            max_lines=args.max_bpe_lines,
            char_vocab_path=args.label,
        )

    # Step 2: Analyze frequency
    label_chars = load_chars_from_label(args.label)
    print(f"\nLabel.txt 字符数: {len(label_chars)}")

    freq = analyze_frequency(bpe_tokenizer, args.corpus, args.sample_ratio)

    # Create frequency report
    print(f"\n  频率 Top-20 token (已解码):")
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    for i, (tid, cnt) in enumerate(freq.most_common(20)):
        decoded = bpe_tokenizer.decode([tid]).strip()
        # Truncate very long decoded text for display
        display = decoded if len(decoded) <= 20 else decoded[:17] + "..."
        flag = " [SPECIAL]" if decoded in special_tokens else ""
        print(f"    {i+1:2d}. ID={tid:5d} freq={cnt:>10,}  '{display}'{flag}")

    # Step 3: Classify and prune
    new_vocab = classify_and_prune(
        bpe_tokenizer, freq, label_chars, args.target_vocab_size
    )

    # Step 4: Save pruned vocabulary as simple JSON
    build_pruned_vocab(new_vocab, str(pruned_tokenizer_path))

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"Dry-run 完成! 跳过 tokenize 步骤。")
        print(f"词表:     {pruned_tokenizer_path} ({len(new_vocab)})")
        print(f"运行时不带 --dry-run 来执行完整流程 (tokenize 数据)")
        print(f"{'='*60}")
        return

    # Step 5: Re-tokenize data with simple char-level encoding
    print(f"\nTokenize 数据...")
    total_lines = sum(1 for _ in open(args.corpus, "r", encoding="utf-8"))
    split_at = int(total_lines * args.val_ratio)

    train_count = 0
    val_count = 0
    line_idx = 0

    with (
        open(str(train_path) + ".tmp", "wb") as train_f,
        open(str(val_path) + ".tmp", "wb") as val_f,
    ):
        multi_index, max_multi_len = build_multi_index(new_vocab)
        for line in tqdm(stream_text(args.corpus), total=total_lines, desc="Tokenize"):
            ids = simple_encode(line, new_vocab, multi_index, max_multi_len)

            arr = np.array(ids, dtype=np.uint16)
            if line_idx < total_lines - split_at:
                train_f.write(arr.tobytes())
                train_count += len(ids)
            else:
                val_f.write(arr.tobytes())
                val_count += len(ids)

            line_idx += 1
            if line_idx % 100000 == 0:
                gc.collect()

    shutil.move(str(train_path) + ".tmp", str(train_path))
    shutil.move(str(val_path) + ".tmp", str(val_path))

    print(f"训练集: {train_count:,} tokens")
    print(f"验证集: {val_count:,} tokens")

    # Save metadata
    meta = {
        "vocab_size": len(new_vocab),
        "initial_bpe_vocab_size": bpe_tokenizer.get_vocab_size(),
        "label_chars": len(label_chars),
        "train_tokens": train_count,
        "val_tokens": val_count,
        "source": args.corpus,
        "pruning_type": f"label_filter+top_multi",
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"词表:     {pruned_tokenizer_path} ({len(new_vocab)})")
    print(f"训练数据: {train_path} ({train_count:,} tokens)")
    print(f"验证数据: {val_path} ({val_count:,} tokens)")
    print(f"元数据:   {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
