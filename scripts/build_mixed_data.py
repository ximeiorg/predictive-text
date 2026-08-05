#!/usr/bin/env python3
"""生成混合语料 all_mixed_cleaned.txt。

- 对新增语料做繁体->简体转换
- 剔除噪声数据集 (egret/chatterbot/sms)
- 过滤 qingyun 敏感词
- articles 全量使用
- 可选合并原有 all_cleaned.txt (百科/书面语)

输出: data/cleaned/all_mixed_cleaned.txt
"""

import argparse
import sys
from pathlib import Path

from opencc import OpenCC

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# 需要过滤的敏感词（qingyun 等含露骨内容）
SENSITIVE_WORDS = ["小骚货", "操你", "逼", "屌", "鸡巴", "傻逼", "卧槽你妈", "艹", "日你"]

# 噪声数据集，直接剔除
EXCLUDE = {"egret_qa", "chatterbot", "sms_zh"}

# 全部保留的口语/评论类
CONVERSATIONAL = [
    "weibo_stc",
    "douban_multiturn",
    "tieba",
    "bilibili",
    "subtitle_dgk",
    "toutiao_news",
    "chnsenti_htl",
    "bilibili_workcell",
    "qingyun",
]

ARTICLES = "articles"


def contains_sensitive(text: str) -> bool:
    return any(w in text for w in SENSITIVE_WORDS)


def load_lines(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


# 句末标点：用于把超长文本切成短句，保证窗口边界落在句子之间
SENTENCE_END = "。！？…"


def split_sentences(text: str, min_len: int = 4, max_len: int = 60):
    """按句末标点把超长文本切成短句，每句返回一行。

    - 保留句末标点在句中（如"你好。"）
    - 丢弃过短(< min_len)的碎片
    - 返回切分后的句子列表；若无法切分则整体返回
    """
    if len(text) <= max_len:
        return [text] if text else []

    sentences = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in SENTENCE_END:
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        if sentences and len(sentences[-1]) + len(tail) <= max_len:
            sentences[-1] += tail
        elif len(tail) >= min_len:
            sentences.append(tail)

    # 过滤过短碎片；若全为空，退回原串
    out = [s for s in sentences if len(s) >= min_len]
    return out if out else [text]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="data/cleaned/20260801")
    parser.add_argument("--output", default="data/cleaned/all_mixed_cleaned.txt")
    parser.add_argument(
        "--skip-articles", action="store_true",
        help="不使用 articles (默认全量使用)",
    )
    parser.add_argument(
        "--include-old", action="store_true",
        help="合并原有 all_cleaned.txt (百科书面语)",
    )
    parser.add_argument(
        "--dedup", action="store_true", help="去重(耗内存, 仅小数据用)"
    )
    parser.add_argument(
        "--split-sentences", action="store_true",
        help="按句末标点把超长文本切分为短句(推荐, 解决 articles 整篇无边界)",
    )
    args = parser.parse_args()

    src = Path(args.src_dir)
    cc = OpenCC("t2s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {}
    total_chars = 0
    seen = set()

    with open(out_path, "w", encoding="utf-8") as out:
        def emit(conv_text, stats_dict):
            """写出经过句子切分的文本，统计行数/字符数"""
            if args.split_sentences:
                pieces = split_sentences(conv_text)
            else:
                pieces = [conv_text] if conv_text else []
            for piece in pieces:
                if not piece:
                    continue
                if args.dedup:
                    if piece in seen:
                        continue
                    seen.add(piece)
                out.write(piece + "\n")
                stats_dict[0] += 1
                stats_dict[1] += len(piece)

        for name in CONVERSATIONAL:
            p = src / f"{name}_cleaned.txt"
            if not p.exists():
                print(f"[skip] 不存在: {p.name}")
                continue
            kept = chars = 0
            stat = [0, 0]
            for s in tqdm(load_lines(p), desc=name, unit="行"):
                if name == "qingyun" and contains_sensitive(s):
                    continue
                conv = cc.convert(s)
                emit(conv, stat)
            kept, chars = stat
            stats[name] = (kept, chars)
            total_chars += chars
            print(f"  {name}: {kept:,} 行, {chars:,} 字符")

        # articles 全量
        if not args.skip_articles:
            p = src / f"{ARTICLES}_cleaned.txt"
            if p.exists():
                stat = [0, 0]
                for s in load_lines(p):
                    conv = cc.convert(s)
                    emit(conv, stat)
                kept, chars = stat
                stats[ARTICLES] = (kept, chars)
                total_chars += chars
                print(f"  {ARTICLES}: {kept:,} 行, {chars:,} 字符")

        # 合并原有百科书面语
        if args.include_old:
            old = Path("data/cleaned/all_cleaned.txt")
            if old.exists():
                stat = [0, 0]
                for s in tqdm(load_lines(old), desc="all_cleaned(old)", unit="行"):
                    emit(s, stat)
                kept, chars = stat
                stats["all_cleaned(old)"] = (kept, chars)
                total_chars += chars
                print(f"  all_cleaned(old): {kept:,} 行, {chars:,} 字符")

    print("\n" + "=" * 60)
    print("混合语料统计:")
    for name, (k, c) in stats.items():
        print(f"  {name:<22} {k:>12,} 行  {c:>14,} 字符")
    print(f"  {'合计':<22} 总字符 {total_chars:>14,}")
    mb = out_path.stat().st_size / 1e6
    print(f"\n输出: {out_path}  ({mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
