"""清洗 2026-08-01 新增语料，输出为连续中文明文（每行一条样本）。

格式统一：只保留中文+常用标点，去掉分词空格，与 data/cleaned/all_cleaned.txt 一致。
输出：data/cleaned/20260801/<name>_cleaned.txt + data/cleaned/20260801/all_new_cleaned.txt
"""

import json
import re
import sys
import csv
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


ALLOWED = re.compile(r"[\u4e00-\u9fff，。！？、；：""（）《》【】…—～·]+")
PUNC_ONLY = re.compile(r"^[，。！？、；：""（）《》【】…—～·\s]+$")


def clean_text(text):
    text = "".join(ALLOWED.findall(text or ""))
    text = re.sub(r"\n+", "", text)
    return text.strip()


def is_ok(text, min_len):
    return len(text) >= min_len and not PUNC_ONLY.match(text)


def writer(out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return open(out_path, "w", encoding="utf-8")


def run(handle, paths, out_path, min_len=8):
    stats = {"total": 0, "kept": 0, "chars": 0}
    with writer(out_path) as out:
        for p in paths:
            for line, text in handle(p):
                stats["total"] += 1
                c = clean_text(text)
                if is_ok(c, min_len):
                    out.write(c + "\n")
                    stats["kept"] += 1
                    stats["chars"] += len(c)
    print(f"  {out_path.name}: {stats['kept']}/{stats['total']:,} 行, {stats['chars']:,} 字符")
    return stats


# ---------- 各格式 handler：迭代 (源行, 提取文本) ----------

def h_txt(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line, line.strip()


def h_douban(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            text = "".join(parts[1:])  # 去掉首列 label
            yield line, text


def h_weibo(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line, line.replace(" ", "")


def h_tieba(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            yield line, "".join(parts)


def h_subtitle(p):
    # 格式: E 分隔场景, M word/word/ 表示台词(按 / 分词)
    buf = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "E":
                yield line, "".join(buf)
                buf = []
            elif line.startswith("M "):
                buf.append(line[2:].replace("/", ""))
        if buf:
            yield "", "".join(buf)


def h_csv(p, col):
    with open(p, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield "", row.get(col, "") if row else ""


def h_articles(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
                yield line, d.get("content", "")
            except json.JSONDecodeError:
                continue


def h_toutiao(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("_!_")
            if len(parts) >= 5:
                yield line, parts[3]  # 新闻标题
            else:
                yield line, ""


def h_chatterbot(p):
    # YAML: - - 问\n - 答
    conv = []
    in_conv = False
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("categories:"):
                in_conv = False
            elif s.startswith("conversations:"):
                in_conv = True
                conv = []
            elif in_conv and s.startswith("- - "):
                conv = [s[4:]]
            elif in_conv and s.startswith("- "):
                conv.append(s[2:])
            elif in_conv and not s:
                if len(conv) >= 2:
                    yield line, "".join(conv)
                conv = []
        if len(conv) >= 2:
            yield "", "".join(conv)


def h_egret(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("+++$+++")
            if len(parts) >= 4:
                yield line, parts[3]
            else:
                yield line, ""


def h_qingyun(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            yield line, "".join(parts)


def h_sms_xml(p):
    text_re = re.compile(r"<text>(.*?)</text>")
    with open(p, encoding="utf-8", errors="replace") as f:
        content = f.read()
    for m in text_re.finditer(content):
        yield "", m.group(1)


def main():
    base = Path("data/20260801")
    out_dir = Path("data/cleaned/20260801")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        # (名称, handler, [路径], min_len)
        ("douban_multiturn", h_douban,
         sorted((base / "raw_chat_corpus/douban-multiturn-100w").glob("*.txt")), 8),
        ("weibo_stc", h_weibo,
         sorted((base / "raw_chat_corpus/weibo-400w").glob("stc_weibo_train_*")), 8),
        ("tieba", h_tieba,
         [(base / "raw_chat_corpus/tieba-305w/tieba.dialogues")], 8),
        ("subtitle_dgk", h_subtitle,
         [(base / "raw_chat_corpus/subtitle-useless/dgk_shooter_min.conv")], 8),
        ("chatterbot", h_chatterbot,
         sorted((base / "raw_chat_corpus/chatterbot-1k/chinese").glob("*.yml")), 8),
        ("egret_qa", h_egret,
         [(base / "raw_chat_corpus/egret-qa-useless/raw/egret_wenda_lines_raw.txt")], 8),
        ("qingyun", h_qingyun,
         [(base / "raw_chat_corpus/qingyun-11w/12万对话语料青云库.csv")], 8),
        ("sms_zh", h_sms_xml,
         [(base / "raw_chat_corpus/sms-useless/smsCorpus_zh_xml_2015.03.09/smsCorpus_zh_2015.03.09.xml")], 8),
        ("bilibili", h_csv,
         [(base / "bilibili.csv")], 8),
        ("bilibili_workcell", h_csv,
         [(base / "bilibilib_gongzuoxibao.csv")], 8),
        ("chnsenti_htl", h_csv,
         [(base / "ChnSentiCorp_htl_all.csv")], 8),
        ("toutiao_news", h_toutiao,
         [(base / "toutiao_cat_data.txt")], 8),
        ("articles", h_articles,
         [(base / "articles.json")], 20),
    ]

    # csv 列名
    csv_cols = {
        "bilibili": "message",
        "bilibili_workcell": "content",
        "chnsenti_htl": "review",
    }

    all_stats = []
    merged_path = out_dir / "all_new_cleaned.txt"
    with writer(merged_path) as merged:
        for name, handle, paths, min_len in specs:
            print(f"处理 {name} ...")
            if name in csv_cols:
                def make_handle(col):
                    return lambda p: h_csv(p, col)
                handle = make_handle(csv_cols[name])
            stats = {"total": 0, "kept": 0, "chars": 0}
            with writer(out_dir / f"{name}_cleaned.txt") as out:
                for p in paths:
                    for line, text in handle(p):
                        stats["total"] += 1
                        c = clean_text(text)
                        if is_ok(c, min_len):
                            out.write(c + "\n")
                            merged.write(c + "\n")
                            stats["kept"] += 1
                            stats["chars"] += len(c)
            print(f"    {name}: {stats['kept']:,}/{stats['total']:,} 行, {stats['chars']:,} 字符")
            all_stats.append((name, stats))

    print("\n" + "=" * 60)
    print("清洗统计汇总:")
    total_k = total_c = 0
    for name, s in all_stats:
        total_k += s["kept"]; total_c += s["chars"]
        rate = s["kept"] / s["total"] * 100 if s["total"] else 0
        print(f"  {name:<22} {s['kept']:>10,} 行  {s['chars']:>12,} 字符  保留率 {rate:5.1f}%")
    print(f"  {'合计':<22} {total_k:>10,} 行  {total_c:>12,} 字符")
    print(f"\n合并文件: {merged_path}  ({merged_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
