"""数据清洗脚本 - 只保留中文和常用标点"""

import json
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def clean_text(text):
    """清洗文本，只保留中文和常用标点"""
    # 定义允许的字符：中文 + 常用标点
    # 中文范围：\u4e00-\u9fff
    # 常用标点：，。！？、；：""''（）《》【】…—～·
    allowed_pattern = re.compile(
        r'[\u4e00-\u9fff，。！？、；：""' "（）《》【】…—～·\n]+"
    )

    # 提取所有匹配的字符
    cleaned = "".join(allowed_pattern.findall(text))

    # 移除多余换行
    cleaned = re.sub(r"\n+", "\n", cleaned).strip()

    # 移除 "unk" (不区分大小写)
    cleaned = re.sub(r"(?i)unk", "", cleaned)

    return cleaned


def is_only_punctuation(text):
    """检查文本是否只包含标点符号"""
    punctuation_pattern = re.compile(r'^[，。！？、；：""' r"（）《》【】…—～·\s]+$")
    return bool(punctuation_pattern.match(text))


def write_cleaned(text: str, out_f, min_len: int = 20):
    """清洗并写入，过滤短文本和纯标点"""
    cleaned = clean_text(text)
    if len(cleaned) >= min_len and not is_only_punctuation(cleaned):
        out_f.write(cleaned + "\n")
        return True
    return False


def process_standard_jsonl(input_path, output_path, content_key="content", input_key="input"):
    """处理标准 jsonl 文件，提取 content_key / input_key 后合并"""
    total_lines = 0
    cleaned_lines = 0

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        for line in tqdm(in_f, desc=f"清洗 {input_path.name}"):
            total_lines += 1

            try:
                data = json.loads(line)
                content = data.get(content_key, "")
                input_text = data.get(input_key, "")

                text = (
                    f"{input_text}\n{content}"
                    if input_text and content
                    else (content or input_text)
                )

                if text and write_cleaned(text, out_f):
                    cleaned_lines += 1

            except json.JSONDecodeError:
                continue

    print(f"  总行数: {total_lines:,}")
    print(f"  清洗后: {cleaned_lines:,}")
    print(f"  保留率: {cleaned_lines / total_lines * 100:.1f}%")


def process_qa_jsonl_file(input_path, output_path):
    """处理 QA 格式 jsonl（question + answer），复用标准流程"""
    return process_standard_jsonl(input_path, output_path, content_key="answer", input_key="question")


def process_chat_jsonl_file(input_path, output_path):
    """处理 Chat 格式 jsonl（messages 字段）"""
    total_lines = 0
    cleaned_lines = 0

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        for line in tqdm(in_f, desc=f"清洗 {input_path.name}"):
            total_lines += 1

            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                texts = []
                for msg in messages:
                    content = msg.get("content", "")
                    if content:
                        texts.append(content)
                if texts and write_cleaned("\n".join(texts), out_f):
                    cleaned_lines += 1

            except json.JSONDecodeError:
                continue

    print(f"  总行数: {total_lines:,}")
    print(f"  清洗后: {cleaned_lines:,}")
    print(f"  保留率: {cleaned_lines / total_lines * 100:.1f}%")


def process_csv_file(input_path, output_path, text_column="review"):
    """处理 CSV 文件，提取指定文本列"""
    import csv

    total_lines = 0
    cleaned_lines = 0

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        reader = csv.DictReader(in_f)
        for row in tqdm(reader, desc=f"清洗 {input_path.name}"):
            total_lines += 1
            text = row.get(text_column, "")
            if text and write_cleaned(text, out_f):
                cleaned_lines += 1

    print(f"  总行数: {total_lines:,}")
    print(f"  清洗后: {cleaned_lines:,}")
    print(f"  保留率: {cleaned_lines / total_lines * 100:.1f}%")


def process_txt_file(input_path, output_path):
    """处理纯文本文件（每行一条样本）"""
    total_lines = 0
    cleaned_lines = 0

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        for line in in_f:
            total_lines += 1
            text = line.strip()
            if text and write_cleaned(text, out_f):
                cleaned_lines += 1

    print(f"  总行数: {total_lines:,}")
    print(f"  清洗后: {cleaned_lines:,}")
    print(f"  保留率: {cleaned_lines / total_lines * 100:.1f}%")


def process_pclue_jsonl_file(input_path, output_path):
    """处理 pclue 格式（每行 JSON，提取 input 字段）"""
    total_lines = 0
    cleaned_lines = 0

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        for line in tqdm(in_f, desc=f"清洗 {input_path.name}"):
            total_lines += 1

            try:
                data = json.loads(line)
                text = data.get("input", "")
                if text and write_cleaned(text, out_f):
                    cleaned_lines += 1

            except json.JSONDecodeError:
                continue

    print(f"  总行数: {total_lines:,}")
    print(f"  清洗后: {cleaned_lines:,}")
    print(f"  保留率: {cleaned_lines / total_lines * 100:.1f}%")


def process_wikipedia_jsonl(input_path, output_path):
    """处理 Wikipedia JSONL 文件（每行一个JSON对象）"""
    total_lines = 0
    cleaned_lines = 0

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "w", encoding="utf-8") as out_f,
    ):
        for line in tqdm(in_f, desc=f"清洗 {input_path.name}"):
            total_lines += 1

            try:
                data = json.loads(line)
                # Wikipedia格式：提取text字段
                text = data.get("text", "")

                if text:
                    cleaned = clean_text(text)

                    # 只保留足够长的文本（至少20个字符）且不只有标点
                    if len(cleaned) >= 20 and not is_only_punctuation(cleaned):
                        out_f.write(cleaned + "\n")
                        cleaned_lines += 1

            except json.JSONDecodeError:
                continue

    print(f"  总行数: {total_lines:,}")
    print(f"  清洗后: {cleaned_lines:,}")
    print(f"  保留率: {cleaned_lines / total_lines * 100:.1f}%")


def process_json_file(input_path, output_path):
    """处理json文件（列表格式）"""
    with open(input_path, "r", encoding="utf-8") as in_f:
        data = json.load(in_f)

    # 假设是列表格式
    if isinstance(data, list):
        with open(output_path, "w", encoding="utf-8") as out_f:
            for item in tqdm(data, desc=f"清洗 {input_path.name}"):
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                cleaned = clean_text(text)
                if len(cleaned) >= 20 and not is_only_punctuation(cleaned):
                    out_f.write(cleaned + "\n")


def main():
    rawdata_dir = Path("rawdata")
    cleaned_dir = Path("data/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    print("开始清洗原始数据...")
    print("=" * 60)

    # ——— JSONL files ———
    # distill_r1_110k.jsonl          -> content + input
    for file_path in rawdata_dir.glob("distill_r1_110k*.jsonl"):
        output_path = cleaned_dir / f"{file_path.stem}_cleaned.txt"
        print(f"\n处理: {file_path.name}")
        process_standard_jsonl(file_path, output_path)

    # chat_train.jsonl                -> question + answer
    for file_path in rawdata_dir.glob("chat_*.jsonl"):
        output_path = cleaned_dir / f"{file_path.stem}_cleaned.txt"
        print(f"\n处理: {file_path.name}")
        process_qa_jsonl_file(file_path, output_path)

    # qwen3*.jsonl                    -> messages (chat format)
    for file_path in rawdata_dir.glob("qwen3*.jsonl"):
        output_path = cleaned_dir / f"{file_path.stem}_cleaned.txt"
        print(f"\n处理: {file_path.name}")
        process_chat_jsonl_file(file_path, output_path)

    # ——— CSV files ———
    for file_path in rawdata_dir.glob("*.csv"):
        output_path = cleaned_dir / f"{file_path.stem}_cleaned.txt"
        print(f"\n处理: {file_path.name}")
        process_csv_file(file_path, output_path)

    # ——— Plain text files ———
    for file_path in rawdata_dir.glob("*.txt"):
        output_path = cleaned_dir / f"{file_path.stem}_cleaned.txt"
        print(f"\n处理: {file_path.name}")
        process_txt_file(file_path, output_path)

    # ——— pclue directory (sub-datasets) ———
    pclue_dir = rawdata_dir / "pclue"
    if pclue_dir.exists():
        for json_file in sorted(pclue_dir.glob("*.json")):
            output_path = cleaned_dir / f"pclue_{json_file.stem}_cleaned.txt"
            print(f"\n处理: pclue/{json_file.name}")
            process_pclue_jsonl_file(json_file, output_path)

    # 合并所有清洗后的文件
    print("\n" + "=" * 60)
    print("合并清洗后的数据...")

    merged_path = cleaned_dir / "all_cleaned.txt"
    with open(merged_path, "w", encoding="utf-8") as out_f:
        for txt_file in sorted(cleaned_dir.glob("*_cleaned.txt")):
            with open(txt_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    stripped = line.strip()
                    if stripped and not is_only_punctuation(stripped):
                        out_f.write(line)

    # 统计最终结果
    total_lines = sum(1 for _ in open(merged_path, "r", encoding="utf-8"))
    total_chars = sum(len(line) for line in open(merged_path, "r", encoding="utf-8"))

    print(f"\n[OK] 清洗完成!")
    print(f"  输出文件: {merged_path}")
    print(f"  总行数: {total_lines:,}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  平均每行: {total_chars / total_lines:.1f} 字符")


if __name__ == "__main__":
    main()
