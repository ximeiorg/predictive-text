import re
from pathlib import Path

INPUT = Path("rawdata/user_post_content.txt")
OUTPUT = Path("rawdata/user_post_content_cleaned.txt")


def clean_line(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None

    # Remove repost markers
    s = re.sub(r'//@\S+?[:：\s]', '', s)
    s = re.sub(r'//@\S+', '', s)

    # Remove @mentions
    s = re.sub(r'@\S+', '', s)

    # Remove URLs
    s = re.sub(r'https?://\S+', '', s)

    # Remove ##double hashtags## (weibo trending topics)
    s = re.sub(r'##.*?##', '', s)

    # Remove #single hashtags# — replace with empty
    s = re.sub(r'#[^#]+?#', '', s)
    s = s.replace('#', '')

    # Remove specific contact patterns (targeted, not greedy)
    s = re.sub(r'[微Vv][信x信X][：:\s]*[A-Za-z0-9_.@]+', '', s)
    s = re.sub(r'QQ[：:\s]*\d+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'weixin[：:\s]*\w+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'(?:手机|电话|联系|咨询)[：:\s]*\d{6,}', '', s)
    s = re.sub(r'(?<!\d)\d{9,}(?!\d)', '', s)

    s = re.sub(r'分享图片', '', s)
    s = re.sub(r'转发微博', '', s)

    # Remove pipe-delimited content and stray pipes
    s = re.sub(r'\|[^|]+\|', '', s)
    s = re.sub(r'\|微博视频|\|视频', '', s)
    s = re.sub(r'\|', '', s)

    # Clean lingering colon + word fragments (remnants of contact info patterns)
    s = re.sub(r'[：:]\s*[A-Za-z0-9_.@]+', '', s)

    # Remove stray ( and ) around nothing
    s = re.sub(r'（）', '', s)
    s = re.sub(r'\(\)', '', s)

    # Normalize whitespace
    s = re.sub(r'\s+', '', s)

    # Trim trailing punctuation
    s = s.rstrip(' ，。！？、；：,.;:!?…~-*')

    if not s or len(s) < 4:
        return None

    return s


def main():
    raw = INPUT.read_text(encoding='utf-8').splitlines()
    cleaned = []
    removed_short = 0
    removed_dupe = 0

    seen = set()
    for line in raw:
        out = clean_line(line)
        if out is None:
            removed_short += 1
            continue
        if out in seen:
            removed_dupe += 1
            continue
        seen.add(out)
        cleaned.append(out)

    OUTPUT.write_text('\n'.join(cleaned) + '\n', encoding='utf-8')

    print(f"Lines read:      {len(raw)}")
    print(f"Lines written:   {len(cleaned)}")
    print(f"Removed (short): {removed_short}")
    print(f"Removed (dupe):  {removed_dupe}")

    # Show samples
    print("\n--- First 10 lines ---")
    for l in cleaned[:10]:
        print(f"  [{len(l):3d}] {l}")


if __name__ == '__main__':
    main()
