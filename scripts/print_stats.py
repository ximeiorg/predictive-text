from pathlib import Path

cleaned_dir = Path("data/cleaned")
total_size = 0
for f in sorted(cleaned_dir.glob("*_cleaned.txt")):
    if f.name == "all_cleaned.txt":
        continue
    size = f.stat().st_size
    with open(f, "r", encoding="utf-8") as fh:
        nlines = sum(1 for _ in fh)
    total_size += size
    label = f.name.replace("_cleaned.txt", "")
    print(f"{label:<40s} {nlines:>10,} lines  {size/1024/1024:>8.1f} MB")

print()
print(f"{'TOTAL':<40s} {'':>10}  {total_size/1024/1024:>8.1f} MB")

af = cleaned_dir / "all_cleaned.txt"
asize = af.stat().st_size
print(f"{'all_cleaned.txt':<40s} {asize/1024/1024:>8.1f} MB")
