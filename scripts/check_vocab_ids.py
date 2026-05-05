import json
v = json.load(open("data/vocab.json", encoding="utf-8"))
id2t = {v[k]: k for k in v}
print("Top 40 token IDs:")
for tid in range(40):
    token = id2t.get(tid, "?")
    display = repr(token) if len(token) <= 10 else repr(token[:10]) + "..."
    print(f"  ID={tid:3d}  {display}")

# Also check: current vocab size
print(f"\nVocab size: {len(v)}")
