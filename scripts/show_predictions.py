import json
d = json.load(open("output/small/ime_eval.json", encoding="utf-8"))
for x in d["samples"]:
    print(f'  {x["input"]:>12s} -> {" | ".join(x["candidates"][:5])}')
