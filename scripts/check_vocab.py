import json
v = json.load(open("data/vocab.json", encoding="utf-8"))
multi = [k for k in v if len(k) > 1 and k not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")]
multi.sort(key=lambda x: v[x])
puncts = set("，。！？、；：""''（）【】《》——…··～·")
punct_tokens = [t for t in multi if all(c in puncts for c in t)]
print(f"纯标点多字: {len(punct_tokens)}")
for t in punct_tokens[:20]:
    print(f"  id={v[t]:5d}  [{t}]")
