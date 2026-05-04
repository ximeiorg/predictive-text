from tokenizers import Tokenizer
import json

t = Tokenizer.from_file("data/vocab.json")
print(f"Vocab size: {t.get_vocab_size()}")

enc = t.encode("你好世界")
print(f"你好世界 tokens: {enc.ids}")
print(f"Decoded: {t.decode(enc.ids)}")

vocab = t.get_vocab()
special = set(["[PAD]", "[BOS]", "[EOS]", "[UNK]"])
normal = [k for k in vocab if k not in special][:10]
print(f"Sample tokens: {normal}")
