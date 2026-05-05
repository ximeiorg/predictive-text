#!/usr/bin/env python3
"""用 ONNX 模型做联想词预测推理"""
import json, sys, numpy as np
from pathlib import Path
try:
    import onnxruntime as ort
except ImportError:
    print("请安装 onnxruntime: uv add onnxruntime"); sys.exit(1)

def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def encode(text, vocab):
    ids = [vocab.get("[BOS]", 1)]
    for ch in text:
        ids.append(vocab.get(ch, vocab.get("[UNK]", 3)))
    return ids

def decode(tok_id, id2token):
    return id2token.get(tok_id, "?")

def predict(onnx_path, vocab_path, prefix, top_k=5):
    vocab = load_vocab(vocab_path)
    id2token = {v: k for k, v in vocab.items()}
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    ids = encode(prefix, vocab)
    inp = np.array([ids], dtype=np.int64)
    logits = session.run(None, {input_name: inp})[0]

    scores = logits[0, -1, :]
    top_idx = np.argsort(scores)[-top_k:][::-1]
    candidates = []
    for i in top_idx:
        token = decode(int(i), id2token)
        if token not in ("[PAD]", "[BOS]", "[EOS]", "[UNK]"):
            candidates.append((token, float(scores[i])))
    return candidates

if __name__ == "__main__":
    onnx = "mobile/small/model_int8_dynamic.onnx"
    vocab = "mobile/small/vocab.json"

    tests = ["你好", "今天天气", "我觉得", "我们一起去", "我在北京"]
    for t in tests:
        cands = predict(onnx, vocab, t)
        items = " | ".join(f"{w}({p:.2f})" for w, p in cands)
        print(f"[{t}] -> {items}")
