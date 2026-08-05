#!/usr/bin/env python3
"""对比 PyTorch 模型与 ONNX 模型的联想候选与 logits 差异。"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model.transformer import create_model
from src.data.dataset import load_vocab


class InferenceTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = load_vocab(tokenizer_path)
        self.vocab_size = getattr(self.tokenizer, "vocab_size", None) or self.tokenizer.get_vocab_size()
        self.eos_id = 2

    def encode(self, text: str):
        raw = self.tokenizer.encode(text)
        ids = raw.ids if hasattr(raw, "ids") else raw
        if ids and ids[-1] == self.eos_id:
            ids = ids[:-1]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def id2token(self, id_):
        if hasattr(self.tokenizer, "id2token"):
            return self.tokenizer.id2token.get(int(id_), "[UNK]")
        return self.tokenizer.decode([int(id_)])


def load_pt(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", ModelConfig())
    elif "state_dict" in checkpoint:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
        hp = checkpoint.get("hyper_parameters", {})
        config = ModelConfig.from_dict(hp.get("model_config", {}))
    else:
        raise KeyError("Unknown checkpoint format")
    model = create_model(config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def pt_logits(model, input_ids, device):
    with torch.no_grad():
        logits = model(torch.tensor([input_ids], dtype=torch.long, device=device))["logits"]
    return logits[0, -1, :].float().numpy()


def onnx_logits(session, input_name, input_ids):
    inp = np.array([input_ids], dtype=np.int64)
    logits = session.run(None, {input_name: inp})[0]
    return logits[0, -1, :]


def top_candidates(logits, tokenizer, top_k):
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    top_ids = np.argsort(probs)[-top_k:][::-1]
    out = []
    for i in top_ids:
        tok = tokenizer.id2token(i)
        if tok in ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<pad>", "<s>", "</s>", "<unk>"]:
            continue
        out.append((tok, float(probs[i]), int(i)))
    return out[:top_k]


def main():
    parser = argparse.ArgumentParser(description="对比 PyTorch 与 ONNX 候选")
    parser.add_argument("--checkpoint", type=str, default="output/base/best_model.pt")
    parser.add_argument("--onnx", type=str, default="mobile/onnx/model.onnx")
    parser.add_argument("--tokenizer", type=str, default="data/vocab.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--texts", type=str, default="今天天气,我觉得,我们一起去,你好,这")
    args = parser.parse_args()

    tokenizer = InferenceTokenizer(args.tokenizer)
    model = load_pt(args.checkpoint, "cpu")

    import onnxruntime as ort
    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name

    texts = [t.strip() for t in args.texts.split(",") if t.strip()]

    print(f"{'文本':<14} | {'PyTorch 候选':<38} | {'ONNX 候选':<38}")
    print("-" * 110)

    for text in texts:
        ids = tokenizer.encode(text)
        if not ids:
            continue
        pl = pt_logits(model, ids, "cpu")
        ol = onnx_logits(session, input_name, ids)

        pc = top_candidates(pl, tokenizer, args.top_k)
        oc = top_candidates(ol, tokenizer, args.top_k)

        pt_str = " ".join(f"{t}({p:.2%})" for t, p, _ in pc)
        on_str = " ".join(f"{t}({p:.2%})" for t, p, _ in oc)

        # 差异量化
        pm = np.exp(pl) / np.exp(pl).sum()
        om = np.exp(ol) / np.exp(ol).sum()
        cos = float(np.dot(pm, om) / (np.linalg.norm(pm) * np.linalg.norm(om)))
        mse = float(np.mean((pm - om) ** 2))
        overlap = len({i for _, _, i in pc} & {i for _, _, i in oc})

        print(f"{text:<14} | {pt_str:<38} | {on_str:<38}")
        print(f"{'':<14} | 余弦={cos:.4f}  MSE={mse:.2e}  候选重叠={overlap}/{args.top_k}")
        print("-" * 110)


if __name__ == "__main__":
    main()
