"""Inference script for word-level prediction model."""

import torch
import torch.serialization
import json
import jieba
from pathlib import Path
import argparse

from src.config import ModelConfig
from src.model.transformer import create_model

torch.serialization.add_safe_globals([ModelConfig])


class WordTokenizer:
    """Simple word-level tokenizer."""

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.word2id = json.load(f)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)

        self.pad_id = self.word2id.get("[PAD]", 0)
        self.bos_id = self.word2id.get("[BOS]", 1)
        self.eos_id = self.word2id.get("[EOS]", 2)
        self.unk_id = self.word2id.get("[UNK]", 3)

    def encode(self, text: str):
        """Encode text to token ids."""
        # 先尝试标准分词
        words = list(jieba.cut(text))
        words = [w for w in words if w.strip()]

        # 检查是否有未登录词
        ids = []
        final_words = []
        for w in words:
            if w in self.word2id:
                ids.append(self.word2id[w])
                final_words.append(w)
            else:
                # 未登录词用全模式切分，只取词表中的词
                sub_words = list(jieba.cut(w, cut_all=True))
                sub_words = [
                    sw for sw in sub_words if sw.strip() and sw in self.word2id
                ]
                if sub_words:
                    for sw in sub_words:
                        ids.append(self.word2id[sw])
                        final_words.append(sw)
                else:
                    # 如果全模式也找不到，标记为UNK
                    ids.append(self.unk_id)
                    final_words.append(w)

        return {"ids": ids, "words": final_words}

    def decode(self, ids: list):
        """Decode token ids to words."""
        return "".join([self.id2word.get(i, "[UNK]") for i in ids])


def load_model(checkpoint_path: str, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", ModelConfig())

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, device


def predict_next_tokens(
    model,
    tokenizer,
    text,
    device,
    max_new_tokens=10,
    temperature=1.0,
    top_k=5,
    deterministic=True,
):
    encoded = tokenizer.encode(text)
    tokens = encoded["ids"]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    input_len = len(tokens)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            deterministic=deterministic,
        )

    all_tokens = output_ids[0].tolist()
    generated_tokens = all_tokens[input_len:]

    special_ids = {tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id}
    generated_tokens = [t for t in generated_tokens if t not in special_ids]

    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def interactive_mode(
    model, tokenizer, device, max_new_tokens=10, temperature=1.0, top_k=5
):
    print("\n" + "=" * 50)
    print("词语联想模型 - 交互模式")
    print("输入文本，模型将预测接下来的词语")
    print("输入 'quit' 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("输入: ").strip()

            if text.lower() == "quit":
                print("退出交互模式")
                break

            if not text:
                continue

            result = predict_next_tokens(
                model,
                tokenizer,
                text,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            print(f"联想: {result}\n")

        except KeyboardInterrupt:
            print("\n退出交互模式")
            break


def main():
    parser = argparse.ArgumentParser(description="Test the word-level prediction model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/best_model.pt",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--vocab", type=str, default="data/vocab.json", help="Vocabulary file"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)"
    )
    parser.add_argument("--text", type=str, help="Input text for prediction")
    parser.add_argument(
        "--max-tokens", type=int, default=10, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, device = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {device}")

    print(f"Loading vocabulary from {args.vocab}...")
    tokenizer = WordTokenizer(args.vocab)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    if args.interactive:
        interactive_mode(
            model, tokenizer, device, args.max_tokens, args.temperature, args.top_k
        )
    elif args.text:
        result = predict_next_tokens(
            model,
            tokenizer,
            args.text,
            device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"输入: {args.text}")
        print(f"联想: {result}")
    else:
        test_texts = [
            "今天天气",
            "我们一起去",
            "这个事情",
            "我觉得",
            "时间过得",
        ]

        print("\n测试联想:")
        print("-" * 50)
        for text in test_texts:
            result = predict_next_tokens(
                model,
                tokenizer,
                text,
                device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"输入: {text}")
            print(f"联想: {result}")
            print("-" * 50)


if __name__ == "__main__":
    main()
