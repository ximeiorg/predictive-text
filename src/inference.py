#!/usr/bin/env python3
"""Inference script for word-level prediction model."""

import torch
import torch.nn.functional as F
import torch.serialization
import json
from pathlib import Path
import argparse
from tokenizers import Tokenizer

from src.config import ModelConfig, list_model_sizes, MODEL_SIZES
from src.model.transformer import create_model
from src.data.dataset import SimpleTokenizer, load_vocab

torch.serialization.add_safe_globals([ModelConfig])


class BPETokenizer:
    """兼容旧版 tokenizers 和新版 SimpleTokenizer 的包装器。"""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = load_vocab(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else self.tokenizer.get_vocab_size()
        self.eos_id = 2  # [EOS]=2 在所有模型中统一
        self.unk_id = self._get_id("[UNK]")

    def _get_id(self, token: str):
        if isinstance(self.tokenizer, SimpleTokenizer):
            return self.tokenizer.vocab.get(token, 0)
        return self.tokenizer.token_to_id(token)

    def encode(self, text: str):
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list):
        if isinstance(self.tokenizer, SimpleTokenizer):
            return "".join(self.tokenizer.id2token.get(i, "[UNK]") for i in ids)
        return self.tokenizer.decode(ids)


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

    # 兼容 Lightning checkpoint 和自定义 checkpoint 格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", ModelConfig())
    elif "state_dict" in checkpoint:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
        hp = checkpoint.get("hyper_parameters", {})
        config = ModelConfig.from_dict(hp.get("model_config", {}))
    else:
        raise KeyError("Unrecognized checkpoint format: no 'model_state_dict' or 'state_dict' found")

    model = create_model(config)
    model.load_state_dict(state_dict, strict=False)
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
):
    """Predict next tokens from input text."""
    tokens = tokenizer.encode(text)
    
    if len(tokens) == 0:
        return []
    
    # SimpleTokenizer 会在末尾加 [EOS]，推理时要去掉
    if tokens[-1] == getattr(tokenizer, 'eos_id', 2):
        tokens = tokens[:-1]
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)["logits"]
            next_token_logits = logits[0, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_id:
                break
            
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if input_ids.size(1) >= model.config.max_seq_len:
                break
    
    return generated


def interactive_mode(model, tokenizer, device):
    """Interactive prediction mode."""
    print("\n" + "=" * 60)
    print("词语联想模型 - 交互模式")
    print("=" * 60)
    print("\n使用说明:")
    print("  • 输入文本，模型会预测接下来的词")
    print("  • 输入 'quit' 或 'exit' 退出")
    print("  • 输入 'help' 查看更多选项")
    print("=" * 60)
    
    config = {
        "max_new_tokens": 10,
        "temperature": 1.0,
        "top_k": 5,
    }
    
    while True:
        try:
            user_input = input("\n请输入文本: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break
            
            if user_input.lower() == "help":
                print("\n帮助信息:")
                print("  • 直接输入文本进行预测")
                print("  • 'config' - 调整参数")
                print("  • 'quit' - 退出")
                print("\n参数说明:")
                print("  • max_new_tokens: 生成词数 (1-20)")
                print("  • temperature: 创造性 (0.1-2.0)")
                print("  • top_k: 候选数量 (1-10)")
                continue
            
            if user_input.lower() == "config":
                print("\n当前配置:")
                for key, value in config.items():
                    print(f"  • {key}: {value}")
                print("\n输入参数 (例如: temperature=0.8):")
                param_input = input("  ").strip()
                if param_input:
                    try:
                        key, value = param_input.split("=")
                        key = key.strip()
                        value = value.strip()
                        if key in config:
                            config[key] = float(value) if key != "max_new_tokens" and key != "top_k" else int(value)
                            print(f"  ✓ {key} 设置为 {config[key]}")
                    except Exception as e:
                        print(f"  ❌ 格式错误: {e}")
                continue
            
            # 预测
            generated_ids = predict_next_tokens(
                model, tokenizer, user_input, device,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_k=config["top_k"],
            )
            
            if generated_ids:
                generated_text = tokenizer.decode(generated_ids)
                print(f"\n预测结果: {user_input}{generated_text}")
            else:
                print("\n未能生成预测")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inference for prediction model")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=list(MODEL_SIZES.keys()),
        help="Model size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (auto if not set)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="data/tokenizer.json", help="Tokenizer path"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--text", type=str, help="Input text for prediction")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sampling")

    args = parser.parse_args()

    # 自动查找 checkpoint
    if args.checkpoint is None:
        checkpoint_path = Path("output") / args.model_size / "best_model.pt"
    else:
        checkpoint_path = Path(args.checkpoint)

    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("   请先训练模型: uv run src/train.py --model-size {} --use-prepared-data".format(args.model_size))
        return

    if not tokenizer_path.exists():
        print(f"❌ Tokenizer 不存在: {tokenizer_path}")
        print("   请先训练 tokenizer: uv run src/train.py --use-prepared-data")
        return

    print(f"📦 加载模型: {checkpoint_path}")
    model, device = load_model(str(checkpoint_path), args.device)
    print(f"✓ 设备: {device}")
    print(f"✓ 模型参数: {model.count_parameters():,}")

    print(f"\n📦 加载 tokenizer: {tokenizer_path}")
    tokenizer = BPETokenizer(str(tokenizer_path))
    print(f"✓ 词汇表大小: {tokenizer.vocab_size}")

    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.text:
        generated_ids = predict_next_tokens(
            model, tokenizer, args.text, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        generated_text = tokenizer.decode(generated_ids)
        print(f"\n输入: {args.text}")
        print(f"预测: {args.text}{generated_text}")
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
