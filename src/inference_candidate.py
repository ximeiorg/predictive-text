#!/usr/bin/env python3
"""词语联想候选 - 交互式体验"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
import argparse
from tokenizers import Tokenizer
from src.config import ModelConfig
from src.model.transformer import create_model


class BPETokenizer:
    """BPE tokenizer wrapper using HuggingFace tokenizers."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")

    def encode(self, text: str):
        """Encode text to token ids."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list):
        """Decode token ids to text."""
        return self.tokenizer.decode(ids)


def load_model(checkpoint_path: str, device: str = "auto"):
    """加载模型"""
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


def get_next_token_candidates(
    model,
    tokenizer,
    text: str,
    device,
    top_k: int = 10,
    temperature: float = 1.0,
):
    """获取下一个token的多个候选"""
    tokens = tokenizer.encode(text)

    if len(tokens) == 0:
        return []

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)["logits"]
        next_token_logits = logits[0, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        probs = F.softmax(next_token_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, min(top_k, tokenizer.vocab_size))

        candidates = []
        for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            token_str = tokenizer.decode([token_id])
            # 过滤特殊 token
            if token_str not in ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<pad>", "<s>", "</s>", "<unk>"]:
                candidates.append((token_str, prob, token_id))

        return candidates


def display_candidates(candidates, show_prob=True):
    """美观地展示候选结果"""
    if not candidates:
        print("  ❌ 没有找到候选")
        return

    print("\n  🎯 联想候选:")
    print("  " + "-" * 50)

    for i, (token, prob, _) in enumerate(candidates, 1):
        if show_prob:
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {i:2d}. {token:10s} {bar} {prob:6.2%}")
        else:
            print(f"  {i:2d}. {token}")

    print("  " + "-" * 50)


def interactive_mode(model, tokenizer, device):
    """交互式联想体验"""

    config = {
        "top_k": 10,
        "temperature": 1.0,
        "show_prob": True,
    }

    print("\n" + "=" * 60)
    print("🎯 词语联想模型 - 交互体验")
    print("=" * 60)
    print("\n📖 使用说明:")
    print("  • 输入任意中文文本，模型会展示多个可能的下一个词")
    print("  • 输入数字选择候选词，自动拼接后继续联想")
    print("  • 输入 'config' 调整参数（候选数量、温度等）")
    print("  • 输入 'help' 查看更多命令")
    print("  • 输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    current_text = ""

    while True:
        try:
            if current_text:
                prompt = f"\n当前文本: 「{current_text}」"
                print(prompt)
                print("  输入新文本 / 选择候选(数字) / 命令:")

            user_input = input("  👤 ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n  👋 感谢使用，再见！")
                break

            elif user_input.lower() == "help":
                print("\n  📚 帮助信息:")
                print("  " + "-" * 50)
                print("  命令列表:")
                print("    • 直接输入文本 → 开始联想")
                print("    • 输入数字(1-10) → 选择候选词并继续")
                print("    • 'clear' → 清空当前文本")
                print("    • 'show' → 显示当前文本")
                print("    • 'again' → 对当前文本重新联想")
                print("    • 'config' → 调整参数")
                print("    • 'help' → 显示帮助")
                print("    • 'quit' → 退出程序")
                print("  " + "-" * 50)
                print("\n  参数说明:")
                print("    • top_k: 候选数量 (1-50)")
                print("    • temperature: 创造性 (0.1-2.0)")
                print("      - 低(0.5): 更确定，常见词")
                print("      - 高(1.5): 更随机，创意词")
                print("    • show_prob: 是否显示概率 (on/off)")
                print("  " + "-" * 50)
                continue

            elif user_input.lower() == "config":
                print("\n  ⚙️  当前配置:")
                print(f"    • 候选数量: {config['top_k']}")
                print(f"    • 温度参数: {config['temperature']}")
                print(f"    • 显示概率: {config['show_prob']}")
                print("\n  调整参数:")
                print("    输入格式: 参数名=值")
                print("    例如: top_k=15, temperature=0.8, show_prob=off")

                param_input = input("  ⚙️  ").strip()
                if param_input:
                    try:
                        for param in param_input.split(","):
                            key, value = param.strip().split("=")
                            key = key.strip().lower()
                            value = value.strip()

                            if key == "top_k":
                                config["top_k"] = int(value)
                                print(f"    ✓ 候选数量设置为 {config['top_k']}")
                            elif key == "temperature":
                                config["temperature"] = float(value)
                                print(f"    ✓ 温度设置为 {config['temperature']}")
                            elif key in ["show_prob", "prob"]:
                                config["show_prob"] = value.lower() in [
                                    "on",
                                    "true",
                                    "yes",
                                    "1",
                                ]
                                print(f"    ✓ 显示概率设置为 {config['show_prob']}")
                    except Exception as e:
                        print(f"    ❌ 参数格式错误: {e}")
                continue

            elif user_input.lower() == "clear":
                current_text = ""
                print("  ✓ 已清空当前文本")
                continue

            elif user_input.lower() == "show":
                if current_text:
                    print(f"  当前文本: 「{current_text}」")
                    print(
                        f"  长度: {len(current_text)} 字符, {len(tokenizer.encode(current_text))} tokens"
                    )
                else:
                    print("  当前文本为空")
                continue

            elif user_input.lower() == "again":
                if not current_text:
                    print("  ❌ 当前文本为空，请先输入文本")
                    continue
                user_input = current_text

            elif user_input.isdigit() and current_text:
                try:
                    choice_idx = int(user_input)
                    candidates = get_next_token_candidates(
                        model,
                        tokenizer,
                        current_text,
                        device,
                        config["top_k"],
                        config["temperature"],
                    )

                    if 1 <= choice_idx <= len(candidates):
                        selected_token, prob, _ = candidates[choice_idx - 1]
                        current_text += selected_token

                        print(f"\n  ✓ 已选择: 「{selected_token}」 (概率 {prob:.2%})")
                        print(f"  新文本: 「{current_text}」")

                        new_candidates = get_next_token_candidates(
                            model,
                            tokenizer,
                            current_text,
                            device,
                            config["top_k"],
                            config["temperature"],
                        )
                        display_candidates(new_candidates, config["show_prob"])
                    else:
                        print(f"  ❌ 请输入 1-{len(candidates)} 之间的数字")
                except Exception as e:
                    print(f"  ❌ 选择失败: {e}")
                continue

            else:
                current_text = user_input

            candidates = get_next_token_candidates(
                model,
                tokenizer,
                current_text,
                device,
                config["top_k"],
                config["temperature"],
            )

            display_candidates(candidates, config["show_prob"])

            if candidates:
                print("\n  💡 提示:")
                print(f"    • 输入数字(1-{len(candidates)}) 选择候选词")
                print("    • 输入新文本重新开始")
                print("    • 输入 'again' 重新联想当前文本")

        except KeyboardInterrupt:
            print("\n\n  👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n  ❌ 错误: {e}")
            print("  请重试或输入 'help' 查看帮助")


def main():
    parser = argparse.ArgumentParser(description="词语联想候选 - 交互式体验")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/best_model.pt",
        help="模型checkpoint路径",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="data/tokenizer.json", help="Tokenizer文件"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="设备 (auto/cpu/cuda/mps)"
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ 模型文件不存在: {args.checkpoint}")
        print("   请先训练模型: uv run src/train.py --use-prepared-data")
        return

    if not Path(args.tokenizer).exists():
        print(f"❌ Tokenizer 不存在: {args.tokenizer}")
        return

    print(f"📦 加载模型: {args.checkpoint}")
    model, device = load_model(args.checkpoint, args.device)
    print(f"✓ 设备: {device}")

    print(f"📦 加载 tokenizer...")
    tokenizer = BPETokenizer(args.tokenizer)
    print(f"✓ 词汇表大小: {tokenizer.vocab_size}")

    interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
