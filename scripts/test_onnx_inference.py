#!/usr/bin/env python3
"""测试 ONNX 推理 - 对比 PyTorch 和 ONNX 结果"""

import torch
import torch.nn.functional as F
import torch.onnx
import numpy as np
import json
from pathlib import Path
import argparse
from tokenizers import Tokenizer
from src.config import ModelConfig
from src.model.transformer import create_model

try:
    import onnxruntime as ort
except ImportError:
    print("请安装 onnxruntime: pip install onnxruntime")
    exit(1)


class BPETokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list):
        return self.tokenizer.decode(ids)


def export_onnx(checkpoint_path: str, output_path: str):
    """导出 ONNX 模型"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids)["logits"]

    wrapper = Wrapper(model)
    wrapper.eval()

    dummy_input = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    print(f"✓ ONNX 导出: {output_path}")
    return output_path


def load_pytorch_model(checkpoint_path: str):
    """加载 PyTorch 模型"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())

    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def load_onnx_model(onnx_path: str):
    """加载 ONNX 模型"""
    session = ort.InferenceSession(onnx_path)

    print(f"\nONNX 模型信息:")
    for inp in session.get_inputs():
        print(f"  输入: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    for out in session.get_outputs():
        print(f"  输出: {out.name}, shape: {out.shape}, type: {out.type}")

    return session


def inference_pytorch(model, tokens: list):
    """PyTorch 推理"""
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)["logits"]

    return logits[0, -1, :].numpy()


def inference_onnx(session, tokens: list):
    """ONNX 推理"""
    input_ids = np.array([tokens], dtype=np.int64)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_ids})

    return outputs[0][0, -1, :]


def get_candidates(logits, tokenizer, top_k=10, temperature=1.0):
    """从 logits 获取候选词"""
    if temperature != 1.0:
        logits = logits / temperature

    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]

    candidates = []
    for idx, prob in zip(top_indices, top_probs):
        token_str = tokenizer.decode([int(idx)])
        if token_str not in [
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "[UNK]",
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
        ]:
            candidates.append((token_str, prob, int(idx)))

    return candidates


def compare_results(pytorch_logits, onnx_logits, text, tokenizer, top_k=10):
    """对比 PyTorch 和 ONNX 结果"""
    print(f"\n{'=' * 60}")
    print(f"输入文本: 「{text}」")
    tokens = tokenizer.encode(text)
    print(f"Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

    diff = np.abs(pytorch_logits - onnx_logits)
    print(f"\n数值对比:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  相对误差: {(diff / (np.abs(pytorch_logits) + 1e-8)).mean():.6f}")

    pytorch_candidates = get_candidates(pytorch_logits, tokenizer, top_k)
    onnx_candidates = get_candidates(onnx_logits, tokenizer, top_k)

    print(f"\nPyTorch 候选词 (Top-{top_k}):")
    for i, (token, prob, _) in enumerate(pytorch_candidates[:5], 1):
        print(f"  {i}. {token:10s} {prob:6.2%}")

    print(f"\nONNX 候选词 (Top-{top_k}):")
    for i, (token, prob, _) in enumerate(onnx_candidates[:5], 1):
        print(f"  {i}. {token:10s} {prob:6.2%}")

    match_count = sum(
        1 for p, o in zip(pytorch_candidates, onnx_candidates) if p[0] == o[0]
    )
    print(
        f"\n候选匹配: {match_count}/{min(len(pytorch_candidates), len(onnx_candidates))}"
    )

    return match_count >= min(len(pytorch_candidates), len(onnx_candidates)) - 1


def test_dynamic_sequence(session, tokenizer):
    """测试动态序列长度"""
    print(f"\n{'=' * 60}")
    print("测试动态序列长度")

    test_cases = [
        "你好",
        "今天天气",
        "中华人民共和国",
        "这是一个比较长的句子用来测试",
    ]

    for text in test_cases:
        tokens = tokenizer.encode(text)
        input_ids = np.array([tokens], dtype=np.int64)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_ids})

        expected_shape = (1, len(tokens), tokenizer.vocab_size)
        actual_shape = outputs[0].shape

        print(f"\n  文本: 「{text}」")
        print(f"  Tokens: {len(tokens)}")
        print(f"  输出形状: {actual_shape} (期望: {expected_shape})")

        if actual_shape == expected_shape:
            print(f"  ✓ 动态序列长度正常")
        else:
            print(f"  ✗ 形状不匹配!")
            return False

    return True


def interactive_compare(pytorch_model, onnx_session, tokenizer):
    """交互式对比 PyTorch 和 ONNX"""
    print("\n" + "=" * 60)
    print("交互式对比 PyTorch vs ONNX")
    print("=" * 60)
    print("输入文本测试，对比两种推理引擎的结果")
    print("输入 'quit' 退出")
    print("=" * 60)

    while True:
        try:
            text = input("\n请输入文本: ").strip()

            if not text:
                continue

            if text.lower() in ["quit", "exit", "q"]:
                break

            tokens = tokenizer.encode(text)

            if len(tokens) == 0:
                print("文本编码失败")
                continue

            pytorch_logits = inference_pytorch(pytorch_model, tokens)
            onnx_logits = inference_onnx(onnx_session, tokens)

            compare_results(pytorch_logits, onnx_logits, text, tokenizer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="测试 ONNX 推理")
    parser.add_argument("--checkpoint", type=str, default="output/small/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--onnx", type=str, default=None, help="已有 ONNX 模型路径")
    parser.add_argument("--interactive", action="store_true", help="交互式对比")
    parser.add_argument("--test-dynamic", action="store_true", help="测试动态序列")

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ 模型不存在: {args.checkpoint}")
        return 1

    if not Path(args.tokenizer).exists():
        print(f"❌ Tokenizer 不存在: {args.tokenizer}")
        return 1

    tokenizer = BPETokenizer(args.tokenizer)
    print(f"✓ Tokenizer: {tokenizer.vocab_size} 词")

    onnx_path = args.onnx
    if not onnx_path:
        onnx_path = "mobile/small/test_model.onnx"
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        export_onnx(args.checkpoint, onnx_path)

    pytorch_model, config = load_pytorch_model(args.checkpoint)
    onnx_session = load_onnx_model(onnx_path)

    if args.test_dynamic:
        success = test_dynamic_sequence(onnx_session, tokenizer)
        if success:
            print("\n✓ 动态序列长度测试通过")
        else:
            print("\n✗ 动态序列长度测试失败")
            return 1

    test_texts = [
        "你好",
        "今天天气很好",
        "中华人民共和国",
        "这是一个测试",
        "输入法联想功能",
    ]

    print("\n" + "=" * 60)
    print("批量测试对比")
    print("=" * 60)

    all_match = True
    for text in test_texts:
        tokens = tokenizer.encode(text)

        pytorch_logits = inference_pytorch(pytorch_model, tokens)
        onnx_logits = inference_onnx(onnx_session, tokens)

        match = compare_results(pytorch_logits, onnx_logits, text, tokenizer)
        all_match = all_match and match

    if all_match:
        print("\n✓ 所有测试通过，PyTorch 和 ONNX 结果一致")
    else:
        print("\n✗ 存在差异，请检查模型导出")

    if args.interactive:
        interactive_compare(pytorch_model, onnx_session, tokenizer)

    return 0


if __name__ == "__main__":
    exit(main())
