"""Inference script for testing the trained model."""

import torch
import torch.serialization
import sentencepiece as spm
from pathlib import Path
import argparse

from src.config import ModelConfig
from src.model.transformer import create_model

torch.serialization.add_safe_globals([ModelConfig])


def load_model(checkpoint_path: str, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', ModelConfig())
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_next_tokens(model, sp, text, device, max_new_tokens=10, temperature=1.0, top_k=5):
    tokens = sp.encode(text, out_type=int)
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_tokens = output_ids[0].tolist()
    generated_text = sp.decode(generated_tokens)
    
    return generated_text


def interactive_mode(model, sp, device, max_new_tokens=10, temperature=1.0, top_k=5):
    print("\n" + "="*50)
    print("输入法预测模型 - 交互模式")
    print("输入文本，模型将预测接下来的词")
    print("输入 'quit' 退出")
    print("="*50 + "\n")
    
    while True:
        try:
            text = input("输入: ").strip()
            
            if text.lower() == 'quit':
                print("退出交互模式")
                break
            
            if not text:
                continue
            
            result = predict_next_tokens(
                model, sp, text, device, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            print(f"预测: {result}\n")
            
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break


def main():
    parser = argparse.ArgumentParser(description="Test the trained model")
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pt", help="Model checkpoint")
    parser.add_argument("--vocab", type=str, default="data/vocab.model", help="Vocabulary file")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--text", type=str, help="Input text for prediction")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, device = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {device}")
    
    print(f"Loading vocabulary from {args.vocab}...")
    sp = spm.SentencePieceProcessor()
    sp.load(args.vocab)
    print(f"Vocabulary size: {sp.get_piece_size()}")
    
    if args.interactive:
        interactive_mode(model, sp, device, args.max_tokens, args.temperature, args.top_k)
    elif args.text:
        result = predict_next_tokens(
            model, sp, args.text, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"输入: {args.text}")
        print(f"预测: {result}")
    else:
        test_texts = [
            "今天天气",
            "我们一起去",
            "这个事情",
            "我觉得",
            "时间过得",
        ]
        
        print("\n测试预测:")
        print("-" * 50)
        for text in test_texts:
            result = predict_next_tokens(
                model, sp, text, device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print(f"输入: {text}")
            print(f"预测: {result}")
            print("-" * 50)


if __name__ == "__main__":
    main()