"""Export model for mobile deployment."""

import torch
import torch.onnx
import torch.serialization
import json
from pathlib import Path
import argparse
from tokenizers import Tokenizer

from src.config import ModelConfig
from src.model.transformer import create_model

torch.serialization.add_safe_globals([ModelConfig])


def export_to_onnx(checkpoint_path: str, output_path: str, tokenizer_path: str, max_seq_len: int = 32):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', ModelConfig())
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randint(0, config.vocab_size, (1, max_seq_len), dtype=torch.long)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=14
    )
    
    print(f"Model exported to {output_path}")
    
    # 导出词汇表
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    vocab_txt_path = Path(output_path).parent / "vocab.txt"
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for i in range(tokenizer.get_vocab_size()):
            token = tokenizer.id_to_token(i)
            f.write(token + '\n')
    
    print(f"Vocabulary exported to {vocab_txt_path}")


def export_quantized(checkpoint_path: str, output_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', ModelConfig())
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cpu')
    model.eval()
    
    state_dict = model.state_dict()
    
    quantized_state_dict = {}
    for key, value in state_dict.items():
        if 'weight' in key and value.dim() >= 2:
            quantized_state_dict[key] = value.to(torch.int8)
        else:
            quantized_state_dict[key] = value
    
    torch.save({
        'model_state_dict': quantized_state_dict,
        'config': config,
        'quantized': True
    }, output_path)
    
    original_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"Quantized model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model for mobile deployment")
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pt", help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json", help="Tokenizer file")
    parser.add_argument("--output-dir", type=str, default="mobile", help="Output directory")
    parser.add_argument("--format", type=str, choices=['onnx', 'quantized', 'both'], default='both', help="Export format")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / "model.onnx"
        export_to_onnx(args.checkpoint, str(onnx_path), args.tokenizer)
    
    if args.format in ['quantized', 'both']:
        quantized_path = output_dir / "model_quantized.pt"
        export_quantized(args.checkpoint, str(quantized_path))


if __name__ == "__main__":
    main()
