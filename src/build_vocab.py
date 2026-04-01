"""Build vocabulary using SentencePiece."""

import sentencepiece as spm
import argparse
from pathlib import Path
from src.config import DataConfig


def build_vocab(input_file: str, vocab_size: int = 5120, model_prefix: str = "data/vocab"):
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece='[PAD]',
        bos_piece='[BOS]',
        eos_piece='[EOS]',
        unk_piece='[UNK]',
        user_defined_symbols=['[PAD]', '[BOS]', '[EOS]', '[UNK]'],
        num_threads=4,
        train_extremely_large_corpus=False
    )
    
    print(f"Vocabulary saved to {model_prefix}.model and {model_prefix}.vocab")
    
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print(f"Vocabulary size: {sp.get_piece_size()}")


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from text corpus")
    parser.add_argument("--input", type=str, default="大王饶命.txt", help="Input text file")
    parser.add_argument("--vocab-size", type=int, default=5120, help="Vocabulary size")
    parser.add_argument("--output-prefix", type=str, default="data/vocab", help="Output model prefix")
    
    args = parser.parse_args()
    
    build_vocab(args.input, args.vocab_size, args.output_prefix)


if __name__ == "__main__":
    main()