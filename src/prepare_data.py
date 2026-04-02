"""Prepare training data from Wikipedia JSON."""

import argparse
from pathlib import Path
from src.data.loader import load_wikipedia_json


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from Wikipedia JSON"
    )
    parser.add_argument(
        "--data-path",
        default="data/wiki_data/wiki_zh_latest.json",
        help="Path to Wikipedia JSON file or directory",
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation ratio"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=32, help="Maximum sequence length"
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("Preparing Training Data")
    print(f"{'=' * 60}")
    print(f"Data path:   {args.data_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Vocab size:  {args.vocab_size}")
    print(f"Val ratio:   {args.val_ratio}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"{'=' * 60}\n")

    vocab_path, train_path, val_path = load_wikipedia_json(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        val_ratio=args.val_ratio,
        max_seq_len=args.max_seq_len,
    )

    print(f"\n{'=' * 60}")
    print("Data preparation complete!")
    print(f"{'=' * 60}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Train data: {train_path}")
    print(f"Val data:   {val_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
