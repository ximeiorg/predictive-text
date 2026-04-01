"""Prepare training data using the flexible data loader."""

import argparse
from pathlib import Path

from src.data.loader import (
    DataLoader, 
    DataLoaderConfig, 
    DataSource,
    create_data_loader_from_config
)
from src.build_vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--config", type=str, default="data_config.json", 
                       help="Data config file (JSON)")
    parser.add_argument("--vocab-size", type=int, default=5120, 
                       help="Vocabulary size")
    parser.add_argument("--vocab-output", type=str, default="data/vocab.model",
                       help="Vocabulary output path")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                       help="Validation split ratio")
    parser.add_argument("--use-cache", action="store_true",
                       help="Use cached data if available")
    parser.add_argument("--cache-name", type=str, default="data_cache",
                       help="Cache name")
    
    parser.add_argument("--add-file", type=str, action="append",
                       help="Add additional file (can specify multiple)")
    parser.add_argument("--add-dir", type=str, action="append",
                       help="Add directory of files (can specify multiple)")
    parser.add_argument("--pattern", type=str, default="*.txt",
                       help="File pattern for directory scanning")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if Path(args.config).exists():
        loader = create_data_loader_from_config(args.config)
    else:
        loader = DataLoader(DataLoaderConfig())
    
    if args.add_file:
        for file_path in args.add_file:
            loader.add_file(file_path)
            
    if args.add_dir:
        for directory in args.add_dir:
            loader.add_directory(directory, args.pattern)
    
    if args.use_cache and loader.load_cache(args.cache_name):
        print(f"\n✓ Using cached data: {len(loader.texts)} texts")
    else:
        texts = loader.load_all()
        loader.save_cache(args.cache_name)
    
    vocab_path = Path(args.vocab_output)
    vocab_model_path = vocab_path.with_suffix('.model')
    
    if not vocab_model_path.exists():
        print(f"\nBuilding vocabulary with size {args.vocab_size}...")
        
        temp_corpus = cache_dir / "temp_corpus.txt"
        with open(temp_corpus, 'w', encoding='utf-8') as f:
            for text in loader.texts:
                f.write(text + '\n')
        
        build_vocab(
            str(temp_corpus), 
            vocab_size=args.vocab_size,
            model_prefix=str(vocab_path.parent / vocab_path.stem)
        )
        
        print(f"✓ Vocabulary saved to {vocab_model_path}")
    else:
        print(f"✓ Using existing vocabulary: {vocab_model_path}")
    
    tokens = loader.tokenize(str(vocab_model_path))
    
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    
    loader.split_train_val(
        tokens, 
        val_ratio=args.val_ratio,
        train_path=str(train_path),
        val_path=str(val_path)
    )
    
    print(f"\n{'='*50}")
    print("Data preparation completed!")
    print(f"{'='*50}")
    print(f"Vocabulary: {vocab_model_path}")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Cache: {cache_dir / args.cache_name}.txt")
    
    stats_file = cache_dir / f"{args.cache_name}_stats.json"
    if stats_file.exists():
        print(f"\nStatistics:")
        import json
        with open(stats_file) as f:
            stats = json.load(f)
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()