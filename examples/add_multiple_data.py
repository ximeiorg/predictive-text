"""
Example script demonstrating how to add multiple data sources.
"""

from src.data.loader import DataLoader, DataLoaderConfig, DataSource

def example_multiple_files():
    loader_config = DataLoaderConfig(
        min_length=5,
        max_length=512,
        shuffle=True,
        remove_duplicates=True
    )
    
    loader = DataLoader(loader_config)
    
    loader.add_file("大王饶命.txt", weight=1.0)
    
    loader.add_file("data/another_novel.txt", weight=1.5)
    
    loader.add_directory("data/corpus", pattern="*.txt")
    
    texts = loader.load_all()
    
    loader.save_cache("my_dataset")
    
    from pathlib import Path
    if Path("data/vocab.model").exists():
        tokens = loader.tokenize("data/vocab.model")
        loader.split_train_val(tokens, val_ratio=0.05)
    
    return loader


def example_incremental():
    loader = DataLoader(DataLoaderConfig())
    
    if loader.load_cache("data_cache"):
        print(f"Loaded {len(loader.texts)} cached texts")
    
    loader.add_file("new_data.txt")
    new_texts = loader.load_all()
    
    loader.save_cache("data_cache")
    
    return loader


def example_with_config():
    import json
    
    config = {
        "sources": [
            {"path": "大王饶命.txt", "weight": 1.0},
            {"path": "data/novel2.txt", "weight": 1.0},
            {"path": "data/novel3.txt", "weight": 0.8},
        ],
        "min_length": 10,
        "max_length": 256,
        "remove_duplicates": True
    }
    
    with open("data_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    from src.data.loader import create_data_loader_from_config
    loader = create_data_loader_from_config("data_config.json")
    
    texts = loader.load_all()
    
    return loader


if __name__ == "__main__":
    print("Example 1: Multiple files")
    loader1 = example_multiple_files()
    print(f"Total texts: {len(loader1.texts)}")
    
    print("\nExample 2: Incremental loading")
    loader2 = example_incremental()
    print(f"Total texts: {len(loader2.texts)}")