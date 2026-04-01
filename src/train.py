"""Training script for the Decoder-Only Transformer."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import sentencepiece as spm
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from src.config import ModelConfig, TrainingConfig, DataConfig
from src.model.transformer import create_model
from src.data.dataset import TextDataset, DataLoader as SimpleDataLoader
from src.data.loader import create_data_loader_from_config, quick_load_single_file


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Decoder-Only Transformer")
    
    parser.add_argument("--data-file", type=str, default=None, 
                       help="Single training data file (backward compatible)")
    parser.add_argument("--data-config", type=str, default=None,
                       help="Data config JSON file (new data loader)")
    parser.add_argument("--vocab-size", type=int, default=5120, help="Vocabulary size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--use-prepared-data", action="store_true",
                       help="Use already prepared data (train.bin/val.bin)")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = ModelConfig(vocab_size=args.vocab_size)
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    vocab_path = data_dir / "vocab.model"
    train_data_path = data_dir / "train.bin"
    val_data_path = data_dir / "val.bin"
    
    if args.use_prepared_data and train_data_path.exists() and vocab_path.exists():
        print("Using prepared data...")
        
    elif args.data_config:
        print(f"Loading data from config: {args.data_config}")
        from src.prepare_data import main as prepare_main
        import sys
        
        old_argv = sys.argv
        sys.argv = ['prepare_data', '--config', args.data_config]
        prepare_main()
        sys.argv = old_argv
        
    elif args.data_file:
        print(f"Quick load single file: {args.data_file}")
        loader = quick_load_single_file(
            args.data_file,
            vocab_path=str(vocab_path) if vocab_path.exists() else None,
            output_dir=str(data_dir),
            val_ratio=0.05
        )
        
        if not vocab_path.exists():
            print("Building vocabulary...")
            import sentencepiece as spm
            temp_vocab_path = str(data_dir / "vocab")
            temp_corpus = data_dir / "cache" / "temp_corpus.txt"
            
            spm.SentencePieceTrainer.train(
                input=str(temp_corpus),
                model_prefix=temp_vocab_path,
                vocab_size=args.vocab_size,
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
                num_threads=4
            )
            print(f"Vocabulary saved to {vocab_path}")
            
            loader.tokenize(str(vocab_path))
            loader.split_train_val(
                loader.tokenize(str(vocab_path)),
                val_ratio=0.05,
                train_path=str(train_data_path),
                val_path=str(val_data_path)
            )
    else:
        print("No data specified. Use --data-file or --data-config")
        print("Or use --use-prepared-data to use existing train.bin/val.bin")
        return
    
    print("Loading datasets...")
    train_dataset = TextDataset(
        str(train_data_path),
        str(vocab_path),
        max_seq_len=model_config.max_seq_len
    )
    val_dataset = TextDataset(
        str(val_data_path),
        str(vocab_path),
        max_seq_len=model_config.max_seq_len
    )
    
    train_loader = SimpleDataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    print("Creating model...")
    model = create_model(model_config)
    model = model.to(device)
    
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * training_config.num_epochs
    scheduler = get_linear_warmup_scheduler(
        optimizer, 
        warmup_steps=training_config.warmup_steps,
        total_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(1, training_config.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, training_config, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': model_config
        }, checkpoint_path)
    
    print("Training completed!")


if __name__ == "__main__":
    main()