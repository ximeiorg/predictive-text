"""Training script for the Decoder-Only Transformer."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from src.config import ModelConfig, TrainingConfig
from src.model.transformer import create_model
from src.data.dataset import WikipediaDataset, DataLoader
from src.data.loader import load_wikipedia_json


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    config,
    device,
    epoch,
    writer,
    global_step,
    output_dir=None,
    model_config=None,
):
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if global_step % 10 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        if output_dir and global_step % config.save_steps == 0:
            checkpoint_path = Path(output_dir) / f"checkpoint_step_{global_step}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": loss.item(),
                    "config": model_config,
                },
                checkpoint_path,
            )
            print(f"\nSaved checkpoint to {checkpoint_path}")

        progress_bar.set_postfix(
            {"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}
        )

    return total_loss / num_batches, global_step


def evaluate(model, dataloader, device, writer=None, global_step=None):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return float("inf")

    avg_loss = total_loss / num_batches

    if writer and global_step:
        writer.add_scalar("val/loss", avg_loss, global_step)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Decoder-Only Transformer")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/wiki_data/wiki_zh_latest.json",
        help="Path to Wikipedia JSON file",
    )
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--use-prepared-data",
        action="store_true",
        help="Use already prepared data (train.bin/val.bin)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation split ratio"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=32, help="Maximum sequence length"
    )

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

    training_config = TrainingConfig(
        batch_size=args.batch_size, learning_rate=args.lr, num_epochs=args.epochs
    )

    vocab_path = data_dir / "vocab.json"
    train_data_path = data_dir / "train.bin"
    val_data_path = data_dir / "val.bin"

    if args.use_prepared_data and train_data_path.exists() and vocab_path.exists():
        print("Using prepared data...")
        tokenizer = Tokenizer.from_file(str(vocab_path))
        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"Actual vocab size from vocab.json: {actual_vocab_size}")
        model_config = ModelConfig(
            vocab_size=actual_vocab_size, max_seq_len=args.max_seq_len
        )
    else:
        print(f"Loading data from: {args.data_path}")
        vocab_path, train_data_path, val_data_path = load_wikipedia_json(
            data_path=args.data_path,
            output_dir=str(data_dir),
            vocab_size=args.vocab_size,
            val_ratio=args.val_ratio,
            max_seq_len=args.max_seq_len,
        )
        tokenizer = Tokenizer.from_file(str(vocab_path))
        actual_vocab_size = tokenizer.get_vocab_size()
        model_config = ModelConfig(
            vocab_size=actual_vocab_size, max_seq_len=args.max_seq_len
        )

    print("\nLoading datasets...")
    train_dataset = WikipediaDataset(
        str(train_data_path), str(vocab_path), max_seq_len=model_config.max_seq_len
    )
    val_dataset = WikipediaDataset(
        str(val_data_path), str(vocab_path), max_seq_len=model_config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=False
    )

    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    print("Creating model...")
    model = create_model(model_config)
    model = model.to(device)

    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    optimizer = AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=0.01
    )

    total_steps = len(train_loader) * training_config.num_epochs
    scheduler = get_linear_warmup_scheduler(
        optimizer, warmup_steps=training_config.warmup_steps, total_steps=total_steps
    )

    # TensorBoard
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir {log_dir}")

    best_val_loss = float("inf")
    global_step = 0

    print("Starting training...")
    for epoch in range(1, training_config.num_epochs + 1):
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            training_config,
            device,
            epoch,
            writer,
            global_step,
            output_dir=output_dir,
            model_config=model_config,
        )
        val_loss = evaluate(model, val_loader, device, writer, global_step)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": model_config,
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": model_config,
            },
            checkpoint_path,
        )

    print("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
