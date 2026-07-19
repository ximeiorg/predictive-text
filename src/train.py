"""Training script for the Decoder-Only Transformer using PyTorch Lightning."""

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import argparse
from pathlib import Path
from tokenizers import Tokenizer
import shutil

from src.config import (
    get_config_manager,
    get_model_config,
    get_training_config,
    get_data_config,
    list_model_sizes,
    ModelConfig,
    TrainingConfig,
    DataConfig,
)
from src.model.lightning_module import DecoderTransformerLightningModule, ModelCheckpointManager
from src.data.dataset import WikipediaDataset, DataLoader, load_vocab


def main():
    parser = argparse.ArgumentParser(
        description="Train Decoder-Only Transformer with Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模型尺寸配置 (从 config.yaml 加载):
  small  - 10.2M 参数, 39 MB (推荐手机)
  base   - 23.0M 参数, 88 MB (默认，PC/服务器)
  large  - 62.9M 参数, 240 MB (追求精度)

配置文件: config.yaml

示例:
  # 训练 small 模型 (自动从 config.yaml 加载参数)
  uv run src/train.py --model-size small --use-prepared-data

  # 自定义单个参数
  uv run src/train.py --model-size small --lr 8e-4 --epochs 30 --use-prepared-data

  # 使用自定义配置文件
  uv run src/train.py --config /path/to/custom_config.yaml --model-size small

  # 训练后自动评估
  uv run src/train.py --model-size small --use-prepared-data --eval-after-train

  # 查看所有可用配置
  uv run python -c "from src.config import list_model_sizes; list_model_sizes()"
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        help="Model size configuration: small, base, large (default: base)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (default: from config.yaml)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size (default: from config.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: from config.yaml)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--use-prepared-data",
        action="store_true",
        help="Use already prepared data (train.bin/val.bin)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation split ratio (default: from config.yaml)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length (default: from config.yaml)",
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="List all available model sizes and exit",
    )
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help="Run evaluation after training completes",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Gradient accumulation batches (default: 1)",
    )
    parser.add_argument(
        "--char-vocab",
        type=str,
        default=None,
        help="Path to char vocabulary file (e.g. label.txt) for char-seeded BPE",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "32", "bf16"],
        help="Training precision (default: 32)",
    )
    parser.add_argument(
        "--prepare-vocab",
        action="store_true",
        help="先运行 analyze_and_prune_vocab.py 生成词表和数据，再训练",
    )
    parser.add_argument(
        "--initial-vocab-size",
        type=int,
        default=12000,
        help="初始 BPE 词表大小 (默认 12000，配合 --prepare-vocab)",
    )

    args = parser.parse_args()

    # 先运行词表准备脚本
    if args.prepare_vocab:
        import subprocess, sys
        print("\n" + "=" * 60)
        print("运行词表准备: analyze_and_prune_vocab.py")
        print("=" * 60)
        result = subprocess.run([
            sys.executable, "scripts/analyze_and_prune_vocab.py",
            "--initial-vocab-size", str(args.initial_vocab_size),
            "--target-vocab-size", str(args.vocab_size or 5000),
        ], check=True)
        # 准备完成后自动标记使用已准备数据
        args.use_prepared_data = True

    # 初始化配置管理器
    config_manager = get_config_manager(args.config)

    # 显示可用配置
    if args.list_sizes:
        list_model_sizes()
        return

    # 从 YAML 加载配置
    model_config = config_manager.get_model_config(args.model_size)
    training_config = config_manager.get_training_config(args.model_size)
    data_config = config_manager.get_data_config(args.model_size)

    # 命令行参数覆盖
    if args.vocab_size is not None:
        model_config.vocab_size = args.vocab_size
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.lr is not None:
        training_config.learning_rate = args.lr
    if args.epochs is not None:
        training_config.num_epochs = args.epochs
    if args.max_seq_len is not None:
        model_config.max_seq_len = args.max_seq_len
    if args.val_ratio is not None:
        data_config.val_ratio = args.val_ratio

    # 输出配置信息
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"模型尺寸:     {args.model_size}")
    print(f"配置文件:     {config_manager.config_path}")
    print(f"模型配置:     {model_config}")
    print(f"设备:         {args.device}")
    print(f"Batch Size:   {training_config.batch_size}")
    print(f"Learning Rate: {training_config.learning_rate}")
    print(f"Epochs:       {training_config.num_epochs}")
    print(f"Max Seq Len:  {model_config.max_seq_len}")
    print(f"Val Ratio:    {data_config.val_ratio}")
    print(f"Vocab Size:   {model_config.vocab_size}")
    print("=" * 60 + "\n")

    output_dir = Path(args.output_dir) / args.model_size
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 显示完整的训练配置
    print("完整训练配置:")
    print(f"  Batch Size:      {training_config.batch_size}")
    print(f"  Learning Rate:   {training_config.learning_rate}")
    print(f"  Epochs:          {training_config.num_epochs}")
    print(f"  Warmup Steps:    {training_config.warmup_steps}")
    print(f"  Max Grad Norm:   {training_config.max_grad_norm}")
    print(f"  Label Smoothing: {training_config.label_smoothing}")
    print(f"  Weight Decay:    {training_config.weight_decay}")
    print(f"  Min LR:          {training_config.min_lr}")
    print(f"  Eval Steps:      {training_config.eval_steps}")
    print()

    vocab_path = data_dir / "vocab.json"
    train_data_path = data_dir / "train.bin"
    val_data_path = data_dir / "val.bin"

    if args.use_prepared_data and train_data_path.exists() and vocab_path.exists():
        print("Using prepared data...")
        tokenizer = load_vocab(str(vocab_path))
        actual_vocab_size = (
            tokenizer.vocab_size
            if hasattr(tokenizer, "vocab_size")
            else tokenizer.get_vocab_size()
        )
        print(f"Actual vocab size from vocab.json: {actual_vocab_size}")
        model_config.vocab_size = actual_vocab_size
    else:
        print(f"\n错误: 找不到准备好的数据文件。请先运行:")
        print(f"  uv run src/train.py --model-size {args.model_size} --prepare-vocab")
        print(f"\n或者使用已存在的数据:")
        print(f"  uv run src/train.py --model-size {args.model_size} --use-prepared-data")
        sys.exit(1)

    print("\nLoading datasets...")
    train_dataset = WikipediaDataset(
        str(train_data_path), str(vocab_path), max_seq_len=model_config.max_seq_len
    )
    val_dataset = WikipediaDataset(
        str(val_data_path), str(vocab_path), max_seq_len=model_config.max_seq_len
    )

    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    # Create Lightning Module
    print("Creating Lightning Module...")
    lightning_module = DecoderTransformerLightningModule(
        model_config=model_config,
        training_config=training_config,
    )
    num_params = lightning_module.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    custom_checkpoint_manager = ModelCheckpointManager(
        output_dir=output_dir,
        save_top_k=3,
        monitor="val/loss",
    )

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="lightning_logs",
    )

    # Determine precision
    if args.precision == "16":
        precision = "16-mixed"
    elif args.precision == "bf16":
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    # Create Trainer
    trainer = L.Trainer(
        max_epochs=training_config.num_epochs,
        accelerator="auto" if args.device == "auto" else args.device,
        devices=1,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=training_config.max_grad_norm,
        callbacks=[checkpoint_callback, custom_checkpoint_manager],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=training_config.eval_steps,
        limit_val_batches=200,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"\nTensorBoard logs: {logger.log_dir}")
    print(f"Run: tensorboard --logdir {output_dir / 'logs'}")

    # Train
    print("\nStarting training...")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save final model
    final_path = output_dir / "final_model.pt"
    lightning_module.save_checkpoint(output_dir, prefix="final")
    print(f"Final model saved to: {final_path}")

    # Copy best model
    best_checkpoint = checkpoint_callback.best_model_path
    if best_checkpoint and Path(best_checkpoint).exists():
        best_path = output_dir / "best_model.pt"
        shutil.copy(best_checkpoint, best_path)
        print(f"Best model copied to: {best_path}")

    print("\nTraining completed!")

    # 训练后自动评估
    if args.eval_after_train:
        print("\n" + "=" * 60)
        print("运行训练后评估...")
        print("=" * 60)

        import subprocess
        import sys

        eval_cmd = [
            sys.executable,
            "src/evaluate.py",
            "--model-size",
            args.model_size,
            "--data-path",
            str(val_data_path),
            "--vocab-path",
            str(vocab_path),
            "--max-samples",
            "2000",
            "--output",
            str(output_dir / "evaluation_report.json"),
        ]

        subprocess.run(eval_cmd)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"模型保存在: {output_dir}")
    print(f"最佳模型: {output_dir / 'best_model.pt'}")
    print(f"\n下一步:")
    print(f"  评估模型: uv run src/evaluate.py --model-size {args.model_size}")
    print(f"  推理测试: uv run src/inference.py --model-size {args.model_size} --interactive")
    print(f"  导出MNN:  uv run src/export_mobile.py --model-size {args.model_size}")
    print(f"  TensorBoard: tensorboard --logdir {output_dir / 'logs'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
