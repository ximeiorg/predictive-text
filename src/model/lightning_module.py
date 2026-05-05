"""PyTorch Lightning Module for Decoder Transformer."""

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any
from pathlib import Path
import shutil

from src.model.transformer import DecoderTransformer
from src.config import ModelConfig, TrainingConfig


class DecoderTransformerLightningModule(L.LightningModule):
    """Lightning Module for Decoder-Only Transformer."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()
        self.model = DecoderTransformer(model_config)
        self.best_val_loss = float("inf")
        self.validation_outputs = []

        self.save_hyperparameters({
            "model_config": vars(model_config),
            "training_config": vars(self.training_config),
        })

    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, on_step=True)

        return loss

    def on_after_backward(self):
        if self.global_step % 50 == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            self.log("train/grad_norm", total_norm, prog_bar=True, on_step=True)

            # Log per-layer norms
            for name, p in self.named_parameters():
                if p.grad is not None and p.dim() >= 2:
                    self.log(f"grad/{name}", p.grad.norm().item(), on_step=True)

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]
        self.validation_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=False)
        self.best_val_loss = min(self.best_val_loss, avg_loss.item())
        self.log("val/best_loss", self.best_val_loss, prog_bar=True, sync_dist=False)
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.training_config.warmup_steps, total_steps // 10)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(self.training_config.min_lr / self.training_config.learning_rate,
                       0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return self.model.count_parameters()

    def save_checkpoint(self, output_dir: Path, prefix: str = ""):
        """Save model checkpoint."""
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if prefix:
            checkpoint_path = checkpoint_dir / f"{prefix}.pt"
        else:
            checkpoint_path = checkpoint_dir / "last.pt"

        state = {
            "model_state_dict": self.state_dict(),
            "config": self.model_config,
        }

        torch.save(state, checkpoint_path)
        return checkpoint_path


class ModelCheckpointManager(L.Callback):
    """Custom checkpoint manager to save top-k best models."""

    def __init__(self, output_dir: Path, save_top_k: int = 3, monitor: str = "val/loss"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.best_checkpoints = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Save checkpoint if validation loss improved."""
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        if isinstance(current, torch.Tensor):
            current = current.item()

        epoch = trainer.current_epoch

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save last checkpoint
        checkpoint_path = checkpoint_dir / "last.pt"
        trainer.save_checkpoint(checkpoint_path)

        # Save epoch checkpoint
        epoch_path = checkpoint_dir / f"epoch={epoch}-val_loss={current:.4f}.pt"
        trainer.save_checkpoint(epoch_path)

        # Track best checkpoints
        self.best_checkpoints.append((epoch_path, current))
        self.best_checkpoints.sort(key=lambda x: x[1])

        # Keep only top-k
        while len(self.best_checkpoints) > self.save_top_k:
            old_path, _ = self.best_checkpoints.pop()
            if old_path.exists() and old_path != epoch_path:
                old_path.unlink()

        # Save best model
        if current == min([c[1] for c in self.best_checkpoints]):
            best_path = self.output_dir / "best_model.pt"
            shutil.copy(epoch_path, best_path)
