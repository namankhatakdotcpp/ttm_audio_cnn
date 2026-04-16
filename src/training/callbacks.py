"""Training callbacks for the TTM audio CNN.

Three callbacks:
  1. WandBSpecgramLogger  — logs 4 sample Mel spectrograms to WandB each epoch.
  2. LRMonitor           — logs per-parameter-group LR every optimizer step.
  3. EarlyStoppingOnPlateau — stops when val/auc hasn't improved for N epochs.
"""

from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


class WandBSpecgramLogger(Callback):
    """Log a grid of Mel spectrograms from the validation batch to WandB.

    Role in pipeline:
        After each validation epoch, grabs the first `num_samples` Mel tensors
        from the first validation batch and uploads them as WandB Image objects
        so that training progress can be visually inspected in the WandB dashboard.

    Args:
        num_samples: How many spectrograms to log per epoch.
        log_every_n_epochs: Log interval; 1 = every epoch.
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self._val_batch: Optional[dict[str, Any]] = None

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture the first validation batch for visualization."""
        if batch_idx == 0:
            self._val_batch = batch

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log spectrograms to WandB at the configured interval."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if self._val_batch is None:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return

        mels: torch.Tensor = self._val_batch["mel"]
        labels: list[int] = self._val_batch["label"]

        images = []
        n = min(self.num_samples, mels.shape[0])
        for i in range(n):
            # mel shape: (1, n_mels, T) — squeeze channel dim for display
            spec = mels[i, 0].cpu().float()  # (n_mels, T)
            # Normalize to [0,1] for image display
            spec_min = spec.min()
            spec_max = spec.max()
            spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-6)
            spec_np = spec_norm.numpy()

            label_str = "TTM" if int(labels[i]) == 1 else "noTTM"
            caption = f"sample_{i} | {label_str} | epoch {trainer.current_epoch}"
            images.append(wandb.Image(spec_np, caption=caption))

        trainer.logger.experiment.log(
            {"val/spectrograms": images, "epoch": trainer.current_epoch}
        )


class LRMonitor(LearningRateMonitor):
    """Thin subclass of Lightning's LearningRateMonitor for explicit import.

    Role in pipeline:
        Logs learning rate(s) to WandB every optimizer step so that the warmup
        ramp and cosine decay are visible in real time on the dashboard.

    Usage:
        Pass `LRMonitor(logging_interval="step")` to Trainer(callbacks=[...]).
    """

    def __init__(self, logging_interval: str = "step") -> None:
        super().__init__(logging_interval=logging_interval, log_momentum=False)


class EarlyStoppingOnPlateau(EarlyStopping):
    """Early stopping on val/auc with a generous patience for audio models.

    Role in pipeline:
        Halts training when val/auc has not improved by `min_delta` for
        `patience` consecutive epochs, preventing wasted compute on runs
        that have already converged or diverged.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in monitored metric to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        super().__init__(
            monitor="val/auc",
            mode="max",
            patience=patience,
            min_delta=min_delta,
            verbose=True,
        )
