"""AudioCNNModule: PyTorch Lightning training/validation loop for the TTM audio model.

Composes AudioCNN + ProjectionHead + a linear binary classifier.
Supports AMP (torch.cuda.amp.autocast), class-weighted BCE loss,
cosine LR schedule with linear warmup, and WandB metric logging.

Output contract for S6/S7:
  After training, embeddings are extracted (no classifier head) and saved as
  (T, 256) tensors. See scripts/extract_embeddings.py.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.data.ttm_audio_dataset import TTMAudioDataset
from src.models.audio_cnn import AudioCNN
from src.models.projection_head import ProjectionHead


class AudioCNNModule(pl.LightningModule):
    """Lightning module encapsulating AudioCNN, ProjectionHead, and a classifier.

    Role in pipeline:
        DataLoader → AudioCNNModule.training_step → loss + WandB logs
        DataLoader → AudioCNNModule.validation_step → AUC, F1, precision, recall

    The classifier head (Linear 256 → 1) is used only during training and
    validation; it is detached before embedding extraction.

    Args:
        backbone_cfg: Kwargs for AudioCNN (pretrained, dropout).
        head_cfg: Kwargs for ProjectionHead (input_dim, hidden_dim, output_dim, dropout).
        train_dataset: Training TTMAudioDataset (for class weight computation).
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        pin_memory: DataLoader pin_memory flag.
        lr: Base learning rate for AdamW.
        weight_decay: AdamW weight decay.
        max_epochs: Total training epochs (for cosine schedule period).
        warmup_epochs: Linear warmup duration in epochs.
        label_smoothing: Label smoothing applied to BCE targets.
        class_weight_mode: "inverse_freq" uses dataset.class_weights(); "none" skips.
        use_amp: Whether to use AMP (autocast). Effective via Lightning's precision flag.
    """

    def __init__(
        self,
        backbone_cfg: Optional[dict[str, Any]] = None,
        head_cfg: Optional[dict[str, Any]] = None,
        train_dataset: Optional[TTMAudioDataset] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 40,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
        class_weight_mode: str = "inverse_freq",
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset"])

        backbone_cfg = backbone_cfg or {}
        head_cfg = head_cfg or {}

        self.backbone = AudioCNN(**backbone_cfg)
        self.proj_head = ProjectionHead(**head_cfg)
        self.classifier = nn.Linear(
            head_cfg.get("output_dim", 256), 1, bias=True
        )

        self.train_dataset = train_dataset

        # Class-weighted BCE loss
        # pos_weight must be a registered buffer so Lightning moves it to the
        # correct device (GPU) automatically via .to(device). If stored as a
        # plain tensor inside BCEWithLogitsLoss it stays on CPU and causes a
        # device mismatch during the first training step.
        pos_weight: Optional[torch.Tensor] = None
        if class_weight_mode == "inverse_freq" and train_dataset is not None:
            weights = train_dataset.class_weights()  # [w_neg, w_pos]
            pos_weight = (weights[1] / weights[0]).unsqueeze(0)  # scalar → (1,)

        # Registered buffer: Lightning moves it to the right device on .to() / .cuda()
        self.register_buffer("pos_weight", pos_weight)  # None or (1,) float
        self.label_smoothing = label_smoothing

        # Metrics (reset per epoch automatically via torchmetrics)
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding (no classifier head).

        Used during embedding extraction in inference mode.

        Args:
            mel: (B, 1, n_mels, T) float tensor.

        Returns:
            (B, 256) L2-normalized embedding tensor.
        """
        features = self.backbone(mel)
        return self.proj_head(features)

    def _forward_logits(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both embedding and classifier logit.

        Args:
            mel: (B, 1, n_mels, T) float tensor.

        Returns:
            Tuple of (embedding (B,256), logit (B,1)).
        """
        features = self.backbone(mel)
        embedding = self.proj_head(features)
        logit = self.classifier(embedding)
        return embedding, logit

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Single training step with AMP-compatible loss computation.

        Args:
            batch: Dict with 'mel' (B,1,n_mels,T), 'label' (B,), and metadata.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        mel: torch.Tensor = batch["mel"]
        labels: torch.Tensor = batch["label"].float()

        # Label smoothing: push targets away from hard 0/1
        eps = self.label_smoothing
        smooth_labels = labels * (1.0 - eps) + 0.5 * eps

        _, logits = self._forward_logits(mel)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), smooth_labels, pos_weight=self.pos_weight
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> None:
        """Single validation step; accumulates metrics for epoch-end logging.

        Args:
            batch: Dict with 'mel', 'label', and metadata.
            batch_idx: Index of the current batch.
        """
        mel: torch.Tensor = batch["mel"]
        labels: torch.Tensor = batch["label"]

        _, logits = self._forward_logits(mel)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), labels.float(), pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs >= 0.5).long()

        self.val_auroc.update(probs, labels)
        self.val_f1.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log aggregated validation metrics at the end of each epoch."""
        self.log("val/auc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute())
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall", self.val_recall.compute())

    # ------------------------------------------------------------------
    # Optimizers and schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with cosine annealing + linear warmup.

        Returns:
            Dict compatible with Lightning's optimizer/scheduler API.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear warmup over warmup_epochs, then cosine anneal to zero
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with weighted random sampling."""
        assert self.train_dataset is not None, "train_dataset must be provided"
        self.train_dataset.augmenter.train()

        # Weighted sampler to counteract class imbalance
        sample_weights = self._sample_weights(self.train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
        )

    @staticmethod
    def _sample_weights(dataset: TTMAudioDataset) -> list[float]:
        """Compute per-sample weights inversely proportional to class frequency."""
        class_weights = dataset.class_weights()  # [w_neg, w_pos]
        return [float(class_weights[s["label"]]) for s in dataset.samples]
