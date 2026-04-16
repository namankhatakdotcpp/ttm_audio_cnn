"""TTMAudioDataModule: Lightning DataModule for the Ego4D TTM audio pipeline.

Separates data-loading concerns from model training. Instantiate this once,
pass it to trainer.fit(model, datamodule=dm), and Lightning handles the rest.

By using a DataModule instead of embedding DataLoaders in the LightningModule:
  - Checkpoints load cleanly (no dataset state in the module's hparams)
  - Data setup is called exactly once, even with DDP
  - Augmentation train/eval modes are set by Lightning at the right time
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.ttm_audio_dataset import TTMAudioDataset


class TTMAudioDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping train/val/test TTMAudioDatasets.

    Role in pipeline:
        Annotation JSON + MP4s → TTMAudioDataModule → DataLoaders
        → AudioCNNModule.training_step / validation_step

    Args:
        annotation_path: Path to the Ego4D TTM annotation JSON file.
        video_dir: Root directory of MP4 video files.
        mel_cache_dir: Directory for cached Mel .pt files.
        mel_cfg: Kwargs forwarded to MelExtractor (sample_rate, n_fft, …).
        augment_cfg: Kwargs forwarded to AudioAugmentPipeline (train split only).
        batch_size: DataLoader batch size for all splits.
        num_workers: DataLoader worker count.
        pin_memory: DataLoader pin_memory flag.
        max_frames: Pad/truncate target. None = no padding (variable length).
        write_cache: Whether on-the-fly extractions are written to mel_cache_dir.
    """

    def __init__(
        self,
        annotation_path: Path,
        video_dir: Path,
        mel_cache_dir: Path,
        mel_cfg: Optional[dict[str, Any]] = None,
        augment_cfg: Optional[dict[str, Any]] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        max_frames: Optional[int] = None,
        write_cache: bool = True,
    ) -> None:
        super().__init__()
        self.annotation_path = Path(annotation_path)
        self.video_dir = Path(video_dir)
        self.mel_cache_dir = Path(mel_cache_dir)
        self.mel_cfg = mel_cfg or {}
        self.augment_cfg = augment_cfg or {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_frames = max_frames
        self.write_cache = write_cache

        self._train_ds: Optional[TTMAudioDataset] = None
        self._val_ds: Optional[TTMAudioDataset] = None
        self._test_ds: Optional[TTMAudioDataset] = None

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate datasets for the requested stage.

        Called by Lightning once per process. `stage` is one of
        "fit", "validate", "test", or "predict".

        Args:
            stage: Lightning stage string.
        """
        if stage in ("fit", None):
            self._train_ds = TTMAudioDataset(
                annotation_path=self.annotation_path,
                mel_cache_dir=self.mel_cache_dir,
                split="train",
                video_dir=self.video_dir,
                mel_cfg=self.mel_cfg,
                augment_cfg=self.augment_cfg,
                max_frames=self.max_frames,
                write_cache=self.write_cache,
            )
            self._val_ds = TTMAudioDataset(
                annotation_path=self.annotation_path,
                mel_cache_dir=self.mel_cache_dir,
                split="val",
                video_dir=self.video_dir,
                mel_cfg=self.mel_cfg,
                max_frames=self.max_frames,
                write_cache=self.write_cache,
            )

        if stage in ("test", "predict"):
            self._test_ds = TTMAudioDataset(
                annotation_path=self.annotation_path,
                mel_cache_dir=self.mel_cache_dir,
                split="test",
                video_dir=self.video_dir,
                mel_cfg=self.mel_cfg,
                max_frames=self.max_frames,
                write_cache=self.write_cache,
            )

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader with class-balanced weighted random sampler."""
        assert self._train_ds is not None, "Call setup('fit') first."

        # Augmentation is enabled during training
        self._train_ds.augmenter.train()

        weights = self._train_ds.class_weights()           # [w_neg, w_pos]
        sample_weights = [
            float(weights[s["label"]]) for s in self._train_ds.samples
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader (no augmentation, no shuffling)."""
        assert self._val_ds is not None, "Call setup('fit') first."
        self._val_ds.augmenter.eval()

        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        """Test DataLoader (no augmentation, no shuffling)."""
        assert self._test_ds is not None, "Call setup('test') first."
        self._test_ds.augmenter.eval()

        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def train_dataset(self) -> TTMAudioDataset:
        """Return the training dataset (available after setup('fit'))."""
        assert self._train_ds is not None, "Call setup('fit') first."
        return self._train_ds

    @property
    def val_dataset(self) -> TTMAudioDataset:
        """Return the validation dataset (available after setup('fit'))."""
        assert self._val_ds is not None, "Call setup('fit') first."
        return self._val_ds
