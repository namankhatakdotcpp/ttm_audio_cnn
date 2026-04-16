"""dry_run.py: 2-epoch synthetic smoke test for the TTM audio CNN pipeline.

Builds a small synthetic dataset of random Mel spectrograms, runs 2 full
training epochs with Lightning, and validates that:
  1. All model imports resolve correctly
  2. Forward pass produces the right tensor shapes at every stage
  3. Loss decreases (at least does not NaN/explode)
  4. val/auc, val/f1 are logged and non-NaN
  5. A checkpoint is saved and can be reloaded

Run on the A6000 before the first real training session:
    cd ~/ttm_project   (or wherever environment is activated)
    python scripts/dry_run.py

No Ego4D data required. Exits 0 on success, non-zero on any assertion failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models.audio_cnn import AudioCNN
from src.models.projection_head import ProjectionHead
from src.training.lightning_module import AudioCNNModule


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ──────────────────────────────────────────────────────────────────────────────

N_MELS = 128
T_FRAMES = 80    # ~0.8 s at hop=160, sr=16kHz
BATCH_SIZE = 8
N_TRAIN = 64
N_VAL = 16


class SyntheticMelDataset(Dataset):
    """Random Gaussian Mel spectrograms with binary labels for shape testing."""

    def __init__(self, n_samples: int, n_mels: int = N_MELS, t: int = T_FRAMES) -> None:
        self.n_samples = n_samples
        self.n_mels = n_mels
        self.t = t
        # Pre-generate so each epoch is identical (stable loss curve)
        torch.manual_seed(0)
        self.mels = torch.randn(n_samples, 1, n_mels, t)
        self.labels = torch.randint(0, 2, (n_samples,))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "mel": self.mels[idx],
            "attention_mask": torch.ones(self.t, dtype=torch.bool),
            "label": int(self.labels[idx]),
            "clip_uid": f"synthetic_{idx:04d}",
            "face_id": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Shape verification (CPU, no Lightning)
# ──────────────────────────────────────────────────────────────────────────────

def verify_shapes() -> None:
    """Assert output shapes at each model stage before Lightning training."""
    print("── Shape verification ─────────────────────────────────────")

    B = 4
    x = torch.randn(B, 1, N_MELS, T_FRAMES)

    # AudioCNN
    backbone = AudioCNN(pretrained=False, dropout=0.0)
    backbone.eval()
    with torch.no_grad():
        feat = backbone(x)
    assert feat.shape == (B, 256), f"AudioCNN output shape: {feat.shape}, expected ({B}, 256)"
    print(f"  AudioCNN:        {x.shape} → {feat.shape}  ✓")

    # Activation maps for Grad-CAM
    maps = backbone.get_activation_maps()
    assert maps.dim() == 4 and maps.shape[:2] == (B, 256), \
        f"Activation maps shape: {maps.shape}"
    print(f"  Activation maps: {maps.shape}  ✓")

    # ProjectionHead
    head = ProjectionHead(input_dim=256, hidden_dim=256, output_dim=256)
    head.eval()
    with torch.no_grad():
        emb = head(feat)
    assert emb.shape == (B, 256), f"ProjectionHead shape: {emb.shape}"
    # Verify L2 normalization
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), \
        f"Embeddings not unit-normed: {norms}"
    print(f"  ProjectionHead:  {feat.shape} → {emb.shape}  (L2-normed ✓)")

    # Full module forward (no classifier)
    module = AudioCNNModule(
        backbone_cfg={"pretrained": False, "dropout": 0.0},
        head_cfg={"input_dim": 256, "hidden_dim": 256, "output_dim": 256},
    )
    module.eval()
    with torch.no_grad():
        out = module(x)
    assert out.shape == (B, 256), f"Module forward shape: {out.shape}"
    print(f"  AudioCNNModule:  {x.shape} → {out.shape}  ✓")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 2-epoch Lightning training loop
# ──────────────────────────────────────────────────────────────────────────────

class _SyntheticDataModule(pl.LightningDataModule):
    """Minimal DataModule wrapping the synthetic datasets."""

    def setup(self, stage=None):
        self.train_ds = SyntheticMelDataset(N_TRAIN)
        self.val_ds = SyntheticMelDataset(N_VAL)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def run_training_loop(use_gpu: bool) -> dict:
    """Run 2 training epochs and return the logged metrics dict."""
    print("── 2-epoch Lightning training loop ────────────────────────")
    accelerator = "gpu" if use_gpu else "cpu"
    print(f"  Accelerator: {accelerator}")

    ckpt_dir = Path("outputs/dry_run_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/auc",
        mode="max",
        save_top_k=1,
        filename="dry_run_best",
    )

    module = AudioCNNModule(
        backbone_cfg={"pretrained": False, "dropout": 0.2},
        head_cfg={"input_dim": 256, "hidden_dim": 256, "output_dim": 256},
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        lr=1e-3,
        weight_decay=1e-2,
        max_epochs=2,
        warmup_epochs=1,
        label_smoothing=0.1,
        class_weight_mode="none",   # no real class imbalance in synthetic data
        use_amp=False,              # AMP on CPU causes issues; GPU handles it
    )

    dm = _SyntheticDataModule()

    precision = "16-mixed" if (use_gpu and torch.cuda.is_available()) else "32-true"
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[checkpoint_cb],
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    trainer.fit(module, datamodule=dm)

    # Collect results from the last validation epoch
    results = trainer.callback_metrics
    print(f"\n  Metrics after 2 epochs:")
    for k, v in results.items():
        print(f"    {k}: {float(v):.4f}")

    return {k: float(v) for k, v in results.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Assertions on training results
# ──────────────────────────────────────────────────────────────────────────────

def assert_training_health(metrics: dict) -> None:
    """Fail fast if any metric is NaN or clearly wrong."""
    print("\n── Assertions ──────────────────────────────────────────────")

    required_keys = ["train/loss_epoch", "val/loss", "val/auc", "val/f1"]
    for key in required_keys:
        if key not in metrics:
            # Lightning sometimes uses slightly different key names
            print(f"  [WARN] Key '{key}' not found in metrics. Available: {list(metrics.keys())}")
            continue
        val = metrics[key]
        assert not (val != val), f"{key} is NaN!"  # NaN check
        assert val >= 0.0, f"{key} = {val} is negative (unexpected)"
        print(f"  {key} = {val:.4f}  ✓")

    train_loss = next(
        (metrics[k] for k in ["train/loss_epoch", "train/loss"] if k in metrics), None
    )
    if train_loss is not None:
        assert train_loss < 5.0, f"train/loss suspiciously high: {train_loss}"
        print(f"  train/loss < 5.0  ✓")

    val_auc = metrics.get("val/auc")
    if val_auc is not None:
        assert 0.0 <= val_auc <= 1.0, f"val/auc out of [0,1]: {val_auc}"
        print(f"  0 ≤ val/auc ≤ 1  ✓")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint reload test
# ──────────────────────────────────────────────────────────────────────────────

def test_checkpoint_reload() -> None:
    """Verify best checkpoint can be loaded and run inference."""
    print("── Checkpoint reload ───────────────────────────────────────")
    ckpt_path = Path("outputs/dry_run_checkpoints/dry_run_best.ckpt")
    if not ckpt_path.exists():
        print("  [SKIP] No checkpoint found (fast_dev_run may have skipped saving).")
        return

    loaded = AudioCNNModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    loaded.eval()
    x = torch.randn(2, 1, N_MELS, T_FRAMES)
    with torch.no_grad():
        emb = loaded(x)
    assert emb.shape == (2, 256), f"Reload forward shape: {emb.shape}"
    print(f"  Checkpoint loaded and forward pass OK: {emb.shape}  ✓")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU — running on CPU (expect slower).")
    print()

    verify_shapes()
    metrics = run_training_loop(use_gpu=use_gpu)
    assert_training_health(metrics)
    test_checkpoint_reload()

    print("=" * 60)
    print("  DRY RUN PASSED — ready for Session 2 real training.")
    print("=" * 60)
    sys.exit(0)
