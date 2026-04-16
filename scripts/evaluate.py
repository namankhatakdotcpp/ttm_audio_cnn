"""evaluate.py: comprehensive evaluation of a trained TTM audio CNN checkpoint.

Runs the model on val and test splits, computes a full suite of binary
classification metrics, prints a formatted table, and saves results to JSON.

Usage:
    python scripts/evaluate.py \\
        checkpoint_path=outputs/checkpoints/<run>/epoch=XX-val_auc=0.XXXX.ckpt \\
        annotation_path=~/ego4d_data/v2/annotations/av_train.json \\
        video_dir=~/ego4d_data/v2/full_scale

Outputs:
    outputs/eval/<run_tag>/metrics.json
    outputs/eval/<run_tag>/pr_curve_{split}.pt     (precision-recall arrays)
    outputs/eval/<run_tag>/roc_curve_{split}.pt    (fpr, tpr arrays)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryROC,
    BinaryRecall,
)
from tqdm import tqdm

from src.data.ttm_audio_dataset import TTMAudioDataset
from src.training.lightning_module import AudioCNNModule

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-split evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    model: AudioCNNModule,
    dataset: TTMAudioDataset,
    batch_size: int,
    num_workers: int,
    device: str,
    split_name: str,
    output_dir: Path,
) -> dict[str, float]:
    """Run inference on one split and return a metrics dict.

    Args:
        model: Loaded AudioCNNModule in eval mode.
        dataset: TTMAudioDataset for the split.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        device: Target device.
        split_name: "val" or "test" (used for logging).
        output_dir: Directory to save curve tensors.

    Returns:
        Dict mapping metric name → float value.
    """
    dataset.augmenter.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Instantiate metrics on the target device
    auc    = BinaryAUROC().to(device)
    acc    = BinaryAccuracy().to(device)
    f1     = BinaryF1Score().to(device)
    prec   = BinaryPrecision().to(device)
    rec    = BinaryRecall().to(device)
    pr_curve  = BinaryPrecisionRecallCurve().to(device)
    roc_curve = BinaryROC().to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {split_name}", unit="batch"):
            mels: torch.Tensor = batch["mel"].to(device)
            labels: torch.Tensor = batch["label"].to(device)

            # Forward: backbone + projection + classifier
            feats = model.backbone(mels)
            emb   = model.proj_head(feats)
            logit = model.classifier(emb).squeeze(1)        # (B,)
            probs = torch.sigmoid(logit)

            auc.update(probs, labels)
            acc.update(probs, labels)
            f1.update(probs, labels)
            prec.update(probs, labels)
            rec.update(probs, labels)
            pr_curve.update(probs, labels)
            roc_curve.update(probs, labels)

    # Compute scalar metrics
    metrics = {
        f"{split_name}/auc":       float(auc.compute()),
        f"{split_name}/accuracy":  float(acc.compute()),
        f"{split_name}/f1":        float(f1.compute()),
        f"{split_name}/precision": float(prec.compute()),
        f"{split_name}/recall":    float(rec.compute()),
    }

    # Save curves for plotting
    precision_vals, recall_vals, _ = pr_curve.compute()
    fpr_vals, tpr_vals, roc_thresholds = roc_curve.compute()

    torch.save(
        {"precision": precision_vals.cpu(), "recall": recall_vals.cpu()},
        output_dir / f"pr_curve_{split_name}.pt",
    )
    torch.save(
        {"fpr": fpr_vals.cpu(), "tpr": tpr_vals.cpu()},
        output_dir / f"roc_curve_{split_name}.pt",
    )

    # Optimal threshold via Youden's J statistic (maximise TPR − FPR)
    j_stat = tpr_vals - fpr_vals
    best_idx = int(j_stat.argmax().item())
    # roc thresholds have len = len(fpr) - 1 edge case; guard it
    if best_idx < len(roc_thresholds):
        metrics[f"{split_name}/optimal_threshold"] = float(roc_thresholds[best_idx])

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../configs", config_name="audio/train")
def main(cfg: DictConfig) -> None:
    """Evaluate a checkpoint on val and test splits.

    Required Hydra overrides:
        checkpoint_path=<path-to-.ckpt>
        annotation_path=<path-to-json>
        video_dir=<path-to-full_scale>

    Optional:
        eval_splits=[val,test]   (default: both)
        eval_batch_size=128
    """
    checkpoint_path = Path(cfg.checkpoint_path)
    annotation_path = Path(cfg.annotation_path)
    video_dir       = Path(cfg.video_dir)
    mel_cache_dir   = Path(cfg.mel_cache_dir)
    output_dir      = Path(cfg.output_dir) / "eval" / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    mel_cfg = dict(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        clip_norm_range=cfg.clip_norm_range,
    )

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = AudioCNNModule.load_from_checkpoint(
        str(checkpoint_path), map_location=device
    )
    model.eval()
    model.to(device)

    batch_size  = int(cfg.get("eval_batch_size", cfg.batch_size))
    num_workers = int(cfg.get("eval_num_workers", cfg.num_workers))
    splits      = list(cfg.get("eval_splits", ["val", "test"]))

    all_metrics: dict[str, float] = {}

    for split in splits:
        dataset = TTMAudioDataset(
            annotation_path=annotation_path,
            mel_cache_dir=mel_cache_dir,
            split=split,
            video_dir=video_dir,
            mel_cfg=mel_cfg,
            write_cache=False,
        )
        if len(dataset) == 0:
            log.warning("Split '%s' has 0 samples — skipping.", split)
            continue

        log.info("Evaluating split '%s' (%d samples) …", split, len(dataset))
        metrics = evaluate_split(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            split_name=split,
            output_dir=output_dir,
        )
        all_metrics.update(metrics)

    # ── Print table ──────────────────────────────────────────────────
    _print_metrics_table(all_metrics, splits)

    # ── Save JSON ────────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved to: %s", metrics_path)


def _print_metrics_table(metrics: dict[str, float], splits: list[str]) -> None:
    """Print a formatted metrics table to stdout."""
    cols = ["auc", "accuracy", "f1", "precision", "recall", "optimal_threshold"]
    col_w = 12

    header = f"{'metric':<24}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for split in splits:
        row = f"  {split:<22}"
        for col in cols:
            key = f"{split}/{col}"
            val = metrics.get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
    print(sep + "\n")


if __name__ == "__main__":
    main()
