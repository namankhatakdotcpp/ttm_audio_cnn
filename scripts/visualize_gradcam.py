"""visualize_gradcam.py: generate Grad-CAM overlays for TTM audio clips.

Loads a trained checkpoint, samples N clips from the val split (or a
user-specified list), computes Grad-CAM heatmaps, and saves side-by-side
figures: original Mel spectrogram (left) + CAM overlay (right).

Also saves a summary PNG grid of all selected samples for quick inspection.

Usage:
    # Default: 8 random val clips
    python scripts/visualize_gradcam.py \\
        checkpoint_path=outputs/checkpoints/<run>/best.ckpt \\
        annotation_path=~/ego4d_data/v2/annotations/av_train.json \\
        video_dir=~/ego4d_data/v2/full_scale

    # Specific clip + face
    python scripts/visualize_gradcam.py \\
        checkpoint_path=... \\
        clip_uids=[abc123,def456] \\
        face_ids=[0,1]

Output:
    outputs/gradcam/<run_tag>/sample_<n>_<uid>_<face>.png  — individual figures
    outputs/gradcam/<run_tag>/summary_grid.png              — composite grid
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import matplotlib
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")

from src.data.ttm_audio_dataset import TTMAudioDataset
from src.evaluation.gradcam_visualizer import GradCAMVisualizer
from src.training.lightning_module import AudioCNNModule

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="audio/train")
def main(cfg: DictConfig) -> None:
    """Generate Grad-CAM visualizations from a trained checkpoint.

    Hydra config fields used (all overrideable on CLI):
        checkpoint_path, annotation_path, video_dir, mel_cache_dir,
        output_dir, n_gradcam_samples, gradcam_class_idx,
        clip_uids (optional list), face_ids (optional list)
    """
    checkpoint_path = Path(cfg.checkpoint_path)
    annotation_path = Path(cfg.annotation_path)
    video_dir       = Path(cfg.video_dir)
    mel_cache_dir   = Path(cfg.mel_cache_dir)
    output_dir      = Path(cfg.output_dir) / "gradcam" / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples:   int = int(cfg.get("n_gradcam_samples", 8))
    class_idx:   int = int(cfg.get("gradcam_class_idx", 1))  # 1 = TTM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    mel_cfg = dict(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        clip_norm_range=cfg.clip_norm_range,
    )

    # ── Load model ────────────────────────────────────────────────────
    log.info("Loading checkpoint: %s", checkpoint_path)
    module = AudioCNNModule.load_from_checkpoint(
        str(checkpoint_path), map_location=device
    )
    module.eval()
    module.to(device)

    # GradCAMVisualizer hooks onto module.backbone (the AudioCNN)
    visualizer = GradCAMVisualizer(module.backbone)

    # ── Load dataset ──────────────────────────────────────────────────
    dataset = TTMAudioDataset(
        annotation_path=annotation_path,
        mel_cache_dir=mel_cache_dir,
        split="val",
        video_dir=video_dir,
        mel_cfg=mel_cfg,
        write_cache=False,
    )
    dataset.augmenter.eval()

    # ── Select samples ────────────────────────────────────────────────
    # User-specified clip list takes precedence
    requested_uids: Optional[list[str]] = list(cfg.get("clip_uids", [])) or None
    requested_fids: Optional[list[int]] = (
        [int(x) for x in cfg.get("face_ids", [])] or None
    )

    if requested_uids:
        indices = _find_indices(dataset, requested_uids, requested_fids)
    else:
        # Balanced sample: half TTM=1, half TTM=0
        pos_idx = [i for i, s in enumerate(dataset.samples) if s["label"] == 1]
        neg_idx = [i for i, s in enumerate(dataset.samples) if s["label"] == 0]
        half = max(1, n_samples // 2)
        random.seed(42)
        sel_pos = random.sample(pos_idx, min(half, len(pos_idx)))
        sel_neg = random.sample(neg_idx, min(half, len(neg_idx)))
        indices = (sel_pos + sel_neg)[:n_samples]

    log.info("Generating Grad-CAM for %d clips (class_idx=%d) …", len(indices), class_idx)

    # ── Generate per-sample figures ───────────────────────────────────
    grid_mels:  list[torch.Tensor]   = []
    grid_cams:  list[torch.Tensor]   = []
    grid_labels: list[str]           = []

    for rank, idx in enumerate(tqdm(indices, unit="clip")):
        item = dataset[idx]
        mel: torch.Tensor = item["mel"].unsqueeze(0).to(device)  # (1,1,n_mels,T)
        label_str = "TTM=1" if item["label"] == 1 else "TTM=0"
        clip_uid  = item["clip_uid"]
        face_id   = item["face_id"]

        heatmap = visualizer.generate(
            mel_tensor=mel,
            class_idx=class_idx,
            classifier_head=module.classifier,
        )  # (n_mels, T) numpy array

        fig_path = output_dir / f"sample_{rank:02d}_{clip_uid}_{face_id}.png"
        visualizer.save_figure(fig_path)

        grid_mels.append(item["mel"].squeeze(0))  # (n_mels, T)
        grid_cams.append(torch.from_numpy(heatmap))
        grid_labels.append(f"{label_str}\n{clip_uid[:8]}…")

    # ── Summary grid ─────────────────────────────────────────────────
    _save_summary_grid(
        mels=grid_mels,
        cams=grid_cams,
        labels=grid_labels,
        path=output_dir / "summary_grid.png",
    )

    log.info("Figures saved to: %s", output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_indices(
    dataset: TTMAudioDataset,
    clip_uids: list[str],
    face_ids: Optional[list[int]],
) -> list[int]:
    """Return dataset indices matching the requested clip_uid/face_id pairs."""
    uid_to_pos = {uid: pos for pos, uid in enumerate(clip_uids)}
    indices: list[int] = []
    for i, s in enumerate(dataset.samples):
        pos = uid_to_pos.get(s["clip_uid"])
        if pos is None:
            continue
        if face_ids is None or s["face_id"] == face_ids[pos]:
            indices.append(i)
    if not indices:
        log.warning("No matching clips found — falling back to first 8 samples.")
        indices = list(range(min(8, len(dataset))))
    return indices


def _save_summary_grid(
    mels: list[torch.Tensor],
    cams: list[torch.Tensor],
    labels: list[str],
    path: Path,
) -> None:
    """Save a (2 × N) grid: top row = spectrograms, bottom row = CAM overlays."""
    n = len(mels)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 5))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, (mel, cam, lbl) in enumerate(zip(mels, cams, labels)):
        mel_np = mel.float().numpy()
        mel_min, mel_max = mel_np.min(), mel_np.max()
        mel_disp = (mel_np - mel_min) / (mel_max - mel_min + 1e-6)

        # Top: raw spectrogram
        axes[0][col].imshow(mel_disp, origin="lower", aspect="auto", cmap="magma")
        axes[0][col].set_title(lbl, fontsize=7)
        axes[0][col].axis("off")

        # Bottom: CAM overlay
        axes[1][col].imshow(mel_disp, origin="lower", aspect="auto", cmap="gray")
        axes[1][col].imshow(cam.numpy(), origin="lower", aspect="auto",
                             cmap="jet", alpha=0.5)
        axes[1][col].axis("off")

    axes[0][0].set_ylabel("Mel", fontsize=7)
    axes[1][0].set_ylabel("CAM", fontsize=7)
    plt.suptitle("Grad-CAM Summary (top: Mel, bottom: CAM overlay)", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Summary grid saved: %s", path)


if __name__ == "__main__":
    main()
