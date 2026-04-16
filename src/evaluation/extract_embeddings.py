"""
extract_embeddings.py: extracts T×256 audio embeddings for all clips.

This is the handoff file to the BiLSTM team (S6/S7).
Output contract:
  - One .pt file per (clip_uid, face_id) pair
  - Shape: (T, 256) where T = number of temporal windows
  - manifest.csv: clip_uid, face_id, label, pt_path, duration_sec, n_windows

Run after training completes:
  python scripts/train_audio_cnn.py  (saves best checkpoint)
  python src/evaluation/extract_embeddings.py --ckpt outputs/checkpoints/best.ckpt
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.ttm_audio_dataset import TTMAudioDataset
from src.models.audio_cnn import AudioCNN
from src.models.projection_head import ProjectionHead


@torch.no_grad()
def extract(
    ckpt_path: Path,
    annotation_path: Path,
    mel_cache_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    backbone   = AudioCNN(pretrained=False).to(device)
    proj_head  = ProjectionHead(input_dim=256, output_dim=256).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Lightning checkpoints store state_dict under 'state_dict' key
    state = ckpt.get("state_dict", ckpt)
    backbone.load_state_dict(
        {k.replace("backbone.", ""): v for k, v in state.items() if "backbone" in k}
    )
    proj_head.load_state_dict(
        {k.replace("proj_head.", ""): v for k, v in state.items() if "proj_head" in k}
    )

    backbone.eval()
    proj_head.eval()

    dataset = TTMAudioDataset(
        annotation_path=annotation_path,
        mel_cache_dir=mel_cache_dir,
        split="val",
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    manifest_rows: list[dict] = []

    for sample in tqdm(loader, desc="Extracting embeddings"):
        mel       = sample["mel"].to(device)        # (1, 1, n_mels, T_frames)
        label     = sample["label"].item()
        clip_uid  = sample["clip_uid"][0]
        face_id   = sample["face_id"].item()
        duration  = sample["duration_sec"].item()

        # mel shape: (1, 1, 128, T_frames)
        # Slide window across T dimension to produce T_windows frames
        # AudioCNN processes one window at a time -> (T_windows, 256)
        mel = mel.squeeze(0)  # (1, 128, T_frames)
        windows = _sliding_windows(mel, window_size=80, stride=16)  # (T_w, 1, 128, 80)
        windows = windows.to(device)

        feats = backbone(windows)          # (T_w, 256)
        embs  = proj_head(feats)           # (T_w, 256)
        embs  = F.normalize(embs, dim=-1)  # L2 normalize

        out_path = output_dir / f"{clip_uid}_{face_id}.pt"
        torch.save(embs.cpu(), out_path)

        manifest_rows.append({
            "clip_uid":     clip_uid,
            "face_id":      face_id,
            "label":        label,
            "pt_path":      str(out_path),
            "duration_sec": round(duration, 3),
            "n_windows":    embs.shape[0],
        })

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} embeddings saved to {output_dir}")
    print(f"Manifest: {manifest_path}")


def _sliding_windows(
    mel: torch.Tensor,    # (1, n_mels, T_frames)
    window_size: int = 80,
    stride: int = 16,
) -> torch.Tensor:
    """Slice a Mel spectrogram into overlapping windows."""
    T = mel.shape[-1]
    starts = list(range(0, max(1, T - window_size + 1), stride))
    windows = []
    for s in starts:
        end = s + window_size
        win = mel[:, :, s:end]
        if win.shape[-1] < window_size:
            # Pad last window
            pad = window_size - win.shape[-1]
            win = F.pad(win, (0, pad))
        windows.append(win)
    return torch.stack(windows)  # (T_w, 1, n_mels, window_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        default=Path("~/ego4d_data/annotations/av_train.json"))
    parser.add_argument("--mel_cache",   type=Path,
                        default=Path("outputs/mel_cache"))
    parser.add_argument("--output_dir",  type=Path,
                        default=Path("outputs/embeddings"))
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    extract(
        ckpt_path=args.ckpt,
        annotation_path=args.annotations.expanduser(),
        mel_cache_dir=args.mel_cache,
        output_dir=args.output_dir,
        device=args.device,
    )
