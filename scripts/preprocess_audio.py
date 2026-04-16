"""preprocess_audio.py: pre-compute and cache all Mel spectrograms.

Reads the TTM annotation JSON, extracts audio segments from each MP4
using torchaudio.load (no subprocess / ffmpeg), and saves the computed
Mel spectrogram to outputs/mel_cache/{clip_uid}_{face_id}.pt.

Design principles:
  - Resumable: skips clips whose .pt cache already exists.
  - Pure torchaudio: no external ffmpeg subprocess calls.
  - All paths via pathlib.Path, no os.path.
  - Logs dataset statistics at the end.

Run via Hydra:
    python scripts/preprocess_audio.py \\
        annotation_path=/path/to/av_train.json \\
        video_dir=/path/to/full_scale \\
        mel_cache_dir=outputs/mel_cache

Or use the provided defaults in configs/audio/mel.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm

# Ensure the src package is importable when running from the project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mel_extractor import MelExtractor

log = logging.getLogger(__name__)


def _find_video(video_dir: Path, video_uid: str, clip_uid: str) -> Optional[Path]:
    """Locate an MP4 file using common Ego4D directory layouts."""
    candidates = [
        video_dir / f"{clip_uid}.mp4",
        video_dir / f"{video_uid}.mp4",
        video_dir / clip_uid / f"{clip_uid}.mp4",
        video_dir / video_uid / f"{video_uid}.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_annotations(annotation_path: Path, splits: list[str]) -> list[dict[str, Any]]:
    """Load and flatten annotation entries for the requested splits."""
    with open(annotation_path, "r") as f:
        data = json.load(f)

    records: list[dict[str, Any]] = []
    videos = data.get("videos", data.get("clips", []))

    for video in videos:
        video_uid = video.get("video_uid", video.get("clip_uid", ""))
        for clip in video.get("clips", [video]):
            clip_uid: str = clip.get("clip_uid", video_uid)
            clip_split: str = clip.get("split", "train")

            if clip_split not in splits:
                continue

            start_sec = float(clip.get("clip_start_sec", 0.0))
            end_sec = float(clip.get("clip_end_sec", 0.0))

            for ann in clip.get("annotation", clip.get("annotations", [])):
                raw_label = ann.get("label", ann.get("ttm_label", "NO"))
                label = 1 if str(raw_label).upper() in ("YES", "1", "TRUE") else 0
                face_id = int(ann.get("person_id", ann.get("face_id", 0)))

                records.append(
                    {
                        "clip_uid": clip_uid,
                        "video_uid": video_uid,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "label": label,
                        "face_id": face_id,
                    }
                )

    return records


@hydra.main(version_base="1.3", config_path="../configs", config_name="audio/mel")
def main(cfg: DictConfig) -> None:
    """Entry point: extract and cache all Mel spectrograms.

    Hydra config fields used:
      cfg.annotation_path, cfg.video_dir, cfg.mel_cache_dir,
      cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels,
      cfg.f_min, cfg.f_max, cfg.clip_norm_range
    """
    annotation_path = Path(cfg.annotation_path)
    video_dir = Path(cfg.video_dir)
    mel_cache_dir = Path(cfg.mel_cache_dir)
    mel_cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = MelExtractor(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        clip_norm_range=cfg.clip_norm_range,
    )

    splits = list(cfg.get("splits", ["train", "val", "test"]))
    records = _load_annotations(annotation_path, splits)
    log.info("Loaded %d annotation entries across splits %s", len(records), splits)

    # Statistics counters
    n_cached = n_extracted = n_skipped = n_missing = 0
    durations: list[float] = []
    label_counts = {0: 0, 1: 0}

    for rec in tqdm(records, desc="Preprocessing audio", unit="clip"):
        clip_uid: str = rec["clip_uid"]
        face_id: int = rec["face_id"]
        label: int = rec["label"]
        start_sec: float = rec["start_sec"]
        end_sec: float = rec["end_sec"]
        duration = end_sec - start_sec

        cache_path = mel_cache_dir / f"{clip_uid}_{face_id}.pt"

        # Skip already-cached files (resumable)
        if cache_path.exists():
            n_cached += 1
            label_counts[label] += 1
            durations.append(duration)
            continue

        video_path = _find_video(video_dir, rec["video_uid"], clip_uid)
        if video_path is None:
            log.warning("MP4 not found for clip_uid=%s — skipping", clip_uid)
            n_missing += 1
            continue

        try:
            frame_offset = int(start_sec * extractor.sample_rate)
            num_frames = int(duration * extractor.sample_rate)

            waveform, sr = torchaudio.load(
                str(video_path),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )

            if sr != extractor.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, extractor.sample_rate)
                waveform = resampler(waveform)

            # Downmix to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            mel = extractor(waveform)   # (1, n_mels, T_frames)
            torch.save(mel, cache_path)

            n_extracted += 1
            label_counts[label] += 1
            durations.append(duration)

        except Exception as exc:
            log.warning("Failed to process %s: %s", clip_uid, exc)
            n_skipped += 1

    # ----------------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------------
    total = n_cached + n_extracted
    log.info("=" * 60)
    log.info("Preprocessing complete")
    log.info("  Already cached:  %d", n_cached)
    log.info("  Newly extracted: %d", n_extracted)
    log.info("  Skipped (error): %d", n_skipped)
    log.info("  Missing MP4:     %d", n_missing)
    log.info("  Total processed: %d", total)

    if durations:
        import statistics
        log.info("Duration stats (seconds):")
        log.info("  min=%.2f  max=%.2f  mean=%.2f  median=%.2f",
                 min(durations), max(durations),
                 sum(durations) / len(durations),
                 statistics.median(durations))

    n_pos = label_counts[1]
    n_neg = label_counts[0]
    log.info("Class balance: TTM=1: %d (%.1f%%)  TTM=0: %d (%.1f%%)",
             n_pos, 100.0 * n_pos / max(total, 1),
             n_neg, 100.0 * n_neg / max(total, 1))
    log.info("Cache directory: %s", mel_cache_dir)


if __name__ == "__main__":
    main()
