"""TTMAudioDataset: PyTorch Dataset for the Ego4D TTM audio benchmark.

Reads the Ego4D V2 AV/TTM annotation JSON, resolves each clip's audio,
and returns normalized Mel spectrograms with labels and metadata.

Cache strategy:
  - If outputs/mel_cache/{clip_uid}_{face_id}.pt exists: load directly.
  - Otherwise: extract audio from the source MP4, run MelExtractor on-the-fly,
    and (optionally) write to cache for subsequent epochs.

Variable-length clips are handled with zero-padding to a fixed temporal length
and an accompanying boolean attention mask (True = valid frame, False = pad).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from src.data.augmentations import AudioAugmentPipeline
from src.data.mel_extractor import MelExtractor


# Type alias for a single dataset item
DataItem = dict[str, Any]


class TTMAudioDataset(Dataset):
    """PyTorch Dataset for the Ego4D TTM audio classification task.

    Role in pipeline:
        Annotation JSON + MP4s/cache → TTMAudioDataset → DataLoader
        → AudioCNNModule

    Each item contains:
      - mel:          Tensor (1, n_mels, T_max) — padded Mel spectrogram
      - attention_mask: BoolTensor (T_max,)    — True where frames are valid
      - label:        int (0 = not TTM, 1 = TTM)
      - clip_uid:     str
      - face_id:      int

    Args:
        annotation_path: Path to the Ego4D TTM annotation JSON file.
        video_dir: Root directory containing per-clip MP4 files.
        mel_cache_dir: Directory for cached .pt Mel spectrograms.
        split: One of "train", "val", "test" — used to filter the JSON.
        mel_cfg: Keyword args forwarded to MelExtractor (sample_rate, n_fft, …).
        augment_cfg: Optional keyword args forwarded to AudioAugmentPipeline.
        max_frames: Fixed temporal dimension for padding/truncation.
                    If None, inferred as the 95th-percentile clip length in
                    the split (computed on first pass — slow first epoch).
        write_cache: If True, save on-the-fly extractions to mel_cache_dir.
    """

    def __init__(
        self,
        annotation_path: Path,
        mel_cache_dir: Path,
        split: str,
        video_dir: Path = Path(""),
        mel_cfg: Optional[dict[str, Any]] = None,
        augment_cfg: Optional[dict[str, Any]] = None,
        augment: Optional[bool] = None,
        max_frames: Optional[int] = None,
        write_cache: bool = True,
    ) -> None:
        super().__init__()
        self.video_dir = Path(video_dir)
        self.mel_cache_dir = Path(mel_cache_dir)
        self.mel_cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.write_cache = write_cache

        mel_cfg = mel_cfg or {}
        self.mel_extractor = MelExtractor(**mel_cfg)
        self.sample_rate: int = self.mel_extractor.sample_rate

        # augment=False disables all augmentation regardless of augment_cfg
        augment_cfg = {} if augment is False else (augment_cfg or {})
        self.augmenter = AudioAugmentPipeline(
            sample_rate=self.sample_rate,
            noise_clip_dir=self.mel_cache_dir,  # reuse cached clips as noise
            **augment_cfg,
        )
        if augment is False:
            self.augmenter.eval()

        self.samples: list[dict[str, Any]] = self._load_annotations(annotation_path)
        self.max_frames = max_frames  # resolved lazily if None

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------

    def _load_annotations(self, annotation_path: Path) -> list[dict[str, Any]]:
        """Parse the Ego4D TTM annotation JSON and return a flat sample list.

        The TTM JSON structure (Ego4D V2 AV benchmark) is:
          {
            "videos": [
              {
                "video_uid": "...",
                "clips": [
                  {
                    "clip_uid": "...",
                    "clip_start_sec": float,
                    "clip_end_sec": float,
                    "annotation": [
                      {
                        "label": "YES" | "NO",
                        "person_id": int,
                        ...
                      }
                    ]
                  }
                ]
              }
            ]
          }

        Returns:
            List of dicts with keys: clip_uid, video_uid, start_sec, end_sec,
            label, face_id.
        """
        with open(annotation_path, "r") as f:
            data = json.load(f)

        samples: list[dict[str, Any]] = []

        # Handle both "videos" and direct "clips" top-level keys
        videos = data.get("videos", data.get("clips", []))

        for video in videos:
            video_uid = video.get("video_uid", video.get("clip_uid", ""))
            clips = video.get("clips", [video])  # single-clip entries

            for clip in clips:
                clip_uid: str = clip.get("clip_uid", video_uid)
                clip_split: str = clip.get("split", self.split)

                # Filter to the requested split
                if clip_split != self.split:
                    continue

                start_sec: float = float(clip.get("clip_start_sec", 0.0))
                end_sec: float = float(clip.get("clip_end_sec", 0.0))
                annotations = clip.get("annotation", clip.get("annotations", []))

                for ann in annotations:
                    raw_label = ann.get("label", ann.get("ttm_label", "NO"))
                    label = 1 if str(raw_label).upper() in ("YES", "1", "TRUE") else 0
                    face_id = int(ann.get("person_id", ann.get("face_id", 0)))

                    samples.append(
                        {
                            "clip_uid": clip_uid,
                            "video_uid": video_uid,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "label": label,
                            "face_id": face_id,
                        }
                    )

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DataItem:
        sample = self.samples[idx]
        clip_uid: str = sample["clip_uid"]
        face_id: int = sample["face_id"]
        label: int = sample["label"]

        mel = self._load_or_extract_mel(sample)

        # Pad / truncate to max_frames along the time axis
        max_t = self._get_max_frames(mel)
        mel, mask = self._pad_or_truncate(mel, max_t)

        # Apply spectrogram augmentation (no-op at eval time)
        mel = self.augmenter.augment_spectrogram(mel)

        duration_sec = (sample["end_sec"] - sample["start_sec"])

        return {
            "mel": mel,                    # (1, n_mels, max_t)
            "attention_mask": mask,        # (max_t,) BoolTensor
            "label": label,
            "clip_uid": clip_uid,
            "face_id": face_id,
            "duration_sec": torch.tensor(duration_sec, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Mel loading / extraction
    # ------------------------------------------------------------------

    def _cache_path(self, clip_uid: str, face_id: int) -> Path:
        return self.mel_cache_dir / f"{clip_uid}_{face_id}.pt"

    def _load_or_extract_mel(self, sample: dict[str, Any]) -> torch.Tensor:
        """Return the Mel spectrogram for a sample, using cache when available.

        Args:
            sample: Dict with clip_uid, video_uid, start_sec, end_sec, face_id.

        Returns:
            Mel tensor of shape (1, n_mels, T_frames).
        """
        cache_p = self._cache_path(sample["clip_uid"], sample["face_id"])

        if cache_p.exists():
            mel: torch.Tensor = torch.load(cache_p, weights_only=True)
            return mel

        # On-the-fly extraction
        mel = self._extract_mel(sample)

        if self.write_cache:
            torch.save(mel, cache_p)

        return mel

    def _extract_mel(self, sample: dict[str, Any]) -> torch.Tensor:
        """Extract and return a Mel spectrogram from the source MP4.

        Searches for the video file using several common naming conventions
        used in the Ego4D V2 full_scale directory layout.

        Args:
            sample: Dict with video_uid, clip_uid, start_sec, end_sec.

        Returns:
            Mel tensor of shape (1, n_mels, T_frames).
        """
        video_uid: str = sample["video_uid"]
        clip_uid: str = sample["clip_uid"]
        start_sec: float = sample["start_sec"]
        end_sec: float = sample["end_sec"]

        # Try candidate paths
        candidates = [
            self.video_dir / f"{clip_uid}.mp4",
            self.video_dir / f"{video_uid}.mp4",
            self.video_dir / clip_uid / f"{clip_uid}.mp4",
        ]
        video_path: Optional[Path] = None
        for c in candidates:
            if c.exists():
                video_path = c
                break

        if video_path is None:
            # Return a zero spectrogram so the batch doesn't crash during
            # development when some clips are not yet downloaded
            n_frames = max(1, int((end_sec - start_sec) * self.sample_rate / self.mel_extractor.hop_length))
            return torch.zeros(1, self.mel_extractor.n_mels, n_frames)

        # Load the audio segment using torchaudio (no subprocess / ffmpeg)
        frame_offset = int(start_sec * self.sample_rate)
        num_frames = int((end_sec - start_sec) * self.sample_rate)

        waveform, sr = torchaudio.load(
            str(video_path),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )

        # Resample if the source file is not at the target rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply waveform augmentation (no-op at eval time)
        waveform = self.augmenter.augment_waveform(waveform)

        # Compute Mel spectrogram: (1, n_mels, T_frames)
        mel = self.mel_extractor(waveform)
        return mel

    # ------------------------------------------------------------------
    # Padding helpers
    # ------------------------------------------------------------------

    def _get_max_frames(self, mel: torch.Tensor) -> int:
        """Return the fixed temporal size to pad/truncate to.

        Uses self.max_frames if set; otherwise defaults to the current clip's
        own length (i.e., no padding — useful for inference).
        """
        if self.max_frames is not None:
            return self.max_frames
        return mel.shape[-1]

    def _pad_or_truncate(
        self, mel: torch.Tensor, max_t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad or truncate a Mel spectrogram along the time axis.

        Args:
            mel: (1, n_mels, T) float tensor.
            max_t: Target temporal dimension.

        Returns:
            Tuple of:
              - padded_mel: (1, n_mels, max_t) float tensor.
              - mask: (max_t,) BoolTensor — True for valid frames.
        """
        t = mel.shape[-1]

        if t >= max_t:
            # Truncate
            mel_out = mel[..., :max_t]
            mask = torch.ones(max_t, dtype=torch.bool)
        else:
            # Zero-pad on the right
            pad_amount = max_t - t
            mel_out = F.pad(mel, (0, pad_amount))
            mask = torch.cat([
                torch.ones(t, dtype=torch.bool),
                torch.zeros(pad_amount, dtype=torch.bool),
            ])

        return mel_out, mask

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for the training split.

        Returns:
            FloatTensor of shape (2,) where weight[c] = N / (2 * count[c]).
        """
        labels = [s["label"] for s in self.samples]
        n_total = len(labels)
        n_pos = sum(labels)
        n_neg = n_total - n_pos

        n_neg = max(n_neg, 1)
        n_pos = max(n_pos, 1)

        w_neg = n_total / (2.0 * n_neg)
        w_pos = n_total / (2.0 * n_pos)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)
