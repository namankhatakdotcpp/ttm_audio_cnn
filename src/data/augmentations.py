"""AudioAugmentPipeline: training-time spectrogram and waveform augmentations.

Augmentations are applied in the order:
  1. Additive noise  (waveform domain — before Mel extraction)
  2. Pitch shift     (waveform domain — before Mel extraction)
  3. SpecAugment     (spectrogram domain — after Mel extraction)

All augmentations are gated by self.training so the same object can be used
in both train and eval DataLoaders without any special handling.

NOTE: Time-stretch is intentionally excluded — it breaks temporal alignment
with the face-track timestamps used by the S6/S7 fusion module.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T


class AudioAugmentPipeline(nn.Module):
    """Augmentation pipeline for TTM audio clips.

    Role in pipeline:
        Waveform → (noise + pitch shift) → MelExtractor → SpecAugment → Tensor

    The pipeline has two entry points:
      - augment_waveform(waveform): call BEFORE MelExtractor
      - augment_spectrogram(mel): call AFTER MelExtractor

    Both methods are no-ops when self.training is False.

    Args:
        sample_rate: Audio sample rate in Hz.
        freq_mask_param: Maximum width of frequency mask (SpecAugment F param).
        time_mask_param: Maximum width of time mask (SpecAugment T param).
        num_masks: Number of frequency AND time masks to apply independently.
        noise_snr_range: (min_snr_db, max_snr_db) for additive noise mixing.
        pitch_shift_semitones: Max pitch shift magnitude in semitones (±).
        noise_clip_dir: Optional directory of .pt waveform files to use as noise
                        sources. If None or empty, Gaussian noise is used instead.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        freq_mask_param: int = 27,
        time_mask_param: int = 40,
        num_masks: int = 2,
        noise_snr_range: tuple[float, float] = (5.0, 20.0),
        pitch_shift_semitones: float = 2.0,
        noise_clip_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.noise_snr_min, self.noise_snr_max = noise_snr_range
        self.pitch_shift_semitones = pitch_shift_semitones
        self.noise_clip_dir = noise_clip_dir

        # Build SpecAugment transforms — applied repeatedly (num_masks times)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_masks = num_masks

        # Cache noise clip paths for fast lookup at runtime
        self._noise_paths: list[Path] = []
        if noise_clip_dir is not None and noise_clip_dir.exists():
            self._noise_paths = list(noise_clip_dir.glob("*.pt"))

    # ------------------------------------------------------------------
    # Waveform-domain augmentations (call BEFORE MelExtractor)
    # ------------------------------------------------------------------

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply waveform-domain augmentations: additive noise + pitch shift.

        Args:
            waveform: Float tensor of shape (1, samples) at self.sample_rate.

        Returns:
            Augmented waveform of the same shape. Identical to input when
            self.training is False.
        """
        if not self.training:
            return waveform

        waveform = self._add_noise(waveform)
        waveform = self._pitch_shift(waveform)
        return waveform

    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mix waveform with a noise source at a random SNR.

        Tries to load a cached Ego4D clip as noise source; falls back to
        Gaussian noise if no clips are available.

        Args:
            waveform: (1, samples) float tensor.

        Returns:
            Noisy waveform of the same shape.
        """
        snr_db = random.uniform(self.noise_snr_min, self.noise_snr_max)
        n_samples = waveform.shape[-1]

        if self._noise_paths:
            noise_path = random.choice(self._noise_paths)
            try:
                noise_clip: torch.Tensor = torch.load(noise_path, weights_only=True)
                # Ensure mono
                if noise_clip.dim() == 2 and noise_clip.shape[0] > 1:
                    noise_clip = noise_clip.mean(dim=0, keepdim=True)
                # Match length by tiling or cropping
                if noise_clip.shape[-1] < n_samples:
                    repeats = (n_samples // noise_clip.shape[-1]) + 1
                    noise_clip = noise_clip.repeat(1, repeats)
                start = random.randint(0, noise_clip.shape[-1] - n_samples)
                noise = noise_clip[..., start : start + n_samples].to(waveform.device)
            except Exception:
                noise = torch.randn_like(waveform)
        else:
            noise = torch.randn_like(waveform)

        # Scale noise to achieve target SNR
        signal_power = waveform.pow(2).mean().clamp_min(1e-9)
        noise_power = noise.pow(2).mean().clamp_min(1e-9)
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = (signal_power / (noise_power * snr_linear)).sqrt()
        return waveform + scale * noise

    def _pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Randomly shift pitch by ±pitch_shift_semitones.

        Args:
            waveform: (1, samples) float tensor.

        Returns:
            Pitch-shifted waveform of the same shape.
        """
        n_steps = random.uniform(-self.pitch_shift_semitones, self.pitch_shift_semitones)
        if abs(n_steps) < 0.1:
            return waveform
        shifted = F.pitch_shift(
            waveform,
            sample_rate=self.sample_rate,
            n_steps=n_steps,
        )
        return shifted

    # ------------------------------------------------------------------
    # Spectrogram-domain augmentations (call AFTER MelExtractor)
    # ------------------------------------------------------------------

    def augment_spectrogram(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment: frequency and time masking.

        Args:
            mel: Float tensor of shape (1, n_mels, T_frames).

        Returns:
            Augmented spectrogram of the same shape. Identical to input when
            self.training is False.
        """
        if not self.training:
            return mel

        for _ in range(self.num_masks):
            mel = self.freq_masking(mel)
            mel = self.time_masking(mel)

        return mel
