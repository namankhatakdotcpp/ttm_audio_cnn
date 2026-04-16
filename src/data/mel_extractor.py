"""
MelExtractor: converts raw waveform to normalized Mel spectrogram.

Critical design choices:
- Per-clip zscore normalization (NOT global dataset stats)
- fmax=8000 Hz: ego-mic speech energy above 8kHz is noise, not signal
- Clamp to [-3, 3] after normalization: prevents outlier frames dominating
- Returns (1, n_mels, T_frames) — channel-first, ready for CNN

This file is the single source of truth for all Mel computation.
preprocess_audio.py and TTMAudioDataset both call this class.
Never compute Mel anywhere else.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelExtractor(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        clip_norm_range: float = 3.0,
    ) -> None:
        super().__init__()

        self.sample_rate     = sample_rate
        self.hop_length      = hop_length      # used by TTMAudioDataset for fallback shapes
        self.n_mels          = n_mels          # used by TTMAudioDataset for fallback shapes
        self.clip_norm_range = clip_norm_range

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,           # power spectrogram before log
            norm="slaney",
            mel_scale="slaney",
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Raw audio tensor of shape (1, N_samples) or (N_samples,)
                      Must already be at self.sample_rate.
        Returns:
            Normalized Mel spectrogram of shape (1, n_mels, T_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, N)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform)     # (1, n_mels, T_frames)
        mel = self.amplitude_to_db(mel)        # log-scale, range ~[-80, 0]

        # Per-clip zscore — critical for Ego4D's diverse recording environments
        mean = mel.mean()
        std  = mel.std().clamp(min=1e-6)       # avoid div-by-zero on silence
        mel  = (mel - mean) / std

        # Clamp outliers — microphone pops and silence artifacts
        mel = mel.clamp(-self.clip_norm_range, self.clip_norm_range)

        return mel  # (1, n_mels, T_frames)

    def resample_if_needed(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Convenience: resample to target sample_rate if needed."""
        if orig_sr != self.sample_rate:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
            waveform  = resampler(waveform)
        return waveform
