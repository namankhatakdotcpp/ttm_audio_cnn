"""integration_test.py: verify S4/S5 core components are compatible with S6/S7.

Tests the three critical files in isolation and end-to-end:
  1. AudioCNN: input (B, 1, n_mels, T) -> output (B, 256)
  2. MelExtractor: raw waveform -> normalized spectrogram (1, 128, T)
  3. AudioCNN + ProjectionHead: (B, 256) L2-normalized embeddings

Run from the project root:
    python integration_test.py

Exits 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from src.data.mel_extractor import MelExtractor
from src.models.audio_cnn import AudioCNN
from src.models.projection_head import ProjectionHead


def test_audio_cnn_shape() -> None:
    """AudioCNN must produce (B, 256) from (B, 1, n_mels, T_frames)."""
    model = AudioCNN(pretrained=False)
    model.eval()

    batch = torch.randn(4, 1, 128, 80)  # 4 clips, 128 mel bins, 80 time frames
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print(f"  [PASS] AudioCNN output shape: {out.shape}")


def test_audio_cnn_layer4_absent() -> None:
    """layer4 must not exist on AudioCNN (removed to reduce params)."""
    model = AudioCNN(pretrained=False)
    assert not hasattr(model, "layer4"), "layer4 should be absent from AudioCNN"
    print("  [PASS] AudioCNN.layer4 is absent")


def test_audio_cnn_first_conv_kernel() -> None:
    """First conv must be (7, 1) kernel — preserves mel-frequency axis."""
    model = AudioCNN(pretrained=False)
    kH, kW = model.conv1.kernel_size
    assert (kH, kW) == (7, 1), f"Expected (7, 1) kernel, got ({kH}, {kW})"
    print(f"  [PASS] AudioCNN conv1 kernel: ({kH}, {kW})")


def test_mel_extractor_shape() -> None:
    """MelExtractor must return (1, 128, T) from a mono waveform."""
    extractor = MelExtractor(sample_rate=16000, n_mels=128)
    # 1 second of 440 Hz sine wave at 16 kHz
    t = torch.linspace(0, 1.0, 16000)
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # (1, 16000)

    mel = extractor(waveform)

    assert mel.shape[0] == 1, f"Expected 1 channel, got {mel.shape[0]}"
    assert mel.shape[1] == 128, f"Expected 128 mel bins, got {mel.shape[1]}"
    assert mel.shape[2] > 0, "Expected non-zero time frames"
    print(f"  [PASS] MelExtractor output shape: {mel.shape}  (1, 128, {mel.shape[2]})")


def test_mel_extractor_normalization() -> None:
    """Per-clip zscore: output should be approximately zero-mean, unit-variance."""
    extractor = MelExtractor(sample_rate=16000, n_mels=128, clip_norm_range=10.0)
    waveform = torch.randn(1, 32000)  # 2 seconds of noise

    mel = extractor(waveform)

    mean = mel.mean().item()
    std = mel.std().item()
    assert abs(mean) < 0.5, f"Mean should be near 0, got {mean:.4f}"
    assert 0.5 < std < 2.0, f"Std should be near 1, got {std:.4f}"
    print(f"  [PASS] MelExtractor normalization: mean={mean:.4f}, std={std:.4f}")


def test_projection_head_l2_norm() -> None:
    """ProjectionHead output must be L2-normalized (unit norm per row)."""
    proj = ProjectionHead(input_dim=256, output_dim=256)
    proj.eval()

    feats = torch.randn(4, 256)
    with torch.no_grad():
        emb = proj(feats)

    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        f"Expected unit norms, got: {norms}"
    print(f"  [PASS] ProjectionHead L2 norms: {norms.tolist()}")


def test_full_pipeline() -> None:
    """AudioCNN -> ProjectionHead -> (4, 256) L2-normalized embeddings."""
    backbone = AudioCNN(pretrained=False)
    proj = ProjectionHead(input_dim=256, output_dim=256)
    backbone.eval()
    proj.eval()

    mels = torch.randn(4, 1, 128, 80)
    with torch.no_grad():
        feats = backbone(mels)    # (4, 256)
        embs = proj(feats)        # (4, 256) — L2 normalized

    assert embs.shape == (4, 256), f"Expected (4, 256), got {embs.shape}"
    norms = embs.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
    print(f"  [PASS] Full pipeline: {mels.shape} -> {embs.shape}, norms ≈ 1.0")


def test_windowed_embedding_contract() -> None:
    """Windowed inference must produce (T_windows, 256) — the S6/S7 contract."""
    from src.evaluation.extract_embeddings import _sliding_windows

    # 5-second clip: 128 mel bins, ~500 frames at 160 hop / 16kHz
    mel = torch.randn(1, 128, 500)
    windows = _sliding_windows(mel, window_size=80, stride=16)

    assert windows.dim() == 4, f"Expected 4D tensor, got {windows.dim()}D"
    assert windows.shape[1] == 1, "Expected 1 channel dim"
    assert windows.shape[2] == 128, "Expected 128 mel bins"
    assert windows.shape[3] == 80, "Expected window_size=80 frames"

    backbone = AudioCNN(pretrained=False)
    proj = ProjectionHead(input_dim=256, output_dim=256)
    backbone.eval()
    proj.eval()

    with torch.no_grad():
        feats = backbone(windows)
        embs = proj(feats)
        embs = F.normalize(embs, dim=-1)

    T = windows.shape[0]
    assert embs.shape == (T, 256), f"Expected ({T}, 256), got {embs.shape}"
    print(f"  [PASS] Windowed embedding contract: mel {mel.shape} -> embeddings {embs.shape}")


if __name__ == "__main__":
    print("\nRunning S4/S5 integration tests...\n")

    tests = [
        test_audio_cnn_shape,
        test_audio_cnn_layer4_absent,
        test_audio_cnn_first_conv_kernel,
        test_mel_extractor_shape,
        test_mel_extractor_normalization,
        test_projection_head_l2_norm,
        test_full_pipeline,
        test_windowed_embedding_contract,
    ]

    failures = []
    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            print(f"  [FAIL] {test_fn.__name__}: {exc}")
            failures.append(test_fn.__name__)

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("ALL INTEGRATION TESTS PASSED")
        sys.exit(0)
