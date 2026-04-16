"""GradCAMVisualizer: gradient-weighted class activation maps for the audio CNN.

Hooks onto AudioCNN.layer3 and computes Grad-CAM heatmaps showing which
frequency-time regions the model attends to when predicting TTM=1.

The visualization is side-by-side: original Mel spectrogram (left) and
spectrogram with the CAM overlay (right), saved as a matplotlib figure.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," ICCV 2017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # non-interactive backend — safe for servers

from src.models.audio_cnn import AudioCNN


class GradCAMVisualizer:
    """Compute and visualize Grad-CAM heatmaps for AudioCNN.layer3.

    Role in pipeline:
        Checkpoint → AudioCNN → GradCAMVisualizer.generate(mel) → heatmap figure

    Usage::

        visualizer = GradCAMVisualizer(model)
        heatmap = visualizer.generate(mel_tensor, class_idx=1)
        visualizer.save_figure(Path("outputs/gradcam_sample.png"))

    Args:
        model: Trained AudioCNN instance in eval mode.
    """

    def __init__(self, model: AudioCNN) -> None:
        self.model = model
        self.model.eval()

        self._last_mel: Optional[torch.Tensor] = None
        self._last_heatmap: Optional[np.ndarray] = None

        self.model.register_gradcam_hooks()

    def generate(
        self,
        mel_tensor: torch.Tensor,
        class_idx: int = 1,
        classifier_head: Optional[torch.nn.Module] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for the given Mel spectrogram.

        Args:
            mel_tensor: Float tensor of shape (1, 1, n_mels, T_frames).
                        Must be on the same device as the model.
            class_idx: Class index to explain (1 = TTM, 0 = not TTM).
            classifier_head: Optional linear head to compute the target score.
                             If None, uses the mean activation of layer3 as proxy.

        Returns:
            Heatmap as a float32 numpy array of shape (n_mels, T_frames),
            normalized to [0, 1].
        """
        assert mel_tensor.shape[0] == 1, "Generate expects a single sample (B=1)."
        self._last_mel = mel_tensor.detach()

        mel_tensor = mel_tensor.clone().requires_grad_(True)

        # Forward pass — hooks capture activations into self._activations (detached)
        features = self._full_forward(mel_tensor)  # (1, 256) — NOT detached

        if classifier_head is not None:
            score = classifier_head(features)[0, 0]
        else:
            # Score = sum of pooled features. This is non-detached so backward
            # propagates gradients all the way back through avgpool → layer3,
            # which triggers the backward hook and populates self._gradients.
            # Using self._activations.mean() would fail because that buffer is
            # detached (stored for display only) and backward would silently
            # produce zero gradients.
            score = features.sum()

        # Backward pass — hooks capture gradients into self._gradients (detached)
        self.model.zero_grad()
        score.backward()

        activations, gradients = self.model.get_activation_maps()

        # Grad-CAM: global-average-pool the gradients, weight activation maps
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(cam)

        # Upsample CAM to input spectrogram resolution
        target_h = mel_tensor.shape[2]
        target_w = mel_tensor.shape[3]
        cam_up = F.interpolate(
            cam, size=(target_h, target_w), mode="bilinear", align_corners=False
        )  # (1,1,n_mels,T)

        cam_np = cam_up.squeeze().cpu().numpy().astype(np.float32)

        # Normalize to [0,1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-6:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        self._last_heatmap = cam_np
        return cam_np

    def _full_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the AudioCNN backbone forward pass directly."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.pool(x)
        return x.flatten(1)

    def save_figure(self, path: Path) -> None:
        """Save side-by-side spectrogram + CAM overlay figure to disk.

        Args:
            path: Output file path (.png recommended).

        Raises:
            RuntimeError: If generate() has not been called yet.
        """
        if self._last_heatmap is None or self._last_mel is None:
            raise RuntimeError("Call generate() before save_figure().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mel_np = self._last_mel.squeeze().cpu().numpy()  # (n_mels, T)
        # Normalize mel to [0,1] for display
        mel_min, mel_max = mel_np.min(), mel_np.max()
        mel_disp = (mel_np - mel_min) / (mel_max - mel_min + 1e-6)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].imshow(mel_disp, origin="lower", aspect="auto", cmap="magma")
        axes[0].set_title("Mel Spectrogram")
        axes[0].set_xlabel("Time frames")
        axes[0].set_ylabel("Mel bins")

        # Overlay: spectrogram as base + CAM as colour overlay
        axes[1].imshow(mel_disp, origin="lower", aspect="auto", cmap="gray")
        axes[1].imshow(
            self._last_heatmap,
            origin="lower",
            aspect="auto",
            cmap="jet",
            alpha=0.5,
        )
        axes[1].set_title("Grad-CAM Overlay (TTM=1)")
        axes[1].set_xlabel("Time frames")
        axes[1].set_ylabel("Mel bins")

        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
