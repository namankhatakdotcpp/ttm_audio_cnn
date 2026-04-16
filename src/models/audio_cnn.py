"""
AudioCNN: Modified ResNet-18 for Ego4D TTM audio classification.

Key design choices:
- 7x1 first conv kernel (not 7x7): preserves formant structure on freq axis
- Layer4 removed: avoids over-parameterisation on small audio patches
- AdaptiveAvgPool2d output: handles variable-length spectrograms
- get_activation_maps(): exposes layer3 feature maps for Grad-CAM

Output: (B, 256) feature vector, passed to ProjectionHead -> 256-d embedding.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AudioCNN(nn.Module):

    def __init__(
        self,
        pretrained: bool = True,
        first_conv_kernel: tuple[int, int] = (7, 1),
        first_conv_stride: tuple[int, int] = (2, 1),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Replace conv1: 3-channel RGB -> 1-channel Mel, narrow freq kernel
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=first_conv_kernel,
            stride=first_conv_stride,
            padding=(first_conv_kernel[0] // 2, 0),
            bias=False,
        )

        # If pretrained: average the 3 input channels into 1 to preserve
        # pretrained weights rather than reinitializing from scratch
        if pretrained:
            with torch.no_grad():
                original_weight = resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1
                ).conv1.weight  # (64, 3, 7, 7)
                # Mean across channel dim, then take center column of freq axis
                new_weight = original_weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                # Slice to (64, 1, 7, 1) — take center freq column
                center = new_weight.shape[3] // 2
                new_weight = new_weight[:, :, :, center : center + 1]
                backbone.conv1.weight.copy_(new_weight)

        self.conv1    = backbone.conv1
        self.bn1      = backbone.bn1
        self.relu     = backbone.relu
        self.maxpool  = backbone.maxpool
        self.layer1   = backbone.layer1
        self.layer2   = backbone.layer2
        self.layer3   = backbone.layer3
        # layer4 intentionally omitted

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        # layer3 output channels = 256 for ResNet-18
        self._feature_dim = 256

        # Storage for Grad-CAM hooks
        self._activation_maps: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._activation_maps = output

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def register_gradcam_hooks(self) -> None:
        """Call once before running Grad-CAM inference."""
        self.layer3.register_forward_hook(self._save_activation)
        self.layer3.register_full_backward_hook(
            lambda m, gi, go: self._save_gradient(go[0])
        )

    def get_activation_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (activations, gradients) for Grad-CAM computation."""
        if self._activation_maps is None or self._gradients is None:
            raise RuntimeError(
                "No activation maps found. Call register_gradcam_hooks() "
                "before forward pass and run backward() first."
            )
        return self._activation_maps, self._gradients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram tensor of shape (B, 1, n_mels, T_frames)
        Returns:
            Feature vector of shape (B, 256)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)   # (B, 256, H', W')

        x = self.pool(x)     # (B, 256, 1, 1)
        x = x.flatten(1)     # (B, 256)
        x = self.dropout(x)

        return x
