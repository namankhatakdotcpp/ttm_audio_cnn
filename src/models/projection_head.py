"""ProjectionHead: maps AudioCNN embeddings to L2-normalized 256-d vectors.

The head follows the standard two-layer MLP + BN design used in contrastive
representation learning (SimCLR-style), adapted here for supervised fine-tuning.
The L2 normalization at the output ensures that dot-product similarity is
equivalent to cosine similarity — a useful property for the downstream S6/S7
fusion module which aggregates audio and video embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with L2-normalized output.

    Role in pipeline:
        AudioCNN output (B, 512)* → ProjectionHead → (B, 256) L2-normed

    *Note: AudioCNN with layer4 removed outputs 256 channels from layer3.
    This head therefore maps 256 → 256 by default. The input_dim arg allows
    the caller to override if the backbone config changes.

    Architecture:
        Linear(input_dim, hidden_dim) → BN1d → ReLU → Dropout → Linear(hidden_dim, output_dim)
        → L2 normalize

    Args:
        input_dim: Dimensionality of the incoming AudioCNN embedding.
        hidden_dim: Width of the intermediate layer.
        output_dim: Dimensionality of the final L2-normalized embedding.
        dropout: Dropout probability between the two linear layers.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize the input embedding.

        Args:
            x: Float tensor of shape (B, input_dim).

        Returns:
            L2-normalized embedding of shape (B, output_dim).
            Each row has unit L2 norm.
        """
        projected = self.net(x)                                   # (B, output_dim)
        return F.normalize(projected, p=2, dim=1)                 # (B, output_dim)
