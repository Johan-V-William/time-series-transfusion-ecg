"""
Binary CNN: 5x Conv1D + ReLU + MaxPool -> Fully-Connected -> Dropout > Softmax
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from src.registry import register_model


class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU + MaxPool (one stage)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BeatCNN(nn.Module):
    """
    Binary classifier: ECG segment → P(VB).

    Input:  x ∈ [B, 1, 512]   (batch, channel, time)
    Output: logits ∈ [B, 2]   (NVB=0, VB=1)

    """

    def __init__(
        self,
        conv_channels: Tuple[int, ...] = (32, 64, 128, 128, 64),
        kernel_sizes:  Tuple[int, ...] = (5,  5,  3,   3,   3),
        pool_size: int = 2,
        fc_hidden: int = 128,
        n_classes: int = 2,
        dropout:   float = 0.5,
        input_length: int = 512,
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == 5

        # Build 5 conv blocks
        channels = [1] + list(conv_channels)
        self.conv_layers = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], kernel_sizes[i], pool_size)
            for i in range(5)
        ])

        # Compute flattened size after 5× MaxPool(2) on input 512
        # 512 → 256 → 128 → 64 → 32 → 16
        conv_out_len = input_length // (pool_size ** 5)
        flat_size = conv_channels[-1] * conv_out_len  # 64 * 16 = 1024

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
            # NOTE: No Softmax here — use CrossEntropyLoss (handles it internally)
            # Use Softmax only at inference: P(VB) = softmax(logits)[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 512] → logits: [B, 2]"""
        for layer in self.conv_layers:
            x = layer(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(VB) per sample. Shape: [B]"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)[:, 1]  # P(class=VB)

def build_cnn(cfg=None) -> BeatCNN:
    if cfg is None:
        return BeatCNN()
    return BeatCNN(
        conv_channels=cfg.conv_channels,
        kernel_sizes=cfg.kernel_sizes,
        pool_size=cfg.pool_size,
        fc_hidden=cfg.fc_hidden,
        n_classes=cfg.n_classes,
        dropout=cfg.dropout,
        input_length=cfg.input_length,
    )
