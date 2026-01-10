"""
EEGNet — 標準ベースライン (Lawhern et al., 2018)
因果版: DepthwiseConv1d を causal padding で実装
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConv1d(nn.Module):
    """因果的深さ方向 1D 畳み込み (左パディングのみ)"""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # 左パディングのみ
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))  # 左側のみパディング
        return self.conv(x)


class EEGNet(nn.Module):
    """
    EEGNet — 因果的実装。
    入力: (batch, n_channels, n_times)
    出力: (batch, n_classes) ロジット (multi-label)
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = 250,
        n_classes: int = 6,
        f1: int = 8,           # temporal filter 数
        d: int = 2,            # depthwise multiplier
        f2: int = 16,          # separable filter 数
        kernel_len: int = 64,  # temporal kernel (128ms @ 500Hz)
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        # Block 1: Temporal + DepthwiseConv (spatial)
        # 因果的: temporal conv を左パディングで実装
        self.temporal_conv = nn.Sequential(
            nn.ZeroPad2d((kernel_len // 2 - 1, kernel_len // 2, 0, 0)),  # causal pad
            nn.Conv2d(1, f1, (1, kernel_len), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.GELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable Conv
        sep_kernel = 16
        self.separable_conv = nn.Sequential(
            nn.ZeroPad2d((sep_kernel // 2 - 1, sep_kernel // 2, 0, 0)),  # causal pad
            nn.Conv2d(f1 * d, f1 * d, (1, sep_kernel), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # 分類ヘッド
        n_flat = f2 * (n_times // 32)
        self.classifier = nn.Linear(n_flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_times)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)
        return self.classifier(x)
