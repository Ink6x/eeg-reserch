"""Focal loss for class-imbalanced binary event detection.

Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection".

FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

For EEG events the positive class is rare (< 5% of windows), so focal loss
helps the model focus on hard examples rather than flooding gradients with
easy negatives.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLoss", "FocalLossConfig"]


@dataclass(frozen=True)
class FocalLossConfig:
    alpha: float = 0.25   # positive class weight
    gamma: float = 2.0    # focusing parameter


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label event detection.

    Parameters
    ----------
    cfg : FocalLossConfig
    """

    def __init__(self, cfg: FocalLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FocalLossConfig()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, n_events) raw scores
        targets : (B, n_events) binary {0, 1}
        Returns : scalar
        """
        alpha = self.cfg.alpha
        gamma = self.cfg.gamma

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                          # probability of correct class
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_weight = alpha_t * (1 - p_t) ** gamma
        return (focal_weight * bce).mean()
