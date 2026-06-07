"""Multi-scale causal EEGNet+ (Component 1).

Five parallel EEGNet branches at window sizes 125/250/500/1000/2000 samples
fused via CrossScaleAttention, with SubjectAdaptiveFiLM at every norm layer.

Architecture
------------
Input: (B, C, W) raw EEG window (causal, zero-padded at the left)

Branch i (window_len=W_i):
  DepthwiseConv(C→C) → FiLMLayer(batch) → ELU
  SeparableConv → FiLMLayer(batch) → ELU → AvgPool → Dropout
  → branch_embed (B, embed_dim)

CrossScaleAttention: 5 branches × embed_dim → fused_embed
Head: Linear → Sigmoid (per event)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg99.models.adapter import FiLMLayer
from eeg99.models.base import BaseModel, ModelMetadata
from eeg99.utils.constants import N_CHANNELS, N_EVENTS, N_SUBJECTS

__all__ = ["MultiScaleEEGNetPlus", "MultiScaleEEGNetConfig"]

_WINDOW_SIZES = (125, 250, 500, 1000, 2000)


@dataclass(frozen=True)
class MultiScaleEEGNetConfig:
    n_channels: int = N_CHANNELS
    n_events: int = N_EVENTS
    n_subjects: int = N_SUBJECTS
    window_sizes: tuple[int, ...] = _WINDOW_SIZES
    branch_embed_dim: int = 64
    film_embed_dim: int = 16
    n_temporal_filters: int = 16   # F1 in EEGNet paper
    d_model: int = 128             # attention model dim
    n_heads: int = 4
    dropout: float = 0.25


# ---------------------------------------------------------------------------
# Single EEGNet branch
# ---------------------------------------------------------------------------

class _CausalEEGNetBranch(nn.Module):
    """One causal EEGNet branch operating on a fixed receptive window."""

    def __init__(self, cfg: MultiScaleEEGNetConfig, window_len: int) -> None:
        super().__init__()
        C = cfg.n_channels
        F1 = cfg.n_temporal_filters
        F2 = F1 * 2

        # Temporal conv: causal via left-only padding
        self.temporal_conv = nn.Conv2d(
            1, F1, kernel_size=(1, min(window_len // 2 + 1, 64)),
            padding=0, bias=False,
        )
        self._temporal_pad = min(window_len // 2, 63)  # left pad width

        self.film1 = FiLMLayer(F1, norm_type="batch",
                               n_subjects=cfg.n_subjects, embed_dim=cfg.film_embed_dim)

        # Depthwise: mix channels, stride=1
        self.depthwise = nn.Conv2d(F1, F2, kernel_size=(C, 1),
                                   groups=F1, bias=False)
        self.film2 = FiLMLayer(F2, norm_type="batch",
                               n_subjects=cfg.n_subjects, embed_dim=cfg.film_embed_dim)

        # Separable conv
        k_sep = min(window_len // 4 + 1, 17)
        self.sep_dw = nn.Conv2d(F2, F2, kernel_size=(1, k_sep),
                                groups=F2, padding=0, bias=False)
        self._sep_pad = k_sep - 1
        self.sep_pw = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.film3 = FiLMLayer(F2, norm_type="batch",
                               n_subjects=cfg.n_subjects, embed_dim=cfg.film_embed_dim)

        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        self.dropout = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(F2 * 4, cfg.branch_embed_dim)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, W)  EEG window
        Returns: (B, branch_embed_dim)
        """
        B = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, C, W)

        # Causal temporal conv: pad left only
        x = F.pad(x, (self._temporal_pad, 0))
        x = self.temporal_conv(x)               # (B, F1, C, T')
        # FiLM on (B, F1) after global avg over space+time
        x = self._film_3d(self.film1, x, subject_id)
        x = F.elu(x)

        # Depthwise (channel mixing)
        x = self.depthwise(x)                   # (B, F2, 1, T')
        x = self._film_3d(self.film2, x, subject_id)
        x = F.elu(x)

        # Separable conv: causal pad
        x = F.pad(x, (self._sep_pad, 0))
        x = self.sep_dw(x)
        x = self.sep_pw(x)
        x = self._film_3d(self.film3, x, subject_id)
        x = F.elu(x)

        x = self.pool(x)                        # (B, F2, 1, 4)
        x = self.dropout(x)
        x = x.view(B, -1)                       # (B, F2*4)
        return self.proj(x)                     # (B, branch_embed_dim)

    @staticmethod
    def _film_3d(
        film: FiLMLayer,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLMLayer to a 4-D tensor (B, C, H, W) by collapsing H×W."""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)           # (B, C, H*W) treated as (B, C, T)
        x_flat = film(x_flat, subject_id)
        return x_flat.view(B, C, H, W)


# ---------------------------------------------------------------------------
# Cross-scale attention fusion
# ---------------------------------------------------------------------------

class _CrossScaleAttention(nn.Module):
    """Attend across branch embeddings to produce fused representation."""

    def __init__(self, n_branches: int, embed_dim: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, branch_embeds: list[torch.Tensor]) -> torch.Tensor:
        """
        branch_embeds: list of (B, embed_dim) tensors, one per branch
        Returns: (B, d_model)
        """
        x = torch.stack(branch_embeds, dim=1)  # (B, n_branches, embed_dim)
        x = self.proj_in(x)                    # (B, n_branches, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.proj_out(x.mean(dim=1))    # (B, d_model)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MultiScaleEEGNetPlus(nn.Module, BaseModel):
    """Multi-scale causal EEGNet+ with SubjectAdaptiveFiLM."""

    def __init__(self, cfg: MultiScaleEEGNetConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg or MultiScaleEEGNetConfig()

        self.branches = nn.ModuleList([
            _CausalEEGNetBranch(self.cfg, w) for w in self.cfg.window_sizes
        ])
        self.fusion = _CrossScaleAttention(
            n_branches=len(self.cfg.window_sizes),
            embed_dim=self.cfg.branch_embed_dim,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
        )
        self.head = nn.Linear(self.cfg.d_model, self.cfg.n_events)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : (B, C, W) float32 EEG window
        subject_id : (B,) long
        Returns    : (B, n_events) logits
        """
        branch_embeds = [branch(x, subject_id) for branch in self.branches]
        fused = self.fusion(branch_embeds)
        return self.head(fused)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict_proba(self, eeg: np.ndarray, subject_id: int) -> np.ndarray:  # type: ignore[override]
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(eeg.T[None]).float().to(device)   # (1, C, T)
        sid = torch.tensor([subject_id], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self(x, sid)                               # (1, E)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        # Expand to (n_samples, n_events) — single window
        return np.tile(probs, (eeg.shape[0], 1)).astype(np.float32)

    @property
    def metadata(self) -> ModelMetadata:
        total = sum(p.numel() for p in self.parameters())
        return ModelMetadata(
            name="MultiScaleEEGNetPlus",
            window_samples=max(self.cfg.window_sizes),
            n_channels=self.cfg.n_channels,
            n_events=self.cfg.n_events,
            n_subjects=self.cfg.n_subjects,
            causal=True,
            n_params=total,
            architecture_family="eegnet",
        )
