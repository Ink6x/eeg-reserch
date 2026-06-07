"""Multi-scale dilated causal TCN for EEG decoding.

Architecture
------------
Input: (B, C, W) EEG window

Stem: depthwise Conv over channels → (B, d_model, W)

3 parallel dilation streams: dilations (1,2,4), (4,8,16), (16,32,64)
  Each: 3× CausalDilatedBlock(d_model) → global avg pool → (B, d_model)

FiLM modulation per stream
Fusion: concat → Linear(3*d_model → d_model) → n_events
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

__all__ = ["MultiScaleTCN", "TCNConfig"]


@dataclass(frozen=True)
class TCNConfig:
    n_channels: int = N_CHANNELS
    n_events: int = N_EVENTS
    n_subjects: int = N_SUBJECTS
    d_model: int = 64
    n_blocks_per_stream: int = 3
    kernel_size: int = 3
    film_embed_dim: int = 16
    dropout: float = 0.2
    dilation_streams: tuple[tuple[int, ...], ...] = (
        (1, 2, 4),
        (4, 8, 16),
        (16, 32, 64),
    )


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _CausalResBlock(nn.Module):
    """Dilated causal conv with residual connection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self._pad = pad
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, groups=channels, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        h = F.pad(x, (self._pad, 0))
        h = F.elu(self.bn(self.conv2(self.conv1(h))))
        return x + self.drop(h)


# ---------------------------------------------------------------------------
# One TCN stream
# ---------------------------------------------------------------------------

class _TCNStream(nn.Module):

    def __init__(
        self,
        d_model: int,
        dilations: tuple[int, ...],
        kernel_size: int,
        dropout: float,
        n_subjects: int,
        film_embed_dim: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _CausalResBlock(d_model, kernel_size, d, dropout)
            for d in dilations
        ])
        self.film = FiLMLayer(d_model, norm_type="layer",
                              n_subjects=n_subjects, embed_dim=film_embed_dim)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T) → (B, d_model)"""
        for blk in self.blocks:
            x = blk(x)
        pooled = x.mean(dim=-1)                  # (B, d_model)
        return self.film(pooled, subject_id)     # (B, d_model) — LayerNorm on (B, C)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MultiScaleTCN(nn.Module, BaseModel):
    """Multi-scale dilated causal TCN with SubjectAdaptiveFiLM."""

    def __init__(self, cfg: TCNConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg or TCNConfig()
        d = self.cfg.d_model

        # Channel-mixing stem: (B, C, W) → (B, d_model, W)
        self.stem = nn.Sequential(
            nn.Conv1d(self.cfg.n_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ELU(),
        )

        self.streams = nn.ModuleList([
            _TCNStream(
                d_model=d,
                dilations=stream_dilations,
                kernel_size=self.cfg.kernel_size,
                dropout=self.cfg.dropout,
                n_subjects=self.cfg.n_subjects,
                film_embed_dim=self.cfg.film_embed_dim,
            )
            for stream_dilations in self.cfg.dilation_streams
        ])

        n_streams = len(self.cfg.dilation_streams)
        self.fusion = nn.Sequential(
            nn.Linear(d * n_streams, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.head = nn.Linear(d, self.cfg.n_events)

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
        h = self.stem(x)                            # (B, d, W)
        stream_embeds = [s(h, subject_id) for s in self.streams]
        fused = self.fusion(torch.cat(stream_embeds, dim=-1))
        return self.head(fused)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict_proba(self, eeg: np.ndarray, subject_id: int) -> np.ndarray:  # type: ignore[override]
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(eeg.T[None]).float().to(device)
        sid = torch.tensor([subject_id], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self(x, sid)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return np.tile(probs, (eeg.shape[0], 1)).astype(np.float32)

    @property
    def metadata(self) -> ModelMetadata:
        total = sum(p.numel() for p in self.parameters())
        return ModelMetadata(
            name="MultiScaleTCN",
            window_samples=0,
            n_channels=self.cfg.n_channels,
            n_events=self.cfg.n_events,
            n_subjects=self.cfg.n_subjects,
            causal=True,
            n_params=total,
            architecture_family="tcn",
        )
