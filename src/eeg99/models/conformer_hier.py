"""HierarchicalCausalConformer (original algorithm #2).

Three attention scales (short / medium / long), all strictly causal via a
causal self-attention mask. SubjectAdaptiveFiLM is injected after every
feed-forward block to adapt representations to each subject.

Architecture
------------
Input: (B, C, W) EEG window

PatchEmbed: Conv1d → (B, T', d_model)

Level 0 (short  context, T' frames):   2 causal conformer blocks
Level 1 (medium context, T'//4 frames): 2 causal conformer blocks + FiLM
Level 2 (long   context, T'//16 frames):2 causal conformer blocks + FiLM

Cross-level fusion: concat + linear → (B, d_model)
Head: Linear → n_events logits
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg99.models.adapter import FiLMLayer, SubjectAdaptiveFiLM
from eeg99.models.base import BaseModel, ModelMetadata
from eeg99.utils.constants import N_CHANNELS, N_EVENTS, N_SUBJECTS

__all__ = ["HierarchicalCausalConformer", "ConformerConfig"]


@dataclass(frozen=True)
class ConformerConfig:
    n_channels: int = N_CHANNELS
    n_events: int = N_EVENTS
    n_subjects: int = N_SUBJECTS
    window_samples: int = 500       # default receptive field
    patch_size: int = 8             # conv stride/kernel = 8 samples = 16ms
    d_model: int = 128
    n_heads: int = 4
    ff_dim: int = 256
    n_blocks_per_level: int = 2
    film_embed_dim: int = 16
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Causal multi-head self-attention
# ---------------------------------------------------------------------------

class _CausalMHSA(nn.Module):
    """Multi-head self-attention with strictly causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        T = x.size(1)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )  # True → mask out (future positions)
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Causal Conformer block
# ---------------------------------------------------------------------------

class _CausalConformerBlock(nn.Module):
    """Pre-norm causal conformer: MHSA + causal depthwise conv + FFN."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float, n_subjects: int, film_embed_dim: int,
                 use_film: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CausalMHSA(d_model, n_heads, dropout)

        # Causal depthwise conv (dilation=1, left-pad to preserve causality)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self._conv_pad = 8  # kernel_size - 1

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.use_film = use_film
        if use_film:
            self.film = SubjectAdaptiveFiLM(d_model, n_subjects=n_subjects,
                                            embed_dim=film_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, d_model)"""
        # MHSA sub-layer
        x = x + self.attn(self.norm1(x))

        # Causal depthwise conv sub-layer  (B, T, d_model) → (B, d_model, T)
        h = self.norm2(x).transpose(1, 2)
        h = F.pad(h, (self._conv_pad, 0))
        h = self.conv(h).transpose(1, 2)        # (B, T, d_model)
        x = x + h

        # FFN sub-layer
        x = x + self.ff(self.norm3(x))

        # FiLM modulation (subject-adaptive scaling/shift)
        if self.use_film and subject_id is not None:
            gamma, beta = self.film(subject_id)             # (B, d_model)
            x = gamma.unsqueeze(1) * x + beta.unsqueeze(1) # (B, T, d_model)

        return x


# ---------------------------------------------------------------------------
# Hierarchical model
# ---------------------------------------------------------------------------

class HierarchicalCausalConformer(nn.Module, BaseModel):
    """Three-level hierarchical causal conformer with SubjectAdaptiveFiLM."""

    def __init__(self, cfg: ConformerConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg or ConformerConfig()
        d = self.cfg.d_model

        # Patch embedding: collapse channels via conv
        self.patch_embed = nn.Sequential(
            nn.Conv1d(self.cfg.n_channels, d,
                      kernel_size=self.cfg.patch_size,
                      stride=self.cfg.patch_size, bias=False),
            nn.GELU(),
        )

        nb = self.cfg.n_blocks_per_level
        kw = dict(d_model=d, n_heads=self.cfg.n_heads, ff_dim=self.cfg.ff_dim,
                  dropout=self.cfg.dropout, n_subjects=self.cfg.n_subjects,
                  film_embed_dim=self.cfg.film_embed_dim)

        # Level 0: short context (every frame)
        self.level0 = nn.ModuleList([
            _CausalConformerBlock(**kw, use_film=(i == nb - 1)) for i in range(nb)
        ])

        # Level 1: medium context (stride-4 pooled)
        self.pool01 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.level1 = nn.ModuleList([
            _CausalConformerBlock(**kw, use_film=(i == nb - 1)) for i in range(nb)
        ])

        # Level 2: long context (stride-4 again → 16× from original)
        self.pool12 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.level2 = nn.ModuleList([
            _CausalConformerBlock(**kw, use_film=(i == nb - 1)) for i in range(nb)
        ])

        # Cross-level fusion: concat CLS tokens from each level
        self.fusion = nn.Sequential(
            nn.Linear(d * 3, d),
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
        h = self.patch_embed(x)             # (B, d, T')
        h = h.transpose(1, 2)              # (B, T', d)

        # Level 0
        for blk in self.level0:
            h = blk(h, subject_id)
        cls0 = h[:, -1, :]                 # last (most recent) token

        # Level 1
        h1 = self.pool01(h.transpose(1, 2)).transpose(1, 2)
        for blk in self.level1:
            h1 = blk(h1, subject_id)
        cls1 = h1[:, -1, :]

        # Level 2
        h2 = self.pool12(h1.transpose(1, 2)).transpose(1, 2)
        for blk in self.level2:
            h2 = blk(h2, subject_id)
        cls2 = h2[:, -1, :]

        fused = self.fusion(torch.cat([cls0, cls1, cls2], dim=-1))  # (B, d)
        return self.head(fused)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict_proba(self, eeg: "np.ndarray", subject_id: int) -> "np.ndarray":  # type: ignore[override]
        import numpy as np
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(eeg.T[None]).float().to(device)   # (1, C, T)
        sid = torch.tensor([subject_id], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self(x, sid)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return np.tile(probs, (eeg.shape[0], 1)).astype(np.float32)

    @property
    def metadata(self) -> ModelMetadata:
        total = sum(p.numel() for p in self.parameters())
        return ModelMetadata(
            name="HierarchicalCausalConformer",
            window_samples=self.cfg.window_samples,
            n_channels=self.cfg.n_channels,
            n_events=self.cfg.n_events,
            n_subjects=self.cfg.n_subjects,
            causal=True,
            n_params=total,
            architecture_family="conformer",
        )
