"""Hybrid model: classical handcrafted features + causal DL encoder fused via MLP.

Architecture
------------
Classical stream (numpy, no autograd):
  BandPower (C×5=160) + TimeDomain (C×5=160) + Riemannian (C×(C+1)//2=528)
  Optionally + FBCSP (n_events×n_bands×n_comp=120) if fitted with labels

DL stream (PyTorch, gradients flow):
  3-layer dilated causal TCN → global avg pool → Linear → 128-dim

Fusion:
  Linear(classical_dim → 128) + FiLMLayer
  concat(classical_proj, dl_embed) → MLP(256 → 128) + FiLMLayer → Linear → n_events

The classical features are frozen feature functions; gradients only flow through the
DL encoder and fusion MLP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg99.features.band_power import BandPowerExtractor
from eeg99.features.fbcsp import FBCSPExtractor
from eeg99.features.riemannian import RiemannianExtractor
from eeg99.features.time_domain import TimeDomainExtractor
from eeg99.models.adapter import FiLMLayer
from eeg99.models.base import BaseModel, ModelMetadata
from eeg99.utils.constants import N_CHANNELS, N_EVENTS, N_SUBJECTS

__all__ = ["HybridModel", "HybridConfig"]

_CLASSICAL_BASE_DIM = 160 + 160 + 528    # band_power + time_domain + riemannian
_FBCSP_DIM = 120                          # 6 events × 5 bands × 4 components


@dataclass(frozen=True)
class HybridConfig:
    n_channels: int = N_CHANNELS
    n_events: int = N_EVENTS
    n_subjects: int = N_SUBJECTS
    use_fbcsp: bool = True        # include FBCSP (requires labelled fit)
    dl_channels: int = 32         # TCN hidden channels
    dl_embed_dim: int = 128
    classical_proj_dim: int = 128
    film_embed_dim: int = 16
    dropout: float = 0.25


# ---------------------------------------------------------------------------
# Small causal dilated TCN encoder
# ---------------------------------------------------------------------------

class _CausalDilatedLayer(nn.Module):
    """Single dilated causal conv → BatchNorm → ELU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__()
        kernel = 3
        self._pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        x = F.pad(x, (self._pad, 0))
        return F.elu(self.bn(self.conv(x)))


class _CausalTCNEncoder(nn.Module):
    """Three dilated causal conv layers → global avg pool → linear."""

    def __init__(self, n_channels: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _CausalDilatedLayer(n_channels, hidden, dilation=1),
            _CausalDilatedLayer(hidden, hidden, dilation=2),
            _CausalDilatedLayer(hidden, hidden * 2, dilation=4),
        )
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, out_dim)"""
        h = self.layers(x)           # (B, hidden*2, T)
        h = h.mean(dim=-1)           # global avg pool → (B, hidden*2)
        return self.proj(h)


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------

class HybridModel(nn.Module, BaseModel):
    """Classical feature stream + causal DL encoder, fused via MLP + FiLM."""

    def __init__(self, cfg: HybridConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg or HybridConfig()
        cfg = self.cfg

        # ---- Classical extractors (not nn.Module — no autograd) ----
        self._bp = BandPowerExtractor()
        self._td = TimeDomainExtractor()
        self._riem = RiemannianExtractor()
        self._fbcsp = FBCSPExtractor() if cfg.use_fbcsp else None
        self._riem_fitted = False
        self._fbcsp_fitted = False

        # ---- DL encoder ----
        self.encoder = _CausalTCNEncoder(
            n_channels=cfg.n_channels,
            hidden=cfg.dl_channels,
            out_dim=cfg.dl_embed_dim,
        )

        # ---- Classical projection ----
        classical_dim = _CLASSICAL_BASE_DIM
        if cfg.use_fbcsp:
            classical_dim += _FBCSP_DIM
        self._classical_dim = classical_dim

        self.classical_proj = nn.Linear(classical_dim, cfg.classical_proj_dim)
        self.film_classical = FiLMLayer(
            cfg.classical_proj_dim, norm_type="layer",
            n_subjects=cfg.n_subjects, embed_dim=cfg.film_embed_dim,
        )

        # ---- Fusion MLP ----
        fused_dim = cfg.classical_proj_dim + cfg.dl_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.film_fusion = FiLMLayer(
            128, norm_type="layer",
            n_subjects=cfg.n_subjects, embed_dim=cfg.film_embed_dim,
        )
        self.head = nn.Linear(128, cfg.n_events)

    # ------------------------------------------------------------------
    # Classical feature fitting (call before first forward)
    # ------------------------------------------------------------------

    def fit_classical(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> HybridModel:
        """Fit stateful classical extractors.

        Parameters
        ----------
        X : np.ndarray  (n_windows, n_channels, window_len)
        y : np.ndarray | None  (n_windows, n_events) — required when use_fbcsp=True
        """
        self._riem.fit(X)
        self._riem_fitted = True

        if self.cfg.use_fbcsp:
            if y is None:
                raise ValueError("y required to fit FBCSP")
            self._fbcsp.fit(X, y)
            self._fbcsp_fitted = True

        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        # --- Classical stream (numpy, detached) ---
        x_np = x.detach().cpu().numpy()           # (B, C, W)
        classical = self._extract_classical(x_np) # (B, classical_dim) float32
        classical_t = torch.from_numpy(classical).to(x.device)

        # --- DL stream ---
        dl_embed = self.encoder(x)                # (B, dl_embed_dim)

        # --- Fusion ---
        c_proj = self.classical_proj(classical_t) # (B, classical_proj_dim)
        c_proj = self.film_classical(c_proj, subject_id)

        fused = torch.cat([c_proj, dl_embed], dim=-1)
        fused = self.fusion(fused)                # (B, 128)
        fused = self.film_fusion(fused, subject_id)

        return self.head(fused)

    def _extract_classical(self, X: np.ndarray) -> np.ndarray:
        """Extract concatenated classical features. No autograd."""
        parts: list[np.ndarray] = [
            self._bp.transform(X),    # (B, 160)
            self._td.transform(X),    # (B, 160)
        ]

        if self._riem_fitted:
            parts.append(self._riem.transform(X))   # (B, 528)
        else:
            # Zero-fill until fit() is called (allows untrained forward pass)
            parts.append(np.zeros((X.shape[0], 528), dtype=np.float32))

        if self.cfg.use_fbcsp:
            if self._fbcsp_fitted:
                parts.append(self._fbcsp.transform(X))  # (B, 120)
            else:
                parts.append(np.zeros((X.shape[0], _FBCSP_DIM), dtype=np.float32))

        return np.concatenate(parts, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict_proba(self, eeg: np.ndarray, subject_id: int) -> np.ndarray:
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
            name="HybridModel",
            window_samples=0,    # flexible — depends on input window
            n_channels=self.cfg.n_channels,
            n_events=self.cfg.n_events,
            n_subjects=self.cfg.n_subjects,
            causal=True,
            n_params=total,
            architecture_family="hybrid",
        )
