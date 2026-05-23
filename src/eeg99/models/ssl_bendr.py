"""CausalBENDR — original algorithm #3.

Causal masked EEG modeling with contrastive auxiliary loss.
Pre-trained on all 12 subjects × 10 series (no labels required).

Architecture
------------
CausalBENDREncoder:
  PatchEmbed: Conv1d(C→d_model, stride=patch_stride) → (B, d_model, T')
  6× CausalResBlock (dilated, doubling dilation 1→32)
  Output: (B, T', d_model) latent sequence

SSL Pre-training (MaskedEEGPretrainer):
  1. Mask 50% of latent patches → replace with learnable mask token
  2. Reconstruction head: 2-layer MLP → predict original latent  (cosine loss)
  3. Contrastive auxiliary (BYOL-style):
     - Augment same window twice (amplitude scale + noise)
     - online_proj(online_enc(aug1)) ← predict → stop_grad(target_enc(aug2))
     - EMA update of target encoder each step

Fine-tuning (CausalBENDRFinetuner):
  Frozen/thawed CausalBENDREncoder + SubjectAdaptiveFiLM + Linear head

References
----------
Kostas et al. (2021) "BENDR: Using Transformers and a Contrastive Self-Supervised
  Paradigm for EEG"; Yang et al. (2023) "BIOT".
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg99.models.adapter import FiLMLayer, SubjectAdaptiveFiLM
from eeg99.models.base import BaseModel, ModelMetadata
from eeg99.utils.constants import N_CHANNELS, N_EVENTS, N_SUBJECTS

__all__ = [
    "CausalBENDREncoder",
    "BENDRConfig",
    "MaskedEEGPretrainer",
    "CausalBENDRFinetuner",
]


@dataclass(frozen=True)
class BENDRConfig:
    n_channels: int = N_CHANNELS
    n_events: int = N_EVENTS
    n_subjects: int = N_SUBJECTS
    patch_size: int = 10            # samples per patch (20ms @ 500Hz)
    patch_stride: int = 5           # 50% overlap
    d_model: int = 256
    n_encoder_layers: int = 6       # dilated conv blocks
    kernel_size: int = 3
    dropout: float = 0.1
    mask_ratio: float = 0.50        # fraction of patches to mask
    film_embed_dim: int = 16
    ema_decay: float = 0.999        # for target encoder in contrastive branch


# ---------------------------------------------------------------------------
# Causal residual conv block
# ---------------------------------------------------------------------------

class _CausalConvBlock(nn.Module):
    """Dilated causal conv + LayerNorm + skip."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self._pad = pad
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              dilation=dilation, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model, T)"""
        h = F.pad(x, (self._pad, 0))
        h = F.gelu(self.conv(h))            # (B, d_model, T)
        # LayerNorm over channel dim
        h = self.norm((x + h).transpose(1, 2)).transpose(1, 2)
        # FFN
        out = h.transpose(1, 2)             # (B, T, d_model)
        out = self.norm2(out + self.ff(out))
        return out.transpose(1, 2)          # (B, d_model, T)


# ---------------------------------------------------------------------------
# CausalBENDREncoder
# ---------------------------------------------------------------------------

class CausalBENDREncoder(nn.Module):
    """Causal dilated conv encoder.  Input (B,C,W) → latent (B,T',d_model)."""

    def __init__(self, cfg: BENDRConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Channel mixing patch embedding (causal strided conv)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(cfg.n_channels, cfg.d_model,
                      kernel_size=cfg.patch_size,
                      stride=cfg.patch_stride, bias=False),
            nn.GELU(),
        )

        dilations = [2 ** i for i in range(cfg.n_encoder_layers)]   # 1,2,4,8,16,32
        self.blocks = nn.ModuleList([
            _CausalConvBlock(cfg.d_model, cfg.kernel_size, d, cfg.dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, W)  raw EEG window
        Returns: (B, T', d_model)
        """
        h = self.patch_embed(x)         # (B, d_model, T')
        for blk in self.blocks:
            h = blk(h)
        return h.transpose(1, 2)        # (B, T', d_model)

    def n_patches(self, window_len: int) -> int:
        """Number of patches for a given input length."""
        return (window_len - self.cfg.patch_size) // self.cfg.patch_stride + 1


# ---------------------------------------------------------------------------
# SSL: MaskedEEGPretrainer
# ---------------------------------------------------------------------------

class MaskedEEGPretrainer(nn.Module):
    """MAE-style masked reconstruction + BYOL-style contrastive loss.

    Training procedure
    ------------------
    1. Encode full window → z  (B, T', d)
    2. Randomly mask 50% of patches → z_masked  (masked positions = mask_token)
    3. Reconstruction head predicts z from z_masked → cosine loss
    4. Contrastive: augment × 2, online_proj(enc(aug1)) predicts
       stop_grad(ema_enc(aug2)); use target EMA encoder

    Parameters
    ----------
    cfg : BENDRConfig
    """

    def __init__(self, cfg: BENDRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.encoder = CausalBENDREncoder(cfg)

        # Target (EMA) encoder — no gradients directly
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Reconstruction head: predict original latent from masked latent
        self.rec_head = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )

        # Projector for contrastive branch
        self.projector = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Predictor (online only, BYOL asymmetry)
        self.predictor = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.LayerNorm(d // 2),
            nn.Linear(d // 2, d),
        )

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_target(self) -> None:
        tau = self.cfg.ema_decay
        for o, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            t.data.mul_(tau).add_(o.data, alpha=1.0 - tau)

    # ------------------------------------------------------------------
    # Masking utility
    # ------------------------------------------------------------------

    def _apply_mask(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z : (B, T', d)
        Returns (z_masked, mask) where mask is (B, T') bool (True = masked).
        """
        B, T, d = z.shape
        n_mask = max(1, int(T * self.cfg.mask_ratio))
        noise = torch.rand(B, T, device=z.device)
        ids_shuffle = noise.argsort(dim=1)
        mask = torch.zeros(B, T, device=z.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)

        mask_tokens = self.mask_token.expand(B, T, d)
        z_masked = torch.where(mask.unsqueeze(-1), mask_tokens, z)
        return z_masked, mask

    # ------------------------------------------------------------------
    # Augmentation (causal-safe: no future info introduced)
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        """Amplitude scale + additive noise (no temporal shift → causal safe)."""
        scale = 0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, device=x.device)
        noise = 0.05 * torch.randn_like(x)
        return x * scale + noise

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        x : (B, C, W)
        Returns dict with keys: loss, rec_loss, contrastive_loss
        """
        # --- Reconstruction branch ---
        z = self.encoder(x)                         # (B, T', d)
        z_masked, mask = self._apply_mask(z.detach().clone())
        z_rec = self.rec_head(z_masked)             # (B, T', d)

        # Cosine reconstruction loss on masked positions only
        rec_loss = _cosine_loss(
            z_rec[mask],
            z.detach()[mask],
        )

        # --- Contrastive branch (BYOL) ---
        aug1 = self._augment(x)
        aug2 = self._augment(x)

        z1 = self.encoder(aug1)                     # (B, T', d)
        online_proj1 = self.projector(z1.mean(1))   # (B, d)
        pred1 = self.predictor(online_proj1)        # (B, d)

        with torch.no_grad():
            z2 = self.target_encoder(aug2)
            target_proj2 = self.projector(z2.mean(1))   # (B, d)

        contrastive_loss = _byol_loss(pred1, target_proj2)

        self._update_target()

        total_loss = rec_loss + 0.5 * contrastive_loss
        return {
            "loss": total_loss,
            "rec_loss": rec_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
        }


# ---------------------------------------------------------------------------
# Fine-tuner (downstream classification)
# ---------------------------------------------------------------------------

class CausalBENDRFinetuner(nn.Module, BaseModel):
    """Pre-trained CausalBENDR encoder + SubjectAdaptiveFiLM + task head.

    Parameters
    ----------
    cfg : BENDRConfig
    freeze_encoder : bool
        If True, encoder weights are frozen during fine-tuning.
    """

    def __init__(
        self,
        cfg: BENDRConfig | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg or BENDRConfig()
        d = self.cfg.d_model

        self.encoder = CausalBENDREncoder(self.cfg)
        self._freeze_encoder = freeze_encoder

        self.film = SubjectAdaptiveFiLM(
            n_features=d,
            n_subjects=self.cfg.n_subjects,
            embed_dim=self.cfg.film_embed_dim,
        )
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.head = nn.Linear(d, self.cfg.n_events)

    def load_pretrained_encoder(self, pretrainer: MaskedEEGPretrainer) -> None:
        """Copy encoder weights from a trained MaskedEEGPretrainer."""
        self.encoder.load_state_dict(pretrainer.encoder.state_dict())
        if self._freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : (B, C, W)
        subject_id : (B,) long
        Returns    : (B, n_events) logits
        """
        z = self.encoder(x)                 # (B, T', d)
        h = z.mean(dim=1)                   # (B, d) global avg pool over time

        gamma, beta = self.film(subject_id)
        h = gamma * self.norm(h) + beta
        h = self.dropout(h)
        return self.head(h)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict_proba(self, eeg: "np.ndarray", subject_id: int) -> "np.ndarray":  # type: ignore[override]
        import numpy as np
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
            name="CausalBENDRFinetuner",
            window_samples=0,
            n_channels=self.cfg.n_channels,
            n_events=self.cfg.n_events,
            n_subjects=self.cfg.n_subjects,
            causal=True,
            n_params=total,
            architecture_family="ssl",
        )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean negative cosine similarity (target detached by caller if needed)."""
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    return 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()


def _byol_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BYOL regression loss (target must already be stop-grad)."""
    return _cosine_loss(pred, target)
