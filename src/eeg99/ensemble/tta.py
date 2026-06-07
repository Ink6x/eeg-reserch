"""Test-time augmentation pipeline (5 causal-safe augmentations averaged).

All augmentations preserve causality — no temporal shifts that would
introduce future information.

Augmentation types
------------------
1. identity          — original signal
2. amplitude_scale   — random scale in [0.8, 1.2]
3. additive_noise    — Gaussian noise σ=0.05
4. channel_dropout   — zero out 10% of channels randomly
5. frequency_phase   — random phase shift in frequency domain (causal-safe
                       because we shift ALL frequencies uniformly, preserving
                       the envelope; no lookahead introduced)

Usage
-----
::

    tta = TTAWrapper(model, n_aug=5)
    probs = tta.predict(x, subject_id)   # averaged over augmentations
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["TTAWrapper", "TTAConfig"]


@dataclass(frozen=True)
class TTAConfig:
    n_aug: int = 5
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.05
    channel_dropout_rate: float = 0.10
    seed: int = 0


# ---------------------------------------------------------------------------
# Augmentation functions  (all operate on torch Tensors, shape (B, C, W))
# ---------------------------------------------------------------------------

def _identity(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    return x


def _amplitude_scale(
    x: torch.Tensor,
    rng: torch.Generator,
    lo: float = 0.8,
    hi: float = 1.2,
) -> torch.Tensor:
    scale = lo + (hi - lo) * torch.rand(x.shape[0], 1, 1, generator=rng, device=x.device)
    return x * scale


def _additive_noise(
    x: torch.Tensor,
    rng: torch.Generator,
    std: float = 0.05,
) -> torch.Tensor:
    noise = torch.zeros_like(x)
    noise.normal_(generator=rng)
    return x + std * noise


def _channel_dropout(
    x: torch.Tensor,
    rng: torch.Generator,
    rate: float = 0.10,
) -> torch.Tensor:
    B, C, W = x.shape
    mask = (torch.rand(B, C, 1, generator=rng, device=x.device) > rate).float()
    return x * mask


def _frequency_phase_shift(
    x: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    """Uniform phase shift applied to all frequency bins (causal safe)."""
    phase = 2 * torch.pi * torch.rand(1, generator=rng, device=x.device)
    # FFT along time axis
    X_f = torch.fft.rfft(x, dim=-1)
    X_f = X_f * torch.exp(1j * phase)
    return torch.fft.irfft(X_f, n=x.shape[-1], dim=-1).to(x.dtype)


_AUG_FNS: list[Callable] = [
    _identity,
    _amplitude_scale,
    _additive_noise,
    _channel_dropout,
    _frequency_phase_shift,
]


# ---------------------------------------------------------------------------
# TTA wrapper
# ---------------------------------------------------------------------------

class TTAWrapper:
    """Wrap any model to perform test-time augmentation.

    Parameters
    ----------
    model : nn.Module
        Model with ``forward(x, subject_id) → (B, n_events)`` signature.
    cfg : TTAConfig
    """

    def __init__(self, model: nn.Module, cfg: TTAConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or TTAConfig()
        self._aug_fns = _AUG_FNS[: self.cfg.n_aug]

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """Average sigmoid predictions over augmentations.

        Parameters
        ----------
        x          : (B, C, W)
        subject_id : (B,)

        Returns
        -------
        torch.Tensor  (B, n_events) averaged probabilities [0, 1]
        """
        self.model.eval()
        rng = torch.Generator(device=x.device)
        rng.manual_seed(self.cfg.seed)

        sum_probs = torch.zeros(
            x.shape[0], _infer_n_events(self.model, x, subject_id),
            device=x.device, dtype=torch.float32,
        )

        for aug_fn in self._aug_fns:
            x_aug = aug_fn(x, rng)
            logits = self.model(x_aug, subject_id)
            sum_probs += torch.sigmoid(logits)

        return sum_probs / len(self._aug_fns)


def _infer_n_events(model: nn.Module, x: torch.Tensor, sid: torch.Tensor) -> int:
    with torch.no_grad():
        out = model(x[:1], sid[:1])
    return out.shape[-1]
