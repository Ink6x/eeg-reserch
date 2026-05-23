"""SubjectAdaptiveFiLM (original algorithm #1).

FiLM-style feature modulation conditioned on a learned subject embedding.
Applied at every normalization layer of every base model to absorb
inter-subject heterogeneity at all depths.

Reference: Perez et al. (2018) "FiLM: Visual Reasoning with a General
Conditioning Layer."
"""
from __future__ import annotations

import torch
import torch.nn as nn

from eeg99.utils.constants import N_SUBJECTS

__all__ = ["SubjectAdaptiveFiLM", "FiLMLayer"]


class SubjectAdaptiveFiLM(nn.Module):
    """Subject embedding → per-channel FiLM scale and shift.

    Parameters
    ----------
    n_subjects : int
        Total number of subjects (embedding table size).
    embed_dim : int
        Dimensionality of the subject embedding vector.
    n_features : int
        Number of feature channels to modulate (γ and β size).
    """

    def __init__(
        self,
        n_features: int,
        n_subjects: int = N_SUBJECTS,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_subjects, embed_dim)
        self.gamma_net = nn.Linear(embed_dim, n_features)
        self.beta_net = nn.Linear(embed_dim, n_features)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        # Small weights so γ≈1, β≈0 at init (stable start) but subjects still differ
        nn.init.normal_(self.gamma_net.weight, std=0.01)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.normal_(self.beta_net.weight, std=0.01)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, subject_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (γ, β) for the given subject batch.

        Parameters
        ----------
        subject_id : torch.Tensor
            Long tensor of shape ``(batch,)``.

        Returns
        -------
        gamma : torch.Tensor  shape ``(batch, n_features)``
        beta  : torch.Tensor  shape ``(batch, n_features)``
        """
        emb = self.embedding(subject_id)   # (B, embed_dim)
        gamma = self.gamma_net(emb)        # (B, n_features)
        beta = self.beta_net(emb)          # (B, n_features)
        return gamma, beta


class FiLMLayer(nn.Module):
    """Drop-in replacement for BatchNorm1d / LayerNorm with FiLM modulation.

    After standard normalisation the output is scaled and shifted by
    subject-specific γ and β produced by :class:`SubjectAdaptiveFiLM`.

    Parameters
    ----------
    n_features : int
        Channel dimension (same as ``SubjectAdaptiveFiLM.n_features``).
    norm_type : str
        ``"batch"`` for BatchNorm1d or ``"layer"`` for LayerNorm.
    """

    def __init__(
        self,
        n_features: int,
        norm_type: str = "batch",
        n_subjects: int = N_SUBJECTS,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        if norm_type == "batch":
            # affine=False: FiLM provides scale/shift
            self.norm: nn.Module = nn.BatchNorm1d(n_features, affine=False)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(n_features, elementwise_affine=False)
        else:
            raise ValueError(f"norm_type must be 'batch' or 'layer', got {norm_type!r}")

        self.film = SubjectAdaptiveFiLM(n_features, n_subjects=n_subjects, embed_dim=embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        """Apply norm then FiLM modulation.

        Parameters
        ----------
        x : torch.Tensor
            For BatchNorm: ``(B, C)`` or ``(B, C, T)``.
            For LayerNorm: ``(B, ..., C)``.
        subject_id : torch.Tensor
            ``(B,)`` long tensor.
        """
        x = self.norm(x)
        gamma, beta = self.film(subject_id)

        # Broadcast over time dimension if present
        if x.dim() == 3:                    # (B, C, T)
            gamma = gamma.unsqueeze(-1)     # (B, C, 1)
            beta = beta.unsqueeze(-1)
        return gamma * x + beta
