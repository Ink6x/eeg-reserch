"""Common interface for every base model in the ensemble pool.

Each architecture (EEGNet+, HierarchicalCausalConformer, multi-scale TCN,
Hybrid, SSL-FT) inherits from :class:`BaseModel` so the ensemble layer can
treat them uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata stored alongside every trained model in the registry."""

    name: str
    window_samples: int
    n_channels: int
    n_events: int
    n_subjects: int
    causal: bool
    n_params: int
    architecture_family: str  # "eegnet" / "conformer" / "tcn" / "hybrid" / "ssl"


class BaseModel(ABC):
    """Abstract base for every eeg99 model.

    Notes
    -----
    The forward signature is intentionally numpy-only at the boundary. Inside
    the implementation each subclass may use torch / sklearn / numpy freely;
    the ensemble layer only needs ``predict_proba``.
    """

    @abstractmethod
    def predict_proba(
        self,
        eeg: np.ndarray,
        subject_id: int,
    ) -> np.ndarray:
        """Return causal per-event probabilities.

        Parameters
        ----------
        eeg : np.ndarray
            Shape ``(n_samples, n_channels)``, dtype ``float32``.
        subject_id : int
            Used by subject-conditional modules (FiLM, embeddings).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_events)``, dtype ``float32``, range ``[0, 1]``.
        """

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata: ...
