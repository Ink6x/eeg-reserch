"""Data augmentation policies (refactored from eegdemo for eeg99).

Phase 0 stub. Implementation in Phase 1.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Augmenter(Protocol):
    """Apply an in-place stochastic transform to a (window, label) pair."""

    def __call__(
        self,
        window: np.ndarray,
        label: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]: ...
