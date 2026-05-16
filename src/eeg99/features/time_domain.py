"""Time-domain statistical features: RMS, line-length, Hjorth parameters.

All features are computed within the causal window (no lookahead).
Output shape: ``(n_windows, n_channels * n_feature_types)``.

Feature types
-------------
* RMS          — root mean square amplitude
* Line length  — sum of |diff| (signal complexity proxy)
* Hjorth activity   — variance of signal
* Hjorth mobility   — sqrt(var(diff) / var(signal))
* Hjorth complexity — mobility(diff(signal)) / mobility(signal)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg99.utils.constants import N_CHANNELS

N_FEATURES_PER_CHANNEL = 5  # rms, line_len, activity, mobility, complexity


@dataclass(frozen=True)
class TimeDomainConfig:
    """Configuration for time-domain feature extraction."""

    eps: float = 1e-8


class TimeDomainExtractor:
    """Compute time-domain features for each window.

    Stateless — no fitting required.

    Parameters
    ----------
    config : TimeDomainConfig
    """

    def __init__(self, config: TimeDomainConfig | None = None) -> None:
        self.config = config or TimeDomainConfig()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract time-domain features.

        Parameters
        ----------
        X : np.ndarray
            ``(n_windows, n_channels, window_len)``, float32.

        Returns
        -------
        np.ndarray
            ``(n_windows, n_channels * 5)``, float32.
        """
        n_win, n_ch, T = X.shape
        features = np.empty((n_win, n_ch * N_FEATURES_PER_CHANNEL), dtype=np.float32)

        for i in range(n_win):
            feats = _extract_one(X[i], self.config.eps)  # (C, 5)
            features[i] = feats.flatten(order="C").astype(np.float32)

        return features

    @property
    def feature_dim(self) -> int:
        return N_CHANNELS * N_FEATURES_PER_CHANNEL


def _extract_one(x: np.ndarray, eps: float) -> np.ndarray:
    """Compute time-domain features for one window.

    Parameters
    ----------
    x : np.ndarray
        ``(n_channels, T)``

    Returns
    -------
    np.ndarray
        ``(n_channels, 5)`` — [rms, line_len, activity, mobility, complexity]
    """
    n_ch = x.shape[0]
    out = np.empty((n_ch, N_FEATURES_PER_CHANNEL), dtype=np.float64)

    dx = np.diff(x, axis=1)            # (C, T-1) first difference
    ddx = np.diff(dx, axis=1)          # (C, T-2) second difference

    # RMS
    out[:, 0] = np.sqrt(np.mean(x ** 2, axis=1))

    # Line length
    out[:, 1] = np.sum(np.abs(dx), axis=1)

    # Hjorth activity = variance
    activity = np.var(x, axis=1)
    out[:, 2] = activity

    # Hjorth mobility = sqrt(var(dx) / var(x))
    mob_sq = np.var(dx, axis=1) / (activity + eps)
    mobility = np.sqrt(np.maximum(mob_sq, 0.0))
    out[:, 3] = mobility

    # Hjorth complexity = mobility(ddx) / mobility(dx)
    mob_dx = np.sqrt(np.maximum(np.var(ddx, axis=1) / (np.var(dx, axis=1) + eps), 0.0))
    out[:, 4] = mob_dx / (mobility + eps)

    return out
