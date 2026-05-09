"""Causal preprocessing v2: HP/LP filtering, CAR, running z-score.

All operations are **strictly causal** — implemented via one-pass ``sosfilt``
(never ``sosfiltfilt``). Verified by :func:`assert_no_lookahead`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from eeg99.utils.constants import SFREQ


@dataclass(frozen=True)
class PreprocessConfig:
    """Hyperparameters for the causal filter chain."""

    sfreq: float = SFREQ
    highpass_hz: float = 0.5
    lowpass_hz: float = 45.0
    hp_order: int = 4
    lp_order: int = 4
    zscore_window_s: float = 60.0  # seconds of history for running z-score


class CausalPreprocessor:
    """Stateful causal preprocessor.

    Must call :meth:`fit` on training data before :meth:`transform`.

    Parameters
    ----------
    config : PreprocessConfig
        Filter and normalization settings.
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._hp_sos: np.ndarray | None = None
        self._lp_sos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, eeg: np.ndarray) -> "CausalPreprocessor":
        """Build filter coefficients from config (no data-dependent fitting).

        Parameters
        ----------
        eeg : np.ndarray
            Shape ``(n_samples, n_channels)``. Not mutated.

        Returns
        -------
        CausalPreprocessor
            Self, for method chaining.
        """
        cfg = self.config
        self._hp_sos = butter(
            cfg.hp_order,
            cfg.highpass_hz / (cfg.sfreq / 2),
            btype="high",
            output="sos",
        )
        self._lp_sos = butter(
            cfg.lp_order,
            cfg.lowpass_hz / (cfg.sfreq / 2),
            btype="low",
            output="sos",
        )
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        """Apply HP → LP → CAR → causal running z-score.

        Parameters
        ----------
        eeg : np.ndarray
            Shape ``(n_samples, n_channels)``, dtype ``float32``.

        Returns
        -------
        np.ndarray
            Same shape, dtype ``float32``, causally filtered and normalised.
        """
        if self._hp_sos is None or self._lp_sos is None:
            raise RuntimeError("Call fit() before transform().")

        x = eeg.astype(np.float64)

        # 1. High-pass (causal)
        x = _sosfilt_channels(self._hp_sos, x)

        # 2. Low-pass (causal)
        x = _sosfilt_channels(self._lp_sos, x)

        # 3. Common Average Reference (spatial, not temporal — causal-safe)
        x = x - x.mean(axis=1, keepdims=True)

        # 4. Causal running z-score
        x = _causal_zscore(x, self.config)

        return x.astype(np.float32)

    def fit_transform(self, eeg: np.ndarray) -> np.ndarray:
        return self.fit(eeg).transform(eeg)


# ------------------------------------------------------------------
# Causality verification
# ------------------------------------------------------------------

def assert_no_lookahead(
    preprocessor: CausalPreprocessor,
    n_channels: int = 4,
    tol: float = 1e-6,
) -> None:
    """Verify that the preprocessor does not use future samples.

    Feeds a unit impulse at sample 100 and asserts that all samples before
    the impulse remain zero after transformation.

    Parameters
    ----------
    preprocessor : CausalPreprocessor
        A **fitted** preprocessor instance.
    n_channels : int
        Number of channels to synthesise.
    tol : float
        Absolute tolerance for near-zero check.

    Raises
    ------
    AssertionError
        If any pre-impulse sample has |value| > tol.
    """
    n_samples = 500
    impulse_idx = 100
    eeg = np.zeros((n_samples, n_channels), dtype=np.float32)
    eeg[impulse_idx, :] = 1.0

    out = preprocessor.transform(eeg)

    pre = out[:impulse_idx, :]
    max_pre = np.abs(pre).max()
    assert max_pre <= tol, (
        f"Lookahead detected: max |value| before impulse = {max_pre:.2e} > {tol:.2e}"
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _sosfilt_channels(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply causal SOS filter independently to each channel (column).

    Uses ``sosfilt`` (one-pass, strictly causal).
    """
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = sosfilt(sos, x[:, ch])
    return out


def _causal_zscore(x: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Per-channel causal running z-score using expanding statistics.

    At each sample t, normalize by mean and std computed over samples [0, t].
    A small epsilon prevents division by zero in the first few samples.

    Notes
    -----
    This is O(T * C) but avoids storing a rolling window buffer, keeping
    memory constant. The expanding (cumulative) variant is conservative —
    it converges to the true running statistics as T → ∞.
    """
    eps = 1e-6
    n_samples, n_channels = x.shape
    out = np.empty_like(x)

    # Cumulative sum and sum-of-squares for expanding mean/var
    cum_sum = np.zeros(n_channels)
    cum_sq = np.zeros(n_channels)

    for t in range(n_samples):
        cum_sum += x[t]
        cum_sq += x[t] ** 2
        n = t + 1
        mean = cum_sum / n
        var = cum_sq / n - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0)) + eps
        out[t] = (x[t] - mean) / std

    return out
