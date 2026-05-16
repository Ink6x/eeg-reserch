"""Multi-band log power features from filterbank output.

For each window, computes the log mean power per (channel × band).
Output shape: ``(n_windows, n_channels * n_bands)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg99.data.filterbank import CausalFilterBank, FilterBankConfig
from eeg99.utils.constants import N_CHANNELS


@dataclass(frozen=True)
class BandPowerConfig:
    """Configuration for band power extraction."""

    filterbank: FilterBankConfig = FilterBankConfig()
    log: bool = True
    eps: float = 1e-8


class BandPowerExtractor:
    """Compute log mean power per (channel, band) for each window.

    This is a stateless extractor (no fitting required).

    Parameters
    ----------
    config : BandPowerConfig
    """

    def __init__(self, config: BandPowerConfig | None = None) -> None:
        self.config = config or BandPowerConfig()
        self._fb = CausalFilterBank(self.config.filterbank)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract band power features.

        Parameters
        ----------
        X : np.ndarray
            ``(n_windows, n_channels, window_len)``, float32.

        Returns
        -------
        np.ndarray
            ``(n_windows, n_channels * n_bands)``, float32.
        """
        n_win, n_ch, T = X.shape
        n_bands = self._fb.n_bands
        features = np.empty((n_win, n_ch * n_bands), dtype=np.float32)

        for i in range(n_win):
            # (T, C, B) from filterbank
            decomposed = self._fb.apply(X[i].T)  # input: (T, C)
            # power: mean of squared signal per (channel, band)
            power = (decomposed ** 2).mean(axis=0)  # (C, B)
            if self.config.log:
                power = np.log(power + self.config.eps)
            features[i] = power.flatten(order="C").astype(np.float32)

        return features

    @property
    def feature_dim(self) -> int:
        return N_CHANNELS * self._fb.n_bands
