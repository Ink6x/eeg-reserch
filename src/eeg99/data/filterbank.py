"""Causal filter-bank decomposition into 5 EEG frequency bands.

Each band uses a causal Butterworth bandpass (``sosfilt``, one-pass).
Output shape: ``(n_samples, n_channels, n_bands)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, sosfilt

from eeg99.utils.constants import SFREQ


class Band(NamedTuple):
    """A single EEG frequency band."""

    name: str
    low_hz: float
    high_hz: float


DEFAULT_BANDS: tuple[Band, ...] = (
    Band("delta", 1.0, 4.0),
    Band("theta", 4.0, 8.0),
    Band("alpha", 8.0, 13.0),
    Band("beta", 13.0, 30.0),
    Band("gamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class FilterBankConfig:
    """Configuration for the causal filter bank."""

    bands: tuple[Band, ...] = DEFAULT_BANDS
    order: int = 4
    sfreq: float = SFREQ


class CausalFilterBank:
    """Decompose EEG into multiple frequency bands using causal filters.

    Parameters
    ----------
    config : FilterBankConfig
        Band definitions and filter order.
    """

    def __init__(self, config: FilterBankConfig | None = None) -> None:
        self.config = config or FilterBankConfig()
        self._sos_list: list[np.ndarray] = []
        self._build_filters()

    def _build_filters(self) -> None:
        nyq = self.config.sfreq / 2.0
        self._sos_list = []
        for band in self.config.bands:
            sos = butter(
                self.config.order,
                [band.low_hz / nyq, band.high_hz / nyq],
                btype="bandpass",
                output="sos",
            )
            self._sos_list.append(sos)

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Decompose EEG into frequency bands.

        Parameters
        ----------
        eeg : np.ndarray
            Shape ``(n_samples, n_channels)``, dtype ``float32``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_channels, n_bands)``, dtype ``float32``.
            Band order matches ``config.bands``.
        """
        n_samples, n_channels = eeg.shape
        n_bands = len(self._sos_list)
        out = np.empty((n_samples, n_channels, n_bands), dtype=np.float32)

        x = eeg.astype(np.float64)
        for b_idx, sos in enumerate(self._sos_list):
            for ch in range(n_channels):
                out[:, ch, b_idx] = sosfilt(sos, x[:, ch])

        return out

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.config.bands)

    @property
    def n_bands(self) -> int:
        return len(self.config.bands)
