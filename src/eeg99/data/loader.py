"""High-throughput IO for WAY-EEG-GAL / Kaggle GAL CSV files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eeg99.utils.constants import (
    CHANNEL_NAMES,
    EVENT_NAMES,
    N_CHANNELS,
    N_EVENTS,
    SFREQ,
)


@dataclass(frozen=True)
class SubjectSeries:
    """One (subject, series) pair of EEG + event labels.

    Attributes
    ----------
    subject_id : int
        1-based subject index (1..12).
    series_id : int
        1-based series index (1..8 train, 9..10 test).
    eeg : np.ndarray
        Shape ``(n_samples, 32)``, dtype ``float32``, 500 Hz.
    events : np.ndarray | None
        Shape ``(n_samples, 6)``, dtype ``float32`` (0/1). ``None`` for test.
    sfreq : float
        Always 500.0 Hz.
    """

    subject_id: int
    series_id: int
    eeg: np.ndarray
    events: np.ndarray | None
    sfreq: float = SFREQ

    def __post_init__(self) -> None:
        assert self.eeg.ndim == 2 and self.eeg.shape[1] == N_CHANNELS
        assert self.eeg.dtype == np.float32
        if self.events is not None:
            assert self.events.ndim == 2 and self.events.shape[1] == N_EVENTS
            assert self.events.shape[0] == self.eeg.shape[0]

    @property
    def n_samples(self) -> int:
        return self.eeg.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq


_DATA_PATTERN = re.compile(r"subj(\d+)_series(\d+)_data\.csv$")


def discover_series(root: Path, split: str = "train") -> list[tuple[int, int]]:
    """Return sorted list of (subject_id, series_id) pairs present on disk.

    Parameters
    ----------
    root : Path
        Dataset root directory (contains ``train/`` and ``test/`` subdirs, or
        data files directly if ``split`` matches a flat directory).
    split : str
        ``"train"`` or ``"test"``.
    """
    directory = root / split if (root / split).is_dir() else root
    pairs: list[tuple[int, int]] = []
    for p in directory.glob("subj*_series*_data.csv"):
        m = _DATA_PATTERN.search(p.name)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return sorted(pairs)


class CSVGALLoader:
    """Read raw CSV pairs into :class:`SubjectSeries`.

    Parameters
    ----------
    root : Path
        Directory containing ``subj{N}_series{M}_data.csv`` files.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def load(self, subject_id: int, series_id: int) -> SubjectSeries:
        """Load one (subject, series) pair from disk.

        Parameters
        ----------
        subject_id : int
            1-based subject index.
        series_id : int
            1-based series index.

        Returns
        -------
        SubjectSeries
            Loaded data. ``events`` is ``None`` if no events file exists (test).
        """
        stem = f"subj{subject_id}_series{series_id}"
        data_path = self._root / f"{stem}_data.csv"
        events_path = self._root / f"{stem}_events.csv"

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        eeg = (
            pd.read_csv(data_path, usecols=list(CHANNEL_NAMES))
            .to_numpy(dtype=np.float32)
        )

        events: np.ndarray | None = None
        if events_path.exists():
            events = (
                pd.read_csv(events_path, usecols=list(EVENT_NAMES))
                .to_numpy(dtype=np.float32)
            )

        return SubjectSeries(
            subject_id=subject_id,
            series_id=series_id,
            eeg=eeg,
            events=events,
        )

    def available(self) -> list[tuple[int, int]]:
        """Return sorted list of (subject_id, series_id) pairs present in root."""
        return discover_series(self._root)
