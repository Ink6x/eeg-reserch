"""Multi-subject, multi-window PyTorch Dataset for EEG decoding."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eeg99.data.loader import SubjectSeries
from eeg99.utils.constants import SFREQ


@dataclass(frozen=True)
class WindowSpec:
    """Specification for one causal sliding window.

    Attributes
    ----------
    length : int
        Window length in samples. The window covers ``[t - length + 1, t]``
        (strictly causal: uses only past and current samples).
    stride : int
        Step between consecutive windows in samples.
    sfreq : float
        Sampling frequency (for informational ``length_s`` property).
    """

    length: int
    stride: int = 1
    sfreq: float = SFREQ

    @property
    def length_s(self) -> float:
        return self.length / self.sfreq


# Canonical window sizes used across the project
WINDOW_SPECS: dict[str, WindowSpec] = {
    "250ms": WindowSpec(length=125),
    "500ms": WindowSpec(length=250),
    "1000ms": WindowSpec(length=500),
    "2000ms": WindowSpec(length=1000),
    "4000ms": WindowSpec(length=2000),
}


class EEGWindowDataset(Dataset):
    """Sliding-window dataset over one or more :class:`SubjectSeries`.

    Each item is a ``(window, label, subject_id)`` triple where:

    * ``window`` — ``float32`` tensor of shape ``(n_channels, window_length)``
    * ``label``  — ``float32`` tensor of shape ``(n_events,)`` at the last sample
    * ``subject_id`` — ``int`` tensor (scalar)

    Only labeled series (``events is not None``) are supported. For test-set
    (SSL pre-training) use :class:`UnlabeledEEGDataset`.

    Parameters
    ----------
    series_list : list[SubjectSeries]
        Pre-processed series to include. All must have ``events`` set.
    spec : WindowSpec
        Window size and stride.
    """

    def __init__(
        self,
        series_list: list[SubjectSeries],
        spec: WindowSpec,
    ) -> None:
        for s in series_list:
            if s.events is None:
                raise ValueError(
                    f"subj{s.subject_id}_series{s.series_id} has no labels. "
                    "Use UnlabeledEEGDataset for unlabeled data."
                )

        self._spec = spec
        self._windows: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._build_index(series_list)

    def _build_index(self, series_list: list[SubjectSeries]) -> None:
        """Pre-compute (eeg_window, label, subject_id) for every valid position."""
        pad = self._spec.length - 1
        stride = self._spec.stride

        for s in series_list:
            eeg = s.eeg          # (T, C)
            events = s.events    # (T, E)
            assert events is not None
            T = eeg.shape[0]

            # Pad the start so every sample from t=0 has a full causal window
            eeg_padded = np.concatenate(
                [np.zeros((pad, eeg.shape[1]), dtype=np.float32), eeg], axis=0
            )

            for t in range(0, T, stride):
                window = eeg_padded[t : t + self._spec.length]  # (W, C)
                label = events[t]                                 # (E,)
                self._windows.append((window.T.copy(), label.copy(), s.subject_id))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, label, subject_id = self._windows[idx]
        return (
            torch.from_numpy(window),                             # (C, W) float32
            torch.from_numpy(label),                              # (E,) float32
            torch.tensor(subject_id, dtype=torch.long),           # scalar
        )


class UnlabeledEEGDataset(Dataset):
    """Sliding-window dataset for SSL pre-training (no labels required).

    Each item is a ``(window, subject_id)`` pair.
    Accepts both labeled and unlabeled :class:`SubjectSeries`.
    """

    def __init__(
        self,
        series_list: list[SubjectSeries],
        spec: WindowSpec,
    ) -> None:
        self._spec = spec
        self._windows: list[tuple[np.ndarray, int]] = []
        self._build_index(series_list)

    def _build_index(self, series_list: list[SubjectSeries]) -> None:
        pad = self._spec.length - 1
        stride = self._spec.stride

        for s in series_list:
            eeg = s.eeg
            T = eeg.shape[0]
            eeg_padded = np.concatenate(
                [np.zeros((pad, eeg.shape[1]), dtype=np.float32), eeg], axis=0
            )
            for t in range(0, T, stride):
                window = eeg_padded[t : t + self._spec.length].T.copy()
                self._windows.append((window, s.subject_id))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window, subject_id = self._windows[idx]
        return (
            torch.from_numpy(window),
            torch.tensor(subject_id, dtype=torch.long),
        )
