"""Dataset-level constants for the WAY-EEG-GAL / Kaggle GAL dataset.

Source: Luciw et al. (2014) Scientific Data 1:140047.
"""
from __future__ import annotations

SFREQ: float = 500.0  # Hz

N_CHANNELS: int = 32
N_EVENTS: int = 6
N_SUBJECTS: int = 12
N_TRAIN_SERIES: int = 8
N_TEST_SERIES: int = 2

CHANNEL_NAMES: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
)

EVENT_NAMES: tuple[str, ...] = (
    "HandStart",
    "FirstDigitTouch",
    "BothStartLoadPhase",
    "LiftOff",
    "Replace",
    "BothReleased",
)

# Fixed event order (HandStart always precedes FirstDigitTouch, etc.)
# Used by EventOrderConsistency Loss (Component 7).
EVENT_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

assert len(CHANNEL_NAMES) == N_CHANNELS
assert len(EVENT_NAMES) == N_EVENTS
