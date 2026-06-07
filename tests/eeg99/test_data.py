"""Phase 1 tests: data loading, preprocessing, filterbank, dataset, cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from eeg99.data.cache import ParquetCache, make_cache_key
from eeg99.data.dataset import EEGWindowDataset, UnlabeledEEGDataset, WindowSpec
from eeg99.data.filterbank import CausalFilterBank
from eeg99.data.loader import CSVGALLoader, SubjectSeries, discover_series
from eeg99.data.preprocess import (
    CausalPreprocessor,
    assert_no_lookahead,
)
from eeg99.utils.constants import (
    CHANNEL_NAMES,
    EVENT_NAMES,
    N_CHANNELS,
    N_EVENTS,
    SFREQ,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_eeg(n_samples: int = 1000, n_channels: int = N_CHANNELS) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_samples, n_channels)).astype(np.float32)


def _make_fake_events(n_samples: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 2, size=(n_samples, N_EVENTS)).astype(np.float32)


def _make_fake_series(subject_id: int = 1, series_id: int = 1) -> SubjectSeries:
    return SubjectSeries(
        subject_id=subject_id,
        series_id=series_id,
        eeg=_make_fake_eeg(),
        events=_make_fake_events(),
    )


def _write_csv_pair(root: Path, subject_id: int, series_id: int, n_samples: int = 200) -> None:
    stem = f"subj{subject_id}_series{series_id}"
    rng = np.random.default_rng(42)

    eeg = rng.normal(size=(n_samples, N_CHANNELS)).astype(np.float32)
    df_data = pd.DataFrame(eeg, columns=list(CHANNEL_NAMES))
    df_data.index = [f"{stem}_{i}" for i in range(n_samples)]
    df_data.index.name = "id"
    df_data.to_csv(root / f"{stem}_data.csv")

    evts = rng.integers(0, 2, size=(n_samples, N_EVENTS)).astype(np.float32)
    df_evts = pd.DataFrame(evts, columns=list(EVENT_NAMES))
    df_evts.index = df_data.index
    df_evts.index.name = "id"
    df_evts.to_csv(root / f"{stem}_events.csv")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_constants_lengths() -> None:
    assert len(CHANNEL_NAMES) == N_CHANNELS == 32
    assert len(EVENT_NAMES) == N_EVENTS == 6
    assert SFREQ == 500.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_loader_roundtrip(tmp_path: Path) -> None:
    _write_csv_pair(tmp_path, subject_id=1, series_id=1)
    loader = CSVGALLoader(tmp_path)
    ss = loader.load(1, 1)

    assert ss.subject_id == 1
    assert ss.series_id == 1
    assert ss.eeg.shape == (200, N_CHANNELS)
    assert ss.eeg.dtype == np.float32
    assert ss.events is not None
    assert ss.events.shape == (200, N_EVENTS)


@pytest.mark.unit
def test_loader_test_split_no_events(tmp_path: Path) -> None:
    """Test-split CSVs have no events file — events must be None."""
    stem = "subj1_series9"
    rng = np.random.default_rng(0)
    eeg = rng.normal(size=(100, N_CHANNELS)).astype(np.float32)
    df = pd.DataFrame(eeg, columns=list(CHANNEL_NAMES))
    df.index.name = "id"
    df.to_csv(tmp_path / f"{stem}_data.csv")

    loader = CSVGALLoader(tmp_path)
    ss = loader.load(1, 9)
    assert ss.events is None


@pytest.mark.unit
def test_discover_series(tmp_path: Path) -> None:
    _write_csv_pair(tmp_path, 1, 1)
    _write_csv_pair(tmp_path, 2, 3)
    pairs = discover_series(tmp_path)
    assert (1, 1) in pairs
    assert (2, 3) in pairs


@pytest.mark.unit
def test_loader_missing_file_raises(tmp_path: Path) -> None:
    loader = CSVGALLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load(99, 99)


@pytest.mark.unit
def test_subject_series_duration() -> None:
    ss = _make_fake_series()
    assert abs(ss.duration_s - 1000 / 500.0) < 1e-6


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_preprocessor_output_shape() -> None:
    eeg = _make_fake_eeg()
    pre = CausalPreprocessor()
    out = pre.fit_transform(eeg)
    assert out.shape == eeg.shape
    assert out.dtype == np.float32


@pytest.mark.unit
def test_preprocessor_no_lookahead() -> None:
    """Critical: causal filter chain must not use future samples."""
    pre = CausalPreprocessor().fit(_make_fake_eeg())
    assert_no_lookahead(pre, n_channels=N_CHANNELS, tol=1e-5)


@pytest.mark.unit
def test_preprocessor_transform_without_fit_raises() -> None:
    pre = CausalPreprocessor()
    with pytest.raises(RuntimeError, match="fit"):
        pre.transform(_make_fake_eeg())


@pytest.mark.unit
def test_preprocessor_reduces_low_frequency() -> None:
    """HP filter at 0.5 Hz should attenuate 0.1 Hz sinusoid."""
    t = np.arange(2000) / SFREQ
    signal = np.sin(2 * np.pi * 0.1 * t).astype(np.float32)
    eeg = np.stack([signal] * N_CHANNELS, axis=1)

    pre = CausalPreprocessor().fit(eeg)
    out = pre.transform(eeg)
    # After 1 second of settling, output should be attenuated significantly
    assert np.abs(out[500:, 0]).mean() < np.abs(signal[500:]).mean() * 0.5


# ---------------------------------------------------------------------------
# Filter bank
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_filterbank_output_shape() -> None:
    fb = CausalFilterBank()
    eeg = _make_fake_eeg(n_samples=500)
    out = fb.apply(eeg)
    assert out.shape == (500, N_CHANNELS, 5)
    assert out.dtype == np.float32


@pytest.mark.unit
def test_filterbank_band_names() -> None:
    fb = CausalFilterBank()
    assert fb.band_names == ("delta", "theta", "alpha", "beta", "gamma")
    assert fb.n_bands == 5


@pytest.mark.unit
def test_filterbank_bands_are_causal() -> None:
    """Verify each band filter is causal via impulse test."""
    fb = CausalFilterBank()
    n = 500
    impulse_idx = 100
    eeg = np.zeros((n, N_CHANNELS), dtype=np.float32)
    eeg[impulse_idx, :] = 1.0

    out = fb.apply(eeg)  # (n, C, 5)
    for b_idx in range(fb.n_bands):
        pre = out[:impulse_idx, :, b_idx]
        assert np.abs(pre).max() < 1e-6, f"Band {fb.band_names[b_idx]} has lookahead"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dataset_length() -> None:
    ss = _make_fake_series()
    spec = WindowSpec(length=125, stride=1)
    ds = EEGWindowDataset([ss], spec)
    assert len(ds) == 1000  # stride=1, T=1000


@pytest.mark.unit
def test_dataset_item_shapes() -> None:
    ss = _make_fake_series()
    spec = WindowSpec(length=125, stride=10)
    ds = EEGWindowDataset([ss], spec)
    window, label, subject_id = ds[0]

    assert window.shape == (N_CHANNELS, 125)
    assert window.dtype == torch.float32
    assert label.shape == (N_EVENTS,)
    assert label.dtype == torch.float32
    assert subject_id.dtype == torch.long


@pytest.mark.unit
def test_dataset_causal_window_first_sample() -> None:
    """First window must be zero-padded (no future samples)."""
    ss = _make_fake_series()
    spec = WindowSpec(length=125, stride=1)
    ds = EEGWindowDataset([ss], spec)
    window, _, _ = ds[0]
    # First 124 samples should be zero-padding
    assert torch.all(window[:, :124] == 0.0)


@pytest.mark.unit
def test_dataset_multi_series() -> None:
    s1 = _make_fake_series(1, 1)
    s2 = _make_fake_series(2, 1)
    spec = WindowSpec(length=50, stride=1)
    ds = EEGWindowDataset([s1, s2], spec)
    assert len(ds) == 2000


@pytest.mark.unit
def test_dataset_rejects_unlabeled() -> None:
    ss = SubjectSeries(1, 9, _make_fake_eeg(), events=None)
    with pytest.raises(ValueError, match="no labels"):
        EEGWindowDataset([ss], WindowSpec(length=50))


@pytest.mark.unit
def test_unlabeled_dataset() -> None:
    ss = SubjectSeries(1, 9, _make_fake_eeg(500), events=None)
    spec = WindowSpec(length=100, stride=5)
    ds = UnlabeledEEGDataset([ss], spec)
    assert len(ds) == 500 // 5
    window, subject_id = ds[0]
    assert window.shape == (N_CHANNELS, 100)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path / ".cache")
    assert cache.get("nonexistent_key") is None


@pytest.mark.unit
def test_cache_put_and_get(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path / ".cache")
    array = np.random.default_rng(0).random((100, 32)).astype(np.float32)
    key = make_cache_key(1, 1, {"sfreq": 500.0})

    cache.put(key, array)
    retrieved = cache.get(key)

    assert retrieved is not None
    assert retrieved.shape == array.shape
    np.testing.assert_allclose(retrieved, array, rtol=1e-5)


@pytest.mark.unit
def test_cache_key_differs_by_config() -> None:
    k1 = make_cache_key(1, 1, {"sfreq": 500.0})
    k2 = make_cache_key(1, 1, {"sfreq": 250.0})
    assert k1 != k2


@pytest.mark.unit
def test_cache_key_differs_by_subject() -> None:
    k1 = make_cache_key(1, 1, {})
    k2 = make_cache_key(2, 1, {})
    assert k1 != k2


@pytest.mark.unit
def test_cache_invalidate(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path / ".cache")
    array = np.zeros((10, 32), dtype=np.float32)
    key = make_cache_key(1, 1, {})

    cache.put(key, array)
    assert cache.get(key) is not None
    removed = cache.invalidate(key)
    assert removed is True
    assert cache.get(key) is None
