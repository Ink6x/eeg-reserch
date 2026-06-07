"""Phase 2 (W3) tests: classical feature extractors."""
from __future__ import annotations

import numpy as np
import pytest

from eeg99.features.band_power import BandPowerExtractor
from eeg99.features.fbcsp import FBCSPConfig, FBCSPExtractor
from eeg99.features.riemannian import RiemannianExtractor
from eeg99.features.time_domain import N_FEATURES_PER_CHANNEL, TimeDomainExtractor
from eeg99.utils.constants import N_CHANNELS, N_EVENTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_WIN = 60
WIN_LEN = 250  # samples


def _make_windows(n_windows: int = N_WIN, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.normal(size=(n_windows, N_CHANNELS, WIN_LEN)).astype(np.float32)


def _make_labels(n_windows: int = N_WIN, rng_seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    # Ensure each event has at least a few positives
    y = np.zeros((n_windows, N_EVENTS), dtype=np.float32)
    for e in range(N_EVENTS):
        pos_idx = rng.choice(n_windows, size=max(5, n_windows // 4), replace=False)
        y[pos_idx, e] = 1.0
    return y


# ---------------------------------------------------------------------------
# FBCSP
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fbcsp_output_shape() -> None:
    cfg = FBCSPConfig(n_components=4)
    ext = FBCSPExtractor(cfg)
    X = _make_windows()
    y = _make_labels()
    feats = ext.fit_transform(X, y)
    expected_dim = N_EVENTS * len(cfg.bands) * cfg.n_components
    assert feats.shape == (N_WIN, expected_dim)
    assert feats.dtype == np.float32


@pytest.mark.unit
def test_fbcsp_transform_without_fit_raises() -> None:
    ext = FBCSPExtractor()
    with pytest.raises(RuntimeError, match="fit"):
        ext.transform(_make_windows())


@pytest.mark.unit
def test_fbcsp_deterministic() -> None:
    X = _make_windows()
    y = _make_labels()
    f1 = FBCSPExtractor().fit_transform(X, y)
    f2 = FBCSPExtractor().fit_transform(X, y)
    np.testing.assert_array_equal(f1, f2)


@pytest.mark.unit
def test_fbcsp_no_nan() -> None:
    feats = FBCSPExtractor().fit_transform(_make_windows(), _make_labels())
    assert not np.isnan(feats).any(), "FBCSP features contain NaN"


@pytest.mark.unit
def test_fbcsp_feature_dim_property() -> None:
    cfg = FBCSPConfig(n_components=2)
    ext = FBCSPExtractor(cfg)
    assert ext.feature_dim == N_EVENTS * len(cfg.bands) * 2


@pytest.mark.unit
def test_fbcsp_degenerate_class(caplog) -> None:
    """If one class is empty, extractor should not crash."""
    X = _make_windows()
    y = np.zeros((N_WIN, N_EVENTS), dtype=np.float32)  # all zeros (no events)
    feats = FBCSPExtractor().fit_transform(X, y)
    assert not np.isnan(feats).any()


# ---------------------------------------------------------------------------
# Riemannian
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_riemannian_output_shape() -> None:
    ext = RiemannianExtractor()
    X = _make_windows()
    feats = ext.fit_transform(X)
    expected_dim = N_CHANNELS * (N_CHANNELS + 1) // 2
    assert feats.shape == (N_WIN, expected_dim)
    assert feats.dtype == np.float32


@pytest.mark.unit
def test_riemannian_transform_without_fit_raises() -> None:
    ext = RiemannianExtractor()
    with pytest.raises(RuntimeError, match="fit"):
        ext.transform(_make_windows())


@pytest.mark.unit
def test_riemannian_no_nan() -> None:
    ext = RiemannianExtractor()
    feats = ext.fit_transform(_make_windows())
    assert not np.isnan(feats).any(), "Riemannian features contain NaN"


@pytest.mark.unit
def test_riemannian_feature_dim_property() -> None:
    ext = RiemannianExtractor()
    assert ext.feature_dim == N_CHANNELS * (N_CHANNELS + 1) // 2


@pytest.mark.unit
def test_riemannian_different_windows_differ() -> None:
    """Two different windows must produce different tangent-space vectors."""
    ext = RiemannianExtractor()
    X = _make_windows(n_windows=20)
    ext.fit(X)
    f0 = ext.transform(X[:1])
    f1 = ext.transform(X[1:2])
    assert not np.allclose(f0, f1), "All tangent vectors are identical"


# ---------------------------------------------------------------------------
# Band power
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_band_power_output_shape() -> None:
    ext = BandPowerExtractor()
    X = _make_windows()
    feats = ext.transform(X)
    assert feats.shape == (N_WIN, N_CHANNELS * 5)
    assert feats.dtype == np.float32


@pytest.mark.unit
def test_band_power_feature_dim_property() -> None:
    ext = BandPowerExtractor()
    assert ext.feature_dim == N_CHANNELS * 5


@pytest.mark.unit
def test_band_power_no_nan() -> None:
    feats = BandPowerExtractor().transform(_make_windows())
    assert not np.isnan(feats).any()


@pytest.mark.unit
def test_band_power_stateless() -> None:
    """BandPowerExtractor is stateless — two calls must match."""
    X = _make_windows()
    f1 = BandPowerExtractor().transform(X)
    f2 = BandPowerExtractor().transform(X)
    np.testing.assert_array_equal(f1, f2)


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_time_domain_output_shape() -> None:
    ext = TimeDomainExtractor()
    X = _make_windows()
    feats = ext.transform(X)
    assert feats.shape == (N_WIN, N_CHANNELS * N_FEATURES_PER_CHANNEL)
    assert feats.dtype == np.float32


@pytest.mark.unit
def test_time_domain_feature_dim_property() -> None:
    ext = TimeDomainExtractor()
    assert ext.feature_dim == N_CHANNELS * 5


@pytest.mark.unit
def test_time_domain_no_nan() -> None:
    feats = TimeDomainExtractor().transform(_make_windows())
    assert not np.isnan(feats).any()


@pytest.mark.unit
def test_time_domain_rms_positive() -> None:
    """RMS must be non-negative for any signal."""
    feats = TimeDomainExtractor().transform(_make_windows())
    # First feature per channel is RMS (columns 0, 5, 10, ...)
    rms_cols = [i * N_FEATURES_PER_CHANNEL for i in range(N_CHANNELS)]
    assert (feats[:, rms_cols] >= 0).all()


@pytest.mark.unit
def test_time_domain_zero_signal() -> None:
    """Zero input should produce zero RMS and zero line length."""
    X = np.zeros((1, N_CHANNELS, WIN_LEN), dtype=np.float32)
    feats = TimeDomainExtractor().transform(X)
    rms_cols = [i * N_FEATURES_PER_CHANNEL for i in range(N_CHANNELS)]
    ll_cols = [i * N_FEATURES_PER_CHANNEL + 1 for i in range(N_CHANNELS)]
    assert np.allclose(feats[0, rms_cols], 0.0, atol=1e-6)
    assert np.allclose(feats[0, ll_cols], 0.0, atol=1e-6)
