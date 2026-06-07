"""Phase 4-6 tests: ensemble infrastructure."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from eeg99.ensemble.bagging import (
    ModelSpec,
    OOFStore,
    build_model_grid,
    geometric_mean_ensemble,
)
from eeg99.ensemble.calibration import HierarchicalBayesianCalibration
from eeg99.ensemble.selection import (
    compute_pairwise_correlation,
    greedy_forward_selection,
    rank_by_mean_auc,
)
from eeg99.ensemble.tta import TTAConfig, TTAWrapper
from eeg99.losses.event_order import EventOrderConfig, EventOrderConsistencyLoss
from eeg99.losses.focal import FocalLoss
from eeg99.utils.constants import N_EVENTS, N_SUBJECTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MODELS = 6
N_SAMPLES = 80
N_EVENTS_ = N_EVENTS


def _preds(M: int = N_MODELS, N: int = N_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 0.9, (M, N, N_EVENTS_)).astype(np.float32)


def _labels(N: int = N_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(1)
    y = np.zeros((N, N_EVENTS_), dtype=np.float32)
    for e in range(N_EVENTS_):
        idx = rng.choice(N, size=N // 5, replace=False)
        y[idx, e] = 1.0
    return y


def _subject_ids(N: int = N_SAMPLES) -> np.ndarray:
    return np.repeat(np.arange(N_SUBJECTS), N // N_SUBJECTS + 1)[:N]


# ---------------------------------------------------------------------------
# ModelSpec / build_model_grid
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_model_grid_count() -> None:
    grid = build_model_grid()
    assert len(grid) == 100   # 5 arch × 4 window × 5 seed


@pytest.mark.unit
def test_model_spec_id_unique() -> None:
    grid = build_model_grid()
    ids = [s.model_id for s in grid]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_model_spec_roundtrip() -> None:
    spec = ModelSpec("eegnet", 250, 3)
    assert ModelSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------------------------------
# OOFStore
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_oof_store_save_load() -> None:
    spec = ModelSpec("eegnet", 250, 0)
    preds = np.random.default_rng(0).uniform(size=(N_SAMPLES, N_EVENTS_)).astype(np.float32)
    aucs = [0.8] * N_EVENTS_

    with tempfile.TemporaryDirectory() as tmpdir:
        store = OOFStore(tmpdir)
        store.save_oof(spec, preds, aucs)
        loaded_preds, loaded_aucs = store.load_oof(spec)

    np.testing.assert_allclose(loaded_preds, preds)
    assert loaded_aucs == aucs


@pytest.mark.unit
def test_oof_store_has_oof() -> None:
    spec = ModelSpec("tcn", 500, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OOFStore(tmpdir)
        assert not store.has_oof(spec)
        store.save_oof(spec, np.zeros((10, N_EVENTS_), dtype=np.float32), [0.5] * N_EVENTS_)
        assert store.has_oof(spec)


@pytest.mark.unit
def test_oof_store_load_all() -> None:
    specs = [ModelSpec("eegnet", 250, s) for s in range(3)]
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OOFStore(tmpdir)
        for spec in specs:
            store.save_oof(spec, np.zeros((10, N_EVENTS_), np.float32), [0.5] * N_EVENTS_)
        all_preds, all_aucs, loaded_specs = store.load_all()

    assert all_preds.shape == (3, 10, N_EVENTS_)
    assert all_aucs.shape == (3, N_EVENTS_)
    assert len(loaded_specs) == 3


# ---------------------------------------------------------------------------
# geometric_mean_ensemble
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_geometric_mean_shape() -> None:
    p = _preds()
    out = geometric_mean_ensemble(p)
    assert out.shape == (N_SAMPLES, N_EVENTS_)
    assert out.dtype == np.float32


@pytest.mark.unit
def test_geometric_mean_range() -> None:
    p = _preds()
    out = geometric_mean_ensemble(p)
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.unit
def test_geometric_mean_weighted() -> None:
    p = _preds()
    w = np.ones(N_MODELS)
    w[0] = 10.0   # heavily weight first model
    out_w = geometric_mean_ensemble(p, weights=w)
    assert out_w.shape == (N_SAMPLES, N_EVENTS_)


@pytest.mark.unit
def test_geometric_mean_single_model() -> None:
    p_single = _preds(M=1)
    out = geometric_mean_ensemble(p_single)
    np.testing.assert_allclose(out, p_single[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rank_by_mean_auc_shape() -> None:
    p = _preds()
    y = _labels()
    ranked = rank_by_mean_auc(p, y)
    assert ranked.shape == (N_MODELS,)
    assert set(ranked.tolist()) == set(range(N_MODELS))


@pytest.mark.unit
def test_greedy_forward_selection_returns_indices() -> None:
    p = _preds()
    y = _labels()
    selected = greedy_forward_selection(p, y, target_n=3)
    assert len(selected) >= 1
    assert all(0 <= idx < N_MODELS for idx in selected)


@pytest.mark.unit
def test_compute_pairwise_correlation_shape() -> None:
    p = _preds()
    corr = compute_pairwise_correlation(p)
    assert corr.shape == (N_MODELS, N_MODELS)
    # Diagonal should be ~1
    np.testing.assert_allclose(np.diag(corr), np.ones(N_MODELS), atol=1e-4)


# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------

class _DummyModel(torch.nn.Module):
    """Always returns zeros (logits)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, N_EVENTS_)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.zeros(B, N_EVENTS_)


@pytest.mark.unit
def test_tta_output_shape() -> None:
    model = _DummyModel()
    tta = TTAWrapper(model, TTAConfig(n_aug=3))
    x = torch.randn(4, 32, 500)
    sid = torch.zeros(4, dtype=torch.long)
    out = tta.predict(x, sid)
    assert out.shape == (4, N_EVENTS_)


@pytest.mark.unit
def test_tta_range() -> None:
    model = _DummyModel()
    tta = TTAWrapper(model)
    x = torch.randn(2, 32, 250)
    sid = torch.zeros(2, dtype=torch.long)
    out = tta.predict(x, sid)
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.unit
def test_tta_identity_model_constant() -> None:
    """Dummy model always 0-logits → sigmoid(0)=0.5."""
    model = _DummyModel()
    tta = TTAWrapper(model)
    x = torch.randn(2, 32, 250)
    sid = torch.zeros(2, dtype=torch.long)
    out = tta.predict(x, sid)
    np.testing.assert_allclose(out.numpy(), 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# HierarchicalBayesianCalibration
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_calibration_fit_predict_shape() -> None:
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(N_SAMPLES, N_EVENTS_)).astype(np.float64)
    labels = (rng.uniform(size=(N_SAMPLES, N_EVENTS_)) > 0.7).astype(np.float32)
    sids = _subject_ids()

    cal = HierarchicalBayesianCalibration(N_SUBJECTS, N_EVENTS_)
    cal.fit(logits, labels, sids)
    probs = cal.predict_proba(logits, sids)

    assert probs.shape == (N_SAMPLES, N_EVENTS_)
    assert (probs >= 0).all() and (probs <= 1).all()


@pytest.mark.unit
def test_calibration_unfitted_raises() -> None:
    cal = HierarchicalBayesianCalibration(N_SUBJECTS, N_EVENTS_)
    with pytest.raises(RuntimeError, match="fit"):
        cal.predict_proba(np.zeros((10, N_EVENTS_)), np.zeros(10, dtype=int))


@pytest.mark.unit
def test_calibration_no_nan() -> None:
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(40, N_EVENTS_))
    labels = (rng.uniform(size=(40, N_EVENTS_)) > 0.6).astype(np.float32)
    sids = np.repeat(np.arange(4), 10)

    cal = HierarchicalBayesianCalibration(4, N_EVENTS_)
    cal.fit(logits, labels, sids)
    probs = cal.predict_proba(logits, sids)
    assert not np.isnan(probs).any()


# ---------------------------------------------------------------------------
# EventOrderConsistencyLoss
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_event_order_loss_zero_for_correct_order() -> None:
    """Perfect ordering (p_0 >> p_1 >> ... >> p_5) should give near-zero loss."""
    cfg = EventOrderConfig(weight=1.0, margin=0.0)
    loss_fn = EventOrderConsistencyLoss(cfg)
    probs = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.2, 0.1]])
    loss = loss_fn(probs)
    assert loss.item() < 1e-6


@pytest.mark.unit
def test_event_order_loss_positive_for_violation() -> None:
    """Reversed order should give positive loss."""
    cfg = EventOrderConfig(weight=1.0, margin=0.0)
    loss_fn = EventOrderConsistencyLoss(cfg)
    probs = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.8, 0.9]])  # reversed
    loss = loss_fn(probs)
    assert loss.item() > 0


@pytest.mark.unit
def test_event_order_loss_sequence() -> None:
    """Works with (B, T, E) input."""
    loss_fn = EventOrderConsistencyLoss()
    probs = torch.rand(2, 10, N_EVENTS_)
    loss = loss_fn(probs)
    assert loss.item() >= 0


@pytest.mark.unit
def test_event_order_loss_differentiable() -> None:
    loss_fn = EventOrderConsistencyLoss()
    probs = torch.rand(4, N_EVENTS_, requires_grad=True)
    loss = loss_fn(probs)
    loss.backward()
    assert probs.grad is not None


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_focal_loss_scalar() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(8, N_EVENTS_)
    targets = (torch.rand(8, N_EVENTS_) > 0.7).float()
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # scalar


@pytest.mark.unit
def test_focal_loss_positive() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(8, N_EVENTS_)
    targets = (torch.rand(8, N_EVENTS_) > 0.5).float()
    assert loss_fn(logits, targets).item() > 0


@pytest.mark.unit
def test_focal_loss_zero_when_perfect() -> None:
    """Near-perfect prediction → loss near 0."""
    loss_fn = FocalLoss()
    big = 20.0
    targets = torch.ones(4, N_EVENTS_)
    logits = targets * big
    assert loss_fn(logits, targets).item() < 1e-3


@pytest.mark.unit
def test_focal_loss_gradient() -> None:
    loss_fn = FocalLoss()
    logits = torch.randn(4, N_EVENTS_, requires_grad=True)
    targets = (torch.rand(4, N_EVENTS_) > 0.5).float()
    loss_fn(logits, targets).backward()
    assert logits.grad is not None
