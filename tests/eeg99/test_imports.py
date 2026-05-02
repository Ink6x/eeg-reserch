"""Smoke test: every eeg99 module imports cleanly.

Catches typos, missing __init__.py, and accidental circular imports before
they reach Phase 1.
"""
from __future__ import annotations

import importlib

import pytest

MODULES: list[str] = [
    "eeg99",
    "eeg99.data",
    "eeg99.data.loader",
    "eeg99.data.preprocess",
    "eeg99.data.filterbank",
    "eeg99.data.augmentation",
    "eeg99.data.dataset",
    "eeg99.data.cache",
    "eeg99.features",
    "eeg99.features.fbcsp",
    "eeg99.features.riemannian",
    "eeg99.features.time_domain",
    "eeg99.features.band_power",
    "eeg99.models",
    "eeg99.models.base",
    "eeg99.models.eegnet_plus",
    "eeg99.models.conformer_hier",
    "eeg99.models.tcn_multi",
    "eeg99.models.hybrid",
    "eeg99.models.ssl_bendr",
    "eeg99.models.adapter",
    "eeg99.losses",
    "eeg99.losses.focal",
    "eeg99.losses.bce_weighted",
    "eeg99.losses.event_order",
    "eeg99.losses.ssl_losses",
    "eeg99.ensemble",
    "eeg99.ensemble.bagging",
    "eeg99.ensemble.stacking",
    "eeg99.ensemble.selection",
    "eeg99.ensemble.tta",
    "eeg99.ensemble.calibration",
    "eeg99.training",
    "eeg99.training.trainer",
    "eeg99.training.schedulers",
    "eeg99.training.optimizers",
    "eeg99.training.metrics",
    "eeg99.pipeline",
    "eeg99.pipeline.stage1_preprocess",
    "eeg99.pipeline.stage2_ssl",
    "eeg99.pipeline.stage3_supervised",
    "eeg99.pipeline.stage4_bagging",
    "eeg99.pipeline.stage5_stacking",
    "eeg99.pipeline.stage6_inference",
    "eeg99.utils",
    "eeg99.utils.seed",
    "eeg99.utils.config",
    "eeg99.utils.registry",
    "eeg99.utils.logging",
]


@pytest.mark.unit
@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.unit
def test_version_string() -> None:
    import eeg99

    assert isinstance(eeg99.__version__, str)
    assert eeg99.__version__.count(".") >= 2
