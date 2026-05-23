"""Smoke contract: BaseModel exposes the predict_proba / metadata interface.

Phase 1+ will add a concrete model and exercise this contract end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from eeg99.models.base import BaseModel, ModelMetadata


class _DummyModel(BaseModel):
    """Minimal concrete instance used only for contract validation in tests."""

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="dummy",
            window_samples=500,
            n_channels=32,
            n_events=6,
            n_subjects=12,
            causal=True,
            n_params=0,
            architecture_family="eegnet",
        )

    def predict_proba(self, eeg: np.ndarray, subject_id: int) -> np.ndarray:
        return np.zeros((eeg.shape[0], 6), dtype=np.float32)


@pytest.mark.unit
def test_base_model_contract() -> None:
    model = _DummyModel()
    assert model.metadata.causal is True
    out = model.predict_proba(np.zeros((10, 32), dtype=np.float32), subject_id=1)
    assert out.shape == (10, 6)
    assert out.dtype == np.float32
