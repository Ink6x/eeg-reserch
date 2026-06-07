"""Massive bagging infrastructure (Component 8).

Grid: 5 architectures × 4 windows × 5 seeds = 100 candidates.
Each candidate is fully specified by a ModelSpec.  OOF predictions
and AUC scores are persisted in a directory so training can be
interrupted and resumed.

Typical use
-----------
1. Generate the full 100-spec grid: ``build_model_grid()``
2. For each spec, train and call ``OOFStore.save_oof()``
3. After all training: ``OOFStore.load_all()`` → select via
   ``ensemble.selection.greedy_forward_selection``
4. Build ensemble predictions: ``geometric_mean_ensemble()``
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ModelSpec",
    "OOFStore",
    "build_model_grid",
    "geometric_mean_ensemble",
]

_ARCHITECTURES = ("eegnet", "conformer", "tcn", "hybrid", "ssl")
_WINDOW_SAMPLES = (125, 250, 500, 1000)   # 4 window sizes
_SEEDS = (0, 1, 2, 3, 4)                  # 5 seeds


@dataclass(frozen=True)
class ModelSpec:
    """Fully specifies one model in the 100-candidate pool."""

    architecture: str       # one of _ARCHITECTURES
    window_samples: int     # one of _WINDOW_SAMPLES
    seed: int               # one of _SEEDS

    @property
    def model_id(self) -> str:
        return f"{self.architecture}_w{self.window_samples}_s{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelSpec:
        return cls(**d)


def build_model_grid() -> list[ModelSpec]:
    """Return all 100 ModelSpecs (5 arch × 4 window × 5 seed)."""
    return [
        ModelSpec(arch, win, seed)
        for arch in _ARCHITECTURES
        for win in _WINDOW_SAMPLES
        for seed in _SEEDS
    ]


# ---------------------------------------------------------------------------
# OOF store
# ---------------------------------------------------------------------------

class OOFStore:
    """Persist and load OOF predictions + metadata to/from a directory.

    Directory layout::

        store_dir/
          <model_id>/
            oof_preds.npy    # (n_samples, n_events) float32
            oof_auc.json     # {"mean_auc": float, "per_event": [float,...]}
            spec.json        # ModelSpec dict

    Parameters
    ----------
    store_dir : Path | str
        Root directory for storing OOF data.
    """

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def save_oof(
        self,
        spec: ModelSpec,
        oof_preds: np.ndarray,
        per_event_auc: list[float],
    ) -> Path:
        """Persist OOF predictions and AUC for one model.

        Parameters
        ----------
        spec : ModelSpec
        oof_preds : np.ndarray  (n_samples, n_events) float32
        per_event_auc : list[float]  length n_events

        Returns
        -------
        Path  directory where data was saved
        """
        model_dir = self.store_dir / spec.model_id
        model_dir.mkdir(exist_ok=True)

        np.save(model_dir / "oof_preds.npy", oof_preds.astype(np.float32))

        mean_auc = float(np.mean(per_event_auc))
        with open(model_dir / "oof_auc.json", "w") as f:
            json.dump({"mean_auc": mean_auc, "per_event": per_event_auc}, f)

        with open(model_dir / "spec.json", "w") as f:
            json.dump(spec.to_dict(), f)

        return model_dir

    def load_oof(self, spec: ModelSpec) -> tuple[np.ndarray, list[float]]:
        """Load OOF predictions and per-event AUC for one model."""
        model_dir = self.store_dir / spec.model_id
        preds = np.load(model_dir / "oof_preds.npy")
        with open(model_dir / "oof_auc.json") as f:
            meta = json.load(f)
        return preds, meta["per_event"]

    def has_oof(self, spec: ModelSpec) -> bool:
        model_dir = self.store_dir / spec.model_id
        return (model_dir / "oof_preds.npy").exists()

    def load_all(self) -> tuple[np.ndarray, np.ndarray, list[ModelSpec]]:
        """Load all available OOF predictions.

        Returns
        -------
        all_preds : np.ndarray  (n_models, n_samples, n_events)
        all_aucs  : np.ndarray  (n_models, n_events)
        specs     : list[ModelSpec]
        """
        specs: list[ModelSpec] = []
        preds_list: list[np.ndarray] = []
        aucs_list: list[list[float]] = []

        for model_dir in sorted(self.store_dir.iterdir()):
            spec_path = model_dir / "spec.json"
            if not spec_path.exists():
                continue
            with open(spec_path) as f:
                spec = ModelSpec.from_dict(json.load(f))
            preds, per_event_auc = self.load_oof(spec)
            specs.append(spec)
            preds_list.append(preds)
            aucs_list.append(per_event_auc)

        if not preds_list:
            raise FileNotFoundError(f"No OOF data found in {self.store_dir}")

        return (
            np.stack(preds_list, axis=0),   # (M, N, E)
            np.array(aucs_list),             # (M, E)
            specs,
        )

    def iter_specs(self) -> Iterator[ModelSpec]:
        """Yield all specs present in the store."""
        for model_dir in sorted(self.store_dir.iterdir()):
            spec_path = model_dir / "spec.json"
            if spec_path.exists():
                with open(spec_path) as f:
                    yield ModelSpec.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Ensemble utilities
# ---------------------------------------------------------------------------

def geometric_mean_ensemble(
    preds: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute (weighted) geometric mean across models.

    Parameters
    ----------
    preds : np.ndarray  (n_models, n_samples, n_events)
    weights : np.ndarray | None  (n_models,) — uniform if None

    Returns
    -------
    np.ndarray  (n_samples, n_events)
    """
    eps = 1e-7
    log_p = np.log(np.clip(preds, eps, 1 - eps))  # (M, N, E)

    if weights is None:
        weights = np.ones(preds.shape[0], dtype=np.float64)

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    # Weighted sum of logs → exp
    weighted_log = (log_p * weights[:, None, None]).sum(axis=0)  # (N, E)
    return np.exp(weighted_log).astype(np.float32)
