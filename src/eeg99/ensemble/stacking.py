"""LightGBM stacking meta-learner on 5-fold OOF predictions (Component 9).

The meta-learner takes OOF predictions from M base models as features and
learns to predict the true labels.  One LightGBM model is trained per event
so that event-specific patterns in the ensemble can be captured.

Cross-validation
----------------
Subject-stratified 5-fold: all windows from the same subject stay together
in the same fold, preventing subject-level leakage.

Usage
-----
::

    meta = StackingMetaLearner(n_folds=5)
    meta.fit(oof_preds, labels, subject_ids)   # (M, N, E), (N, E), (N,)
    test_preds = meta.predict(test_preds_stack) # (M, N_test, E) → (N_test, E)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["StackingMetaLearner", "StackingConfig"]


@dataclass(frozen=True)
class StackingConfig:
    n_folds: int = 5
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "n_jobs": -1,
    })


class StackingMetaLearner:
    """Per-event LightGBM meta-learner trained on base model OOF predictions.

    Parameters
    ----------
    cfg : StackingConfig
    """

    def __init__(self, cfg: StackingConfig | None = None) -> None:
        self.cfg = cfg or StackingConfig()
        self._models: list[list] = []   # [event_idx][fold_idx] → lgbm model
        self._fitted = False
        self._n_events: int = 0

    # ------------------------------------------------------------------

    def fit(
        self,
        oof_preds: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
    ) -> "StackingMetaLearner":
        """Fit the stacking meta-learner.

        Parameters
        ----------
        oof_preds   : (n_models, n_samples, n_events)
        labels      : (n_samples, n_events)
        subject_ids : (n_samples,) int — used for subject-stratified CV
        """
        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError as exc:
            raise ImportError("lightgbm is required for stacking") from exc

        n_models, n_samples, n_events = oof_preds.shape
        self._n_events = n_events
        # Feature matrix: stack model predictions → (n_samples, n_models * n_events)
        # We train one model per event, using all model OOF preds as features.
        folds = _subject_stratified_folds(subject_ids, self.cfg.n_folds)

        self._models = []
        for e in range(n_events):
            y = labels[:, e]
            X = oof_preds[:, :, e].T   # (n_samples, n_models)
            fold_models = _fit_lgbm_cv(X, y, folds, self.cfg.lgbm_params)
            self._models.append(fold_models)

        self._fitted = True
        return self

    def predict(self, test_preds: np.ndarray) -> np.ndarray:
        """Predict on test set base model outputs.

        Parameters
        ----------
        test_preds : (n_models, n_test_samples, n_events)

        Returns
        -------
        np.ndarray  (n_test_samples, n_events)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        n_models, n_test, n_events = test_preds.shape
        result = np.zeros((n_test, n_events), dtype=np.float32)

        for e in range(n_events):
            X_test = test_preds[:, :, e].T   # (n_test, n_models)
            fold_preds = np.array([
                m.predict(X_test) for m in self._models[e]
            ])                               # (n_folds, n_test)
            result[:, e] = fold_preds.mean(axis=0).astype(np.float32)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subject_stratified_folds(
    subject_ids: np.ndarray,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) pairs, keeping subjects intact per fold."""
    unique_subjects = np.unique(subject_ids)
    np.random.default_rng(42).shuffle(unique_subjects)
    subject_chunks = np.array_split(unique_subjects, n_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in subject_chunks:
        val_mask = np.isin(subject_ids, chunk)
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        folds.append((train_idx, val_idx))

    return folds


def _fit_lgbm_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
) -> list:
    """Fit one LightGBM model per fold. Returns list of fitted models."""
    import lightgbm as lgb

    models = []
    for train_idx, val_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            # Degenerate fold: use a trivial constant model
            models.append(_ConstantPredictor(float(y[train_idx].mean())))
            continue

        fit_params = dict(params)
        n_estimators = fit_params.pop("n_estimators", 500)
        early = fit_params.pop("early_stopping_rounds", 50)

        model = lgb.LGBMClassifier(n_estimators=n_estimators, **fit_params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(early, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        models.append(model)

    return models


class _ConstantPredictor:
    """Trivial predictor returning a constant probability (degenerate fold)."""

    def __init__(self, p: float) -> None:
        self._p = p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._p, dtype=np.float32)
