"""Diversity-based model selection (greedy forward selection).

Algorithm
---------
1. Start with the single best-AUC model.
2. Greedily add the model that maximises ensemble AUC on the OOF set.
3. Stop when ``target_n`` models are selected or adding any model hurts.

References
----------
Caruana et al. (2004) "Ensemble Selection from Libraries of Models".
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from eeg99.ensemble.bagging import geometric_mean_ensemble

__all__ = [
    "greedy_forward_selection",
    "compute_pairwise_correlation",
    "rank_by_mean_auc",
]


def greedy_forward_selection(
    oof_preds: np.ndarray,
    labels: np.ndarray,
    target_n: int = 40,
    max_rounds: int = 200,
) -> list[int]:
    """Select a diverse ensemble via greedy forward selection on OOF AUC.

    Parameters
    ----------
    oof_preds : np.ndarray  (n_models, n_samples, n_events)
    labels    : np.ndarray  (n_samples, n_events) binary
    target_n  : int  desired number of models
    max_rounds : int  maximum greedy rounds (with replacement allowed)

    Returns
    -------
    list[int]  indices of selected models (may contain duplicates if with-
               replacement improves AUC)
    """
    n_models = oof_preds.shape[0]
    selected: list[int] = []

    # Start with best single model
    single_aucs = np.array([_mean_auc(oof_preds[i], labels) for i in range(n_models)])
    best_start = int(np.argmax(single_aucs))
    selected.append(best_start)
    best_ensemble_auc = single_aucs[best_start]

    for _ in range(min(target_n - 1, max_rounds)):
        best_candidate = -1
        best_auc = best_ensemble_auc

        for m in range(n_models):
            candidate_set = selected + [m]
            candidate_preds = geometric_mean_ensemble(oof_preds[candidate_set])
            auc = _mean_auc(candidate_preds, labels)
            if auc > best_auc:
                best_auc = auc
                best_candidate = m

        if best_candidate == -1:
            break  # No improvement possible

        selected.append(best_candidate)
        best_ensemble_auc = best_auc

    return selected


def compute_pairwise_correlation(oof_preds: np.ndarray) -> np.ndarray:
    """Compute mean Pearson correlation matrix across events.

    Parameters
    ----------
    oof_preds : np.ndarray  (n_models, n_samples, n_events)

    Returns
    -------
    np.ndarray  (n_models, n_models)
    """
    n_models, n_samples, n_events = oof_preds.shape
    corr_sum = np.zeros((n_models, n_models), dtype=np.float64)

    for e in range(n_events):
        X = oof_preds[:, :, e]      # (n_models, n_samples)
        # Normalise each model's predictions
        mu = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - mu) / std
        corr_sum += (X_norm @ X_norm.T) / n_samples

    return (corr_sum / n_events).astype(np.float32)


def rank_by_mean_auc(
    oof_preds: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return model indices sorted by mean OOF AUC (descending).

    Parameters
    ----------
    oof_preds : np.ndarray  (n_models, n_samples, n_events)
    labels    : np.ndarray  (n_samples, n_events)

    Returns
    -------
    np.ndarray  (n_models,) sorted indices
    """
    aucs = np.array([_mean_auc(oof_preds[i], labels) for i in range(oof_preds.shape[0])])
    return np.argsort(aucs)[::-1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_auc(preds: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-event ROC AUC, skipping degenerate events."""
    n_events = preds.shape[1] if preds.ndim == 2 else labels.shape[1]
    aucs: list[float] = []
    p = preds if preds.ndim == 2 else preds[:, :]
    for e in range(n_events):
        y_true = labels[:, e]
        if len(np.unique(y_true)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true, p[:, e])))
    return float(np.mean(aucs)) if aucs else 0.5
