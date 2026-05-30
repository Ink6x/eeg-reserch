"""HierarchicalBayesianCalibration — original algorithm #5.

Per-subject Platt scaling with a hierarchical Gaussian prior shared across
subjects.  Each subject gets its own sigmoid parameters (a_i, b_i) that are
regularised toward the population mean (μ_a, μ_b) estimated from all
subjects simultaneously.

Model
-----
P(y=1 | z, i) = σ(a_i * z + b_i)           (per-subject Platt)

Priors (hierarchical):
  a_i ~ N(μ_a, σ_a²)    b_i ~ N(μ_b, σ_b²)
  μ_a ~ N(1, 1)         μ_b ~ N(0, 1)

Optimisation: MAP estimate via L-BFGS (no sampling needed).

Reference: Niculescu-Mizil & Caruana (2005) for Platt scaling.
Hierarchical formulation is our original contribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["HierarchicalBayesianCalibration", "CalibrationConfig"]


@dataclass(frozen=True)
class CalibrationConfig:
    max_iter: int = 200
    prior_sigma_a: float = 0.5   # prior std for per-subject scale
    prior_sigma_b: float = 0.5   # prior std for per-subject bias
    prior_mu_a: float = 1.0      # population prior mean for a
    prior_mu_b: float = 0.0      # population prior mean for b
    tol: float = 1e-6


class HierarchicalBayesianCalibration:
    """Hierarchical per-subject Platt scaling calibration.

    Fit one (a_i, b_i) pair per subject, regularised toward a shared
    population mean that is also learned from data.

    Parameters
    ----------
    cfg : CalibrationConfig
    n_subjects : int
    n_events : int
    """

    def __init__(
        self,
        n_subjects: int,
        n_events: int,
        cfg: CalibrationConfig | None = None,
    ) -> None:
        self.n_subjects = n_subjects
        self.n_events = n_events
        self.cfg = cfg or CalibrationConfig()

        # Per-event, per-subject parameters (a, b)
        # Shape: (n_events, n_subjects)
        self._a = np.ones((n_events, n_subjects), dtype=np.float64)
        self._b = np.zeros((n_events, n_subjects), dtype=np.float64)
        self._mu_a = np.ones(n_events, dtype=np.float64)
        self._mu_b = np.zeros(n_events, dtype=np.float64)
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
    ) -> "HierarchicalBayesianCalibration":
        """Fit calibration parameters.

        Parameters
        ----------
        logits      : (n_samples, n_events)  raw model outputs (pre-sigmoid)
        labels      : (n_samples, n_events)  binary {0, 1}
        subject_ids : (n_samples,)  int in [0, n_subjects)
        """
        from scipy.optimize import minimize

        for e in range(self.n_events):
            z = logits[:, e]
            y = labels[:, e]
            result = minimize(
                fun=self._neg_log_posterior,
                x0=self._pack_params(e),
                args=(z, y, subject_ids),
                method="L-BFGS-B",
                options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.tol},
            )
            self._unpack_params(e, result.x)

        self._fitted = True
        return self

    def predict_proba(
        self,
        logits: np.ndarray,
        subject_ids: np.ndarray,
    ) -> np.ndarray:
        """Apply calibration.

        Parameters
        ----------
        logits      : (n_samples, n_events)
        subject_ids : (n_samples,)

        Returns
        -------
        np.ndarray  (n_samples, n_events) calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        probs = np.empty_like(logits, dtype=np.float32)
        for e in range(self.n_events):
            a = self._a[e, subject_ids]    # (n_samples,)
            b = self._b[e, subject_ids]
            probs[:, e] = _sigmoid(a * logits[:, e] + b).astype(np.float32)

        return probs

    # ------------------------------------------------------------------
    # Internal MAP optimisation helpers
    # ------------------------------------------------------------------

    def _pack_params(self, event_idx: int) -> np.ndarray:
        """Flatten (mu_a, mu_b, a_0..a_S, b_0..b_S) into a 1-D array."""
        e = event_idx
        return np.concatenate([
            [self._mu_a[e], self._mu_b[e]],
            self._a[e],
            self._b[e],
        ])

    def _unpack_params(self, event_idx: int, params: np.ndarray) -> None:
        S = self.n_subjects
        e = event_idx
        self._mu_a[e] = params[0]
        self._mu_b[e] = params[1]
        self._a[e] = params[2: 2 + S]
        self._b[e] = params[2 + S: 2 + 2 * S]

    def _neg_log_posterior(
        self,
        params: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
    ) -> float:
        S = self.n_subjects
        mu_a, mu_b = params[0], params[1]
        a = params[2: 2 + S]
        b = params[2 + S: 2 + 2 * S]

        # Log-likelihood: binary cross-entropy
        a_i = a[subject_ids]
        b_i = b[subject_ids]
        logits_cal = a_i * z + b_i
        log_lik = -np.mean(
            y * _log_sigmoid(logits_cal) + (1 - y) * _log_sigmoid(-logits_cal)
        )

        cfg = self.cfg
        # Hierarchical prior: a_i ~ N(mu_a, sigma_a^2)
        prior_a = np.sum((a - mu_a) ** 2) / (2 * cfg.prior_sigma_a ** 2)
        prior_b = np.sum((b - mu_b) ** 2) / (2 * cfg.prior_sigma_b ** 2)
        # Hyper-prior on mu_a, mu_b
        prior_mu_a = (mu_a - cfg.prior_mu_a) ** 2 / 2
        prior_mu_b = (mu_b - cfg.prior_mu_b) ** 2 / 2

        return log_lik + (prior_a + prior_b + prior_mu_a + prior_mu_b) / len(y)


# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))
