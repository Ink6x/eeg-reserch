"""Causal Riemannian tangent-space feature extractor.

Reference: Barachant et al. (2013) "Classification of covariance matrices
using a Riemannian-based kernel for BCI applications", Neurocomputing 112,
pp. 172-178. DOI: 10.1016/j.neucom.2012.12.039.

Implementation
--------------
We use the ``pyriemann`` library (wrapping geomstats) for the SPD manifold
operations. If pyriemann is unavailable a pure-scipy approximation is used
(Euclidean tangent via matrix logarithm), which is less accurate but
dependency-free.

Causality note: Windows are fed in already extracted causally. Computing the
covariance over a window's content introduces no future information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg99.utils.constants import N_CHANNELS


@dataclass(frozen=True)
class RiemannianConfig:
    """Riemannian tangent-space feature config."""

    estimator: str = "lwf"      # Ledoit-Wolf shrinkage covariance
    tangent_metric: str = "riemann"
    regularization: float = 1e-6


class RiemannianExtractor:
    """Extract tangent-space features from EEG covariance matrices.

    Fit
    ---
    Compute the Riemannian mean of training covariance matrices. This
    mean is the reference point for tangent space projection.

    Transform
    ---------
    For each window:
      1. Estimate regularised covariance: C = X X^T / T + reg * I
      2. Project to tangent space at training mean
      3. Vectorise upper triangle → feature vector

    Output shape: ``(n_windows, n_ch * (n_ch + 1) // 2)``.
    For 32 channels: 32 * 33 // 2 = 528 features per window.

    Parameters
    ----------
    config : RiemannianConfig
    """

    def __init__(self, config: RiemannianConfig | None = None) -> None:
        self.config = config or RiemannianConfig()
        self._mean_cov: np.ndarray | None = None
        self._fitted = False
        self._use_pyriemann = _check_pyriemann()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> RiemannianExtractor:
        """Estimate the Riemannian mean of training covariance matrices.

        Parameters
        ----------
        X : np.ndarray
            ``(n_windows, n_channels, window_len)``

        Returns
        -------
        RiemannianExtractor
            Self.
        """
        covs = self._compute_covs(X)          # (n_win, C, C)

        if self._use_pyriemann:
            from pyriemann.utils.mean import gmean
            self._mean_cov = gmean(covs)
        else:
            # Euclidean mean (less accurate but no extra deps)
            self._mean_cov = covs.mean(axis=0)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project windows to tangent-space feature vectors.

        Parameters
        ----------
        X : np.ndarray
            ``(n_windows, n_channels, window_len)``

        Returns
        -------
        np.ndarray
            ``(n_windows, n_ch*(n_ch+1)//2)``, float32.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        covs = self._compute_covs(X)          # (n_win, C, C)
        n_win = covs.shape[0]
        n_ch = covs.shape[1]
        feat_dim = n_ch * (n_ch + 1) // 2

        features = np.empty((n_win, feat_dim), dtype=np.float32)

        if self._use_pyriemann:
            from pyriemann.tangentspace import TangentSpace
            ts = TangentSpace(metric=self.config.tangent_metric)
            ts.fit(np.array([self._mean_cov]))  # set reference
            # pyriemann TangentSpace.transform expects (n, C, C)
            tang = ts.transform(covs)           # (n_win, feat_dim)
            features[:] = tang.astype(np.float32)
        else:
            for i in range(n_win):
                features[i] = _tangent_vec_scipy(covs[i], self._mean_cov)

        return features

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_dim(self) -> int:
        return N_CHANNELS * (N_CHANNELS + 1) // 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_covs(self, X: np.ndarray) -> np.ndarray:
        """Compute regularised sample covariances for all windows.

        Parameters
        ----------
        X : np.ndarray
            ``(n_windows, n_channels, T)``

        Returns
        -------
        np.ndarray
            ``(n_windows, n_channels, n_channels)``, float64.
        """
        reg = self.config.regularization
        n_win, n_ch, T = X.shape
        covs = np.empty((n_win, n_ch, n_ch), dtype=np.float64)
        for i in range(n_win):
            x = X[i].astype(np.float64)
            C = x @ x.T / T
            C += reg * np.eye(n_ch)
            covs[i] = C
        return covs


# ---------------------------------------------------------------------------
# Scipy fallback (no pyriemann)
# ---------------------------------------------------------------------------

def _tangent_vec_scipy(C: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """Approximate tangent vector via matrix logarithm (Euclidean approx).

    Returns the upper-triangle of ``logm(C_ref^{-1/2} C C_ref^{-1/2})``,
    which is the exact Riemannian log map when C_ref is at the identity.
    """
    from scipy.linalg import inv, logm, sqrtm

    sqrt_ref = sqrtm(C_ref)
    inv_sqrt_ref = inv(sqrt_ref)
    S = inv_sqrt_ref @ C @ inv_sqrt_ref
    L = logm(S).real          # matrix log
    idx = np.triu_indices(L.shape[0])
    return L[idx].astype(np.float32)


def _check_pyriemann() -> bool:
    try:
        import pyriemann  # noqa: F401
        return True
    except ImportError:
        return False
