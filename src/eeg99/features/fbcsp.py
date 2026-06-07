"""Filter-Bank Common Spatial Pattern (FBCSP) feature extractor.

Reference: Ang et al. (2008) "Filter Bank Common Spatial Pattern (FBCSP)
in Brain-Computer Interface", IEEE IJCNN, pp. 2390-2397.

Adaptation for eeg99
---------------------
* Causal windows are fed in (future-leak-free by construction in dataset.py).
* One-vs-rest CSP: one set of spatial filters per (event × band) pair.
* 6 events × 5 bands × n_components = feature vector per window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt

from eeg99.utils.constants import N_EVENTS, SFREQ

# ---------------------------------------------------------------------------
# Band definitions (mirrored from filterbank for standalone use)
# ---------------------------------------------------------------------------

_DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),   # delta
    (4.0, 8.0),   # theta
    (8.0, 13.0),  # alpha
    (13.0, 30.0), # beta
    (30.0, 45.0), # gamma
)


@dataclass(frozen=True)
class FBCSPConfig:
    """FBCSP hyperparameters."""

    n_components: int = 4       # CSP components per band (even; half from each tail)
    filter_order: int = 4
    sfreq: float = SFREQ
    bands: tuple[tuple[float, float], ...] = _DEFAULT_BANDS
    regularization: float = 1e-6  # ridge added to covariance for numerical stability
    log_variance: bool = True


class FBCSPExtractor:
    """Fit-then-transform FBCSP feature extractor.

    Fit
    ---
    Provide labelled windows ``X`` of shape ``(n_windows, n_channels, window_len)``
    and binary event labels ``y`` of shape ``(n_windows, n_events)``.
    CSP spatial filters are estimated per (event × band).

    Transform
    ---------
    Apply learned filters to new windows, return log-variance features.
    Output shape: ``(n_windows, n_events * n_bands * n_components)``.

    Parameters
    ----------
    config : FBCSPConfig
        Hyperparameters.
    """

    def __init__(self, config: FBCSPConfig | None = None) -> None:
        self.config = config or FBCSPConfig()
        # _filters[event_idx][band_idx] → (n_channels, n_components)
        self._filters: list[list[np.ndarray]] = []
        self._sos_list: list[np.ndarray] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> FBCSPExtractor:
        """Estimate CSP filters from labelled training windows.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_windows, n_channels, window_len)``.
        y : np.ndarray
            Shape ``(n_windows, n_events)``, binary {0, 1}.

        Returns
        -------
        FBCSPExtractor
            Self.
        """
        cfg = self.config
        self._build_bandpass_filters()
        n_events = y.shape[1]
        n_bands = len(cfg.bands)

        self._filters = []
        for e in range(n_events):
            band_filters: list[np.ndarray] = []
            for b in range(n_bands):
                X_filt = self._apply_band(X, b)  # (n_win, C, T)
                W = _fit_csp_binary(
                    X_filt, y[:, e],
                    n_components=cfg.n_components,
                    reg=cfg.regularization,
                )
                band_filters.append(W)
            self._filters.append(band_filters)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract FBCSP log-variance features.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_windows, n_channels, window_len)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_windows, n_events * n_bands * n_components)``, float32.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        cfg = self.config
        n_windows = X.shape[0]
        n_events = len(self._filters)
        n_bands = len(cfg.bands)
        n_comp = cfg.n_components
        out_dim = n_events * n_bands * n_comp

        features = np.empty((n_windows, out_dim), dtype=np.float32)
        col = 0
        for e in range(n_events):
            for b in range(n_bands):
                X_filt = self._apply_band(X, b)          # (n_win, C, T)
                W = self._filters[e][b]                   # (C, n_comp)
                projected = np.einsum("wct,ck->wkt", X_filt, W)  # (n_win, n_comp, T)
                if cfg.log_variance:
                    feats = np.log(np.var(projected, axis=2) + 1e-8)
                else:
                    feats = np.var(projected, axis=2)
                features[:, col : col + n_comp] = feats.astype(np.float32)
                col += n_comp

        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def feature_dim(self) -> int:
        cfg = self.config
        return N_EVENTS * len(cfg.bands) * cfg.n_components

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bandpass_filters(self) -> None:
        nyq = self.config.sfreq / 2.0
        self._sos_list = []
        for low, high in self.config.bands:
            sos = butter(
                self.config.filter_order,
                [low / nyq, high / nyq],
                btype="bandpass",
                output="sos",
            )
            self._sos_list.append(sos)

    def _apply_band(self, X: np.ndarray, band_idx: int) -> np.ndarray:
        """Bandpass-filter all windows for one band.

        Parameters
        ----------
        X : np.ndarray
            ``(n_windows, n_channels, window_len)``

        Returns
        -------
        np.ndarray
            Same shape, band-filtered.
        """
        sos = self._sos_list[band_idx]
        n_win, n_ch, T = X.shape
        out = np.empty_like(X)
        for w in range(n_win):
            for ch in range(n_ch):
                out[w, ch] = sosfilt(sos, X[w, ch])
        return out


# ---------------------------------------------------------------------------
# CSP math (generalized eigenvalue problem)
# ---------------------------------------------------------------------------

def _cov(X: np.ndarray, reg: float) -> np.ndarray:
    """Regularised sample covariance of ``X`` shaped ``(n_channels, T)``."""
    C = X @ X.T / X.shape[1]
    C += reg * np.eye(C.shape[0])
    return C


def _fit_csp_binary(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    reg: float = 1e-6,
) -> np.ndarray:
    """Fit CSP spatial filters for a single binary event.

    Parameters
    ----------
    X : np.ndarray
        ``(n_windows, n_channels, T)``
    y : np.ndarray
        ``(n_windows,)`` binary labels {0, 1}.
    n_components : int
        Number of spatial filters to return (even).

    Returns
    -------
    np.ndarray
        ``(n_channels, n_components)`` spatial filter matrix.
    """
    mask_pos = y > 0.5
    mask_neg = ~mask_pos

    # Class covariances as mean over windows
    if mask_pos.sum() == 0 or mask_neg.sum() == 0:
        # Degenerate: return identity (no discrimination possible)
        n_ch = X.shape[1]
        return np.eye(n_ch, n_components)

    C_pos = np.mean([_cov(X[i], reg) for i in np.where(mask_pos)[0]], axis=0)
    C_neg = np.mean([_cov(X[i], reg) for i in np.where(mask_neg)[0]], axis=0)

    # Generalized eigenvalue problem: C_pos W = λ (C_pos + C_neg) W
    try:
        _, W = eigh(C_pos, C_pos + C_neg)
    except np.linalg.LinAlgError:
        n_ch = X.shape[1]
        return np.eye(n_ch, n_components)

    # Select n_components/2 from each end (most discriminative)
    half = n_components // 2
    idx = np.concatenate([np.arange(half), np.arange(-half, 0)])
    return W[:, idx]
