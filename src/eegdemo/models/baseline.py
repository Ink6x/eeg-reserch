"""
ベースライン分類器: 帯域パワー + LogisticRegression / LightGBM
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..constants import SAMPLING_RATE
from ..features import extract_band_power_features


class BandPowerClassifier:
    """
    帯域パワー特徴 + LogisticRegression による multi-label 分類器。
    ベースラインとして使用。
    """

    def __init__(self, C: float = 1.0, window_ms: float = 500.0) -> None:
        self.window_ms = window_ms
        base = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(base, n_jobs=-1)),
        ])

    def fit(
        self, eeg: np.ndarray, labels: np.ndarray, fs: float = SAMPLING_RATE
    ) -> BandPowerClassifier:
        """eeg: (n_frames, n_ch), labels: (n_frames, 6)"""
        X, centers = extract_band_power_features(eeg, fs, self.window_ms, step_ms=4.0)
        y = labels[centers]
        self.model.fit(X, y)
        self._fs = fs
        return self

    def predict_proba(self, eeg: np.ndarray, fs: float | None = None) -> np.ndarray:
        """returns: (n_windows, 6) の確率値"""
        if fs is None:
            fs = self._fs
        X, _ = extract_band_power_features(eeg, fs, self.window_ms, step_ms=4.0)
        # MultiOutputClassifier.predict_proba は list of (n, 2) を返す
        proba_list = self.model.predict_proba(X)
        return np.stack([p[:, 1] for p in proba_list], axis=1)  # (n_windows, 6)

    def predict(self, eeg: np.ndarray, fs: float | None = None) -> np.ndarray:
        if fs is None:
            fs = self._fs
        X, _ = extract_band_power_features(eeg, fs, self.window_ms, step_ms=4.0)
        return self.model.predict(X)
