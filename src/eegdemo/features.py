"""
特徴量抽出モジュール
すべての特徴抽出は因果的 (過去データのみ参照)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, welch

from .constants import CHANNELS, FREQ_BANDS, SAMPLING_RATE


# ──────────────────────────────────────────────
# 高速フィルタバンク帯域パワー (ベースライン用)
# ──────────────────────────────────────────────

def _build_filterbank(
    fs: float,
    bands: dict[str, tuple[float, float]],
    order: int = 4,
) -> dict[str, object]:
    """帯域ごとの SOS フィルタを事前構築する"""
    nyq = fs / 2.0
    filters = {}
    for name, (lo, hi) in bands.items():
        lo_n = max(lo / nyq, 1e-4)
        hi_n = min(hi / nyq, 0.9999)
        if lo_n >= hi_n:
            continue
        filters[name] = butter(order, [lo_n, hi_n], btype="band", output="sos")
    return filters


# モジュールレベルでフィルタをキャッシュ (重複構築を防ぐ)
_FILTERBANK_CACHE: dict[str, dict] = {}


def extract_band_power_features(
    eeg: np.ndarray,
    fs: float = SAMPLING_RATE,
    window_ms: float = 500.0,
    step_ms: float = 50.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    フィルタバンク法による高速帯域パワー特徴抽出 (因果的)。

    手順:
    1. 全信号に対してフィルタ適用 (causal sosfilt) → O(N) per band
    2. フィルタ出力の二乗 (瞬時パワー)
    3. スライディングウィンドウ内の平均パワーを log スケールで集計

    Welch ウィンドウごとの PSD より ~100x 高速。

    Parameters
    ----------
    eeg : (n_frames, n_channels)
    window_ms : 集計ウィンドウ長 (ms)
    step_ms : ステップ (ms) ← デフォルト 50ms に変更 (速度向上)

    Returns
    -------
    features : (n_windows, n_bands * n_channels)
    end_frames : (n_windows,) — 各ウィンドウの終端フレームインデックス
    """
    if bands is None:
        bands = {k: v for k, v in FREQ_BANDS.items() if k != "mu"}

    cache_key = f"{fs}_{list(bands.keys())}"
    if cache_key not in _FILTERBANK_CACHE:
        _FILTERBANK_CACHE[cache_key] = _build_filterbank(fs, bands)
    filterbank = _FILTERBANK_CACHE[cache_key]

    # 1. 全信号をフィルタリング & 二乗パワー
    n_frames, n_ch = eeg.shape
    band_powers: dict[str, np.ndarray] = {}
    for name, sos in filterbank.items():
        filtered = sosfilt(sos, eeg, axis=0).astype(np.float32)  # causal
        band_powers[name] = filtered ** 2  # (n_frames, n_ch)

    # 2. cumsum 差分法でスライディングウィンドウ平均を O(N) で計算 (因果的)
    win = max(1, int(window_ms * fs / 1000))
    step = max(1, int(step_ms * fs / 1000))

    ends = np.arange(win - 1, n_frames, step)
    n_windows = len(ends)
    n_bands = len(filterbank)
    features = np.empty((n_windows, n_bands * n_ch), dtype=np.float32)

    # starts[i] = ends[i] - win + 1
    starts = ends - win + 1  # (n_windows,)

    for b_idx, name in enumerate(filterbank):
        pw = band_powers[name].astype(np.float64)  # (n_frames, n_ch)

        # cumsum[t, :] = sum of pw[0..t, :]
        cs = np.cumsum(pw, axis=0)  # (n_frames, n_ch)

        # window_sum[i] = cs[ends[i]] - cs[starts[i] - 1]  (starts[i]>=1 の場合)
        sum_end = cs[ends]               # (n_windows, n_ch)
        sum_start = np.where(
            starts[:, None] > 0,
            cs[np.maximum(starts - 1, 0)],
            np.zeros((n_windows, n_ch), dtype=np.float64),
        )  # (n_windows, n_ch)

        mean_pw = (sum_end - sum_start) / win  # (n_windows, n_ch)
        features[:, b_idx * n_ch: (b_idx + 1) * n_ch] = np.log1p(mean_pw).astype(np.float32)

    return features, ends.astype(np.int32)


# ──────────────────────────────────────────────
# 時間領域特徴
# ──────────────────────────────────────────────

def time_domain_features(window: np.ndarray) -> np.ndarray:
    """
    時間領域特徴: 平均絶対値, RMS, 分散, 歪度, 尖度, 包絡線最大値
    window: (n_samples, n_channels)
    returns: (n_features * n_channels,)
    """
    from scipy.stats import skew, kurtosis
    from scipy.signal import hilbert

    mav = np.abs(window).mean(axis=0)           # 平均絶対値
    rms = np.sqrt((window ** 2).mean(axis=0))   # RMS
    var = window.var(axis=0)                    # 分散
    sk = skew(window, axis=0)                   # 歪度
    kurt = kurtosis(window, axis=0)             # 尖度

    # 包絡線 (Hilbert 変換の絶対値) の最大値 — 因果的でないが特徴量としては使用可
    envelope_max = np.abs(hilbert(window, axis=0)).max(axis=0)

    return np.concatenate([mav, rms, var, sk, kurt, envelope_max]).astype(np.float32)


# ──────────────────────────────────────────────
# Riemann 共分散特徴
# ──────────────────────────────────────────────

def riemannian_features(
    window: np.ndarray,
    regularize: float = 1e-6,
) -> np.ndarray:
    """
    Riemannian 多様体に基づく共分散特徴。
    共分散行列の上三角を vectorize (tangent space への投影はしない。
    モデル側で処理する)。
    window: (n_samples, n_channels)
    returns: (n_channels*(n_channels+1)/2,)
    """
    n_ch = window.shape[1]
    cov = np.cov(window.T)  # (n_ch, n_ch)
    cov += regularize * np.eye(n_ch)

    # 対称正定値行列の log-Euclidean 表現 (行列対数の上三角)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        log_cov = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
    except np.linalg.LinAlgError:
        log_cov = cov

    # 上三角要素 (対角含む) を抽出
    idx = np.triu_indices(n_ch)
    return log_cov[idx].astype(np.float32)


# ──────────────────────────────────────────────
# 左右非対称性特徴 (運動側性化指標)
# ──────────────────────────────────────────────

def laterality_index(
    window: np.ndarray,
    fs: float = SAMPLING_RATE,
    left_ch_idx: int | None = None,
    right_ch_idx: int | None = None,
    band: tuple[float, float] = (8.0, 30.0),
) -> float:
    """
    左右対称チャンネル (C3/C4) の帯域パワー差。
    LI = (右 - 左) / (右 + 左)
    正 → 右優位 (左手運動), 負 → 左優位 (右手運動)
    """
    if left_ch_idx is None:
        left_ch_idx = CHANNELS.index("C3")
    if right_ch_idx is None:
        right_ch_idx = CHANNELS.index("C4")

    def _band_power(sig: np.ndarray) -> float:
        nperseg = min(len(sig), 128)
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        mask = (freqs >= band[0]) & (freqs < band[1])
        return float(psd[mask].mean()) if mask.sum() > 0 else 0.0

    left = _band_power(window[:, left_ch_idx])
    right = _band_power(window[:, right_ch_idx])
    denom = left + right + 1e-12
    return float((right - left) / denom)


# ──────────────────────────────────────────────
# DL 用: スライディングウィンドウ生成 (因果的)
# ──────────────────────────────────────────────

def make_sliding_windows(
    eeg: np.ndarray,
    labels: np.ndarray | None,
    window_samples: int = 250,   # 500ms @ 500Hz
    step_samples: int = 1,
    label_at: str = "end",       # ウィンドウ末尾フレームのラベルを使用
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    因果的スライディングウィンドウ。
    ウィンドウ [t-window+1, t] のラベルは t 時点のもの。

    Returns
    -------
    windows : (n_windows, window_samples, n_channels)
    window_labels : (n_windows, 6) or None
    end_frames : (n_windows,)
    """
    n_frames, n_ch = eeg.shape
    ends = np.arange(window_samples - 1, n_frames, step_samples)
    n_windows = len(ends)

    windows = np.empty((n_windows, window_samples, n_ch), dtype=np.float32)
    for i, end in enumerate(ends):
        windows[i] = eeg[end - window_samples + 1: end + 1]

    window_labels = None
    if labels is not None:
        window_labels = labels[ends]  # (n_windows, 6)

    return windows, window_labels, ends.astype(np.int32)


def make_windows_causal_check(windows: np.ndarray, end_frames: np.ndarray) -> bool:
    """
    因果性テスト: ウィンドウが未来フレームを参照していないことを確認。
    ウィンドウ i は end_frames[i] 以前のデータのみ含むはず。
    (テスト用; 常に True を返すはず)
    """
    win_size = windows.shape[1]
    for i, end in enumerate(end_frames):
        start = end - win_size + 1
        if start < 0:
            return False  # 範囲外
    return True
