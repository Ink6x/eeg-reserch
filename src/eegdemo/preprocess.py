"""
前処理モジュール
【重要】すべてのフィルタは因果的 (scipy.signal.sosfilt / lfilter のみ使用)
filtfilt / zero-phase フィルタは未来情報を参照するため使用禁止
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch


# ──────────────────────────────────────────────
# 因果フィルタ
# ──────────────────────────────────────────────

def causal_bandpass(
    eeg: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """
    因果的バンドパスフィルタ。
    未来データを参照しないため sosfilt (前向きのみ) を使用。
    """
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)


def causal_highpass(
    eeg: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """因果的ハイパスフィルタ"""
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)


def causal_lowpass(
    eeg: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """因果的ローパスフィルタ"""
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)


def causal_notch(
    eeg: np.ndarray,
    fs: float,
    notch_hz: float,
    q: float = 30.0,
) -> np.ndarray:
    """因果的ノッチフィルタ (線路ノイズ除去)"""
    b, a = iirnotch(notch_hz, q, fs)
    from scipy.signal import lfilter
    return lfilter(b, a, eeg, axis=0).astype(np.float32)


# ──────────────────────────────────────────────
# リファレンス
# ──────────────────────────────────────────────

def apply_car(eeg: np.ndarray) -> np.ndarray:
    """
    Common Average Re-reference (CAR).
    全チャンネルの平均を各フレームから引く。
    CAR 自体は空間的操作なので因果制約に違反しない。
    """
    return (eeg - eeg.mean(axis=1, keepdims=True)).astype(np.float32)


# ──────────────────────────────────────────────
# 因果的正規化 (データリーク防止)
# ──────────────────────────────────────────────

def causal_running_zscore(
    eeg: np.ndarray,
    fs: float,
    warmup_sec: float = 5.0,
) -> np.ndarray:
    """
    Expanding window z-score (完全因果的, numpy ベクトル化版)。
    フレーム t の正規化には 0..t-1 のサンプルの統計量のみ使用。

    実装:
    - cumsum を使って expanding mean/variance を O(N) で計算 (因果的)
    - フレーム t の正規化に使う統計量は cumsum[t-1] から算出 (未来参照なし)
    - warmup_sec 以前は warmup 区間の統計量で代用 (std が不安定なため)
    """
    n, c = eeg.shape
    eeg64 = eeg.astype(np.float64)
    warmup = max(2, int(warmup_sec * fs))

    # expanding sum と sum-of-squares (インデックスをずらして causal にする)
    cum_sum = np.cumsum(eeg64, axis=0)          # cumsum[t] = sum(0..t)
    cum_sq = np.cumsum(eeg64 ** 2, axis=0)      # cumsum of x^2

    # フレーム t の統計量 = 0..t-1 のデータ (causal: t フレーム分の累積を使う)
    # counts[t] = t (t=0 → 0, t=1 → 1, ...)
    counts = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)  # (n, 1)

    # フレーム t に対して sum(0..t-1) = cumsum[t-1] を使う (1フレームずらす)
    sum_prev = np.vstack([np.zeros((1, c), dtype=np.float64), cum_sum[:-1]])   # (n, c)
    sq_prev = np.vstack([np.zeros((1, c), dtype=np.float64), cum_sq[:-1]])     # (n, c)
    cnt_prev = np.maximum(counts - 1, 1)  # フレーム t に使うサンプル数

    mean_prev = sum_prev / cnt_prev
    var_prev = np.maximum(sq_prev / cnt_prev - mean_prev ** 2, 1e-16)
    std_prev = np.sqrt(var_prev) + 1e-8

    # フレーム 0 は統計量が 0 件 → 0 で埋める
    # フレーム 1 以降は expanding stats を使用 (完全因果的)
    out = (eeg64 - mean_prev) / std_prev
    out[0] = 0.0  # フレーム 0 の統計量は存在しないため

    return out.astype(np.float32)


# ──────────────────────────────────────────────
# パイプライン
# ──────────────────────────────────────────────

def preprocess(
    eeg: np.ndarray,
    fs: float = 500.0,
    highpass_hz: float = 0.5,
    lowpass_hz: float = 45.0,
    notch_hz: float | None = None,
    reference: str = "CAR",
    normalize: bool = True,
    warmup_sec: float = 5.0,
) -> np.ndarray:
    """
    EEG 前処理パイプライン (全処理が因果的)。
    入力: eeg (n_frames, n_channels), float
    出力: 前処理済み eeg, float32
    """
    eeg = eeg.astype(np.float64)

    # 1. ハイパスフィルタ (ドリフト除去, RP を削らない 0.5 Hz)
    eeg = causal_highpass(eeg, fs, highpass_hz)

    # 2. ローパスフィルタ (高周波ノイズ除去)
    eeg = causal_lowpass(eeg, fs, lowpass_hz)

    # 3. ノッチフィルタ (必要な場合のみ)
    if notch_hz is not None:
        eeg = causal_notch(eeg, fs, notch_hz)

    # 4. リファレンス
    if reference == "CAR":
        eeg = apply_car(eeg)

    # 5. 因果的 z-score 正規化
    if normalize:
        eeg = causal_running_zscore(eeg, fs, warmup_sec)

    return eeg.astype(np.float32)


def detect_line_noise(eeg: np.ndarray, fs: float = 500.0) -> dict[str, float]:
    """
    FFT で 50/60 Hz のパワーを確認し、ノッチフィルタの必要性を判断する。
    返り値: {"50hz_snr": ..., "60hz_snr": ...} (単位: dB)
    """
    from scipy.fft import rfft, rfftfreq
    spectrum = np.abs(rfft(eeg, axis=0)) ** 2  # (freq_bins, channels)
    freqs = rfftfreq(len(eeg), 1.0 / fs)

    def _snr(target_hz: float, bandwidth: float = 2.0) -> float:
        signal_mask = np.abs(freqs - target_hz) < bandwidth / 2
        noise_mask = (np.abs(freqs - target_hz) < bandwidth) & ~signal_mask
        if noise_mask.sum() == 0:
            return 0.0
        signal_power = spectrum[signal_mask].mean()
        noise_power = spectrum[noise_mask].mean()
        return float(10 * np.log10(signal_power / (noise_power + 1e-12)))

    return {
        "50hz_snr_db": _snr(50.0),
        "60hz_snr_db": _snr(60.0),
    }
