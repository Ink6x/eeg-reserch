"""
因果制約テスト — 最も重要なテスト。
未来データを参照するコードはここで必ず検出される。
"""
import numpy as np
import pytest

from src.eegdemo.features import make_sliding_windows, make_windows_causal_check
from src.eegdemo.preprocess import (
    causal_highpass,
    causal_lowpass,
    causal_running_zscore,
    preprocess,
)

FS = 500.0
N_FRAMES = 2000
N_CH = 32
RNG = np.random.default_rng(0)


@pytest.fixture
def dummy_eeg() -> np.ndarray:
    return RNG.standard_normal((N_FRAMES, N_CH)).astype(np.float32)


# ──────────────────────────────────────────────
# フィルタの因果性テスト
# ──────────────────────────────────────────────

def test_causal_highpass_no_future_leak(dummy_eeg: np.ndarray) -> None:
    """
    フレーム T 以降のデータを 0 に置換した後でフィルタしたとき、
    フレーム T 未満の出力が変化しないことを確認。
    (因果フィルタなら変化しない)
    """
    split = N_FRAMES // 2
    original = causal_highpass(dummy_eeg, FS, cutoff=0.5)

    # 後半を 0 に置換
    eeg_truncated = dummy_eeg.copy()
    eeg_truncated[split:] = 0.0
    filtered_truncated = causal_highpass(eeg_truncated, FS, cutoff=0.5)

    # 前半は同じはず
    np.testing.assert_allclose(
        original[:split],
        filtered_truncated[:split],
        rtol=1e-5,
        err_msg="causal_highpass が未来データを参照しています (filtfilt 使用の疑い)",
    )


def test_causal_lowpass_no_future_leak(dummy_eeg: np.ndarray) -> None:
    split = N_FRAMES // 2
    original = causal_lowpass(dummy_eeg, FS, cutoff=45.0)

    eeg_truncated = dummy_eeg.copy()
    eeg_truncated[split:] = 0.0
    filtered_truncated = causal_lowpass(eeg_truncated, FS, cutoff=45.0)

    np.testing.assert_allclose(
        original[:split],
        filtered_truncated[:split],
        rtol=1e-5,
        err_msg="causal_lowpass が未来データを参照しています (filtfilt 使用の疑い)",
    )


def test_running_zscore_no_future_leak(dummy_eeg: np.ndarray) -> None:
    """
    running z-score が expanding window (過去のみ) を使っていることを確認。
    フレーム T 以降を置換しても前半の正規化結果が変化しないはず。
    """
    split = N_FRAMES // 2
    original = causal_running_zscore(dummy_eeg, FS, warmup_sec=1.0)

    eeg_truncated = dummy_eeg.copy()
    eeg_truncated[split:] = 9999.0  # 極端な値で未来参照の漏れを検出しやすくする
    normalized_truncated = causal_running_zscore(eeg_truncated, FS, warmup_sec=1.0)

    np.testing.assert_allclose(
        original[:split],
        normalized_truncated[:split],
        rtol=1e-4,
        err_msg="causal_running_zscore が未来データを参照しています",
    )


# ──────────────────────────────────────────────
# スライディングウィンドウの因果性テスト
# ──────────────────────────────────────────────

def test_sliding_windows_are_causal(dummy_eeg: np.ndarray) -> None:
    """
    ウィンドウの内容がすべて end_frame 以前のフレームで構成されることを確認。
    """
    labels = np.zeros((N_FRAMES, 6), dtype=np.int8)
    windows, _, end_frames = make_sliding_windows(
        dummy_eeg, labels, window_samples=250, step_samples=10
    )
    assert make_windows_causal_check(
        windows, end_frames
    ), "スライディングウィンドウが未来フレームを含んでいます"


def test_sliding_window_content_matches_eeg(dummy_eeg: np.ndarray) -> None:
    """
    ウィンドウの内容が元の EEG と一致することを確認。
    """
    labels = np.zeros((N_FRAMES, 6), dtype=np.int8)
    window_samples = 100
    windows, _, end_frames = make_sliding_windows(
        dummy_eeg, labels, window_samples=window_samples, step_samples=50
    )

    for i, end in enumerate(end_frames[:5]):  # 最初の5件を検証
        expected = dummy_eeg[end - window_samples + 1: end + 1]
        np.testing.assert_array_equal(
            windows[i], expected,
            err_msg=f"ウィンドウ {i} の内容が EEG と一致しません",
        )


# ──────────────────────────────────────────────
# 前処理パイプライン全体のテスト
# ──────────────────────────────────────────────

def test_preprocess_pipeline_no_future_leak(dummy_eeg: np.ndarray) -> None:
    """前処理パイプライン全体で未来漏れがないことを確認"""
    split = N_FRAMES // 2
    proc_full = preprocess(dummy_eeg, FS)

    eeg_half = dummy_eeg.copy()
    eeg_half[split:] = 0.0
    proc_half = preprocess(eeg_half, FS)

    np.testing.assert_allclose(
        proc_full[:split],
        proc_half[:split],
        rtol=1e-4,
        err_msg="preprocess パイプラインが未来データを参照しています",
    )


def test_preprocess_output_shape(dummy_eeg: np.ndarray) -> None:
    """前処理後の shape が変わらないことを確認"""
    out = preprocess(dummy_eeg, FS)
    assert out.shape == dummy_eeg.shape, f"形状変化: {dummy_eeg.shape} → {out.shape}"


def test_preprocess_output_dtype(dummy_eeg: np.ndarray) -> None:
    """前処理後が float32 であることを確認"""
    out = preprocess(dummy_eeg, FS)
    assert out.dtype == np.float32, f"dtype: {out.dtype}"
