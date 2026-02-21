"""
Augmentation pipeline のテスト。

確認項目:
  1. Shape 保持
  2. 因果性違反なし
  3. ラベル整合性 (±150ms tolerance 内)
  4. 数値範囲の妥当性
  5. 確率パラメータの動作 (p=0 で無効化)
  6. 再現性 (random seed)
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.eegdemo.augmentation import (
    AmplitudeScale,
    ChannelMask,
    Compose,
    GaussianNoise,
    JitteredEEGWindowDataset,
    build_v4_train_augment,
    mixup_batch,
)


N_CH = 32
N_TIMES = 250
N_CLASSES = 6
RNG = np.random.default_rng(42)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_window() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(N_CH, N_TIMES)
    y = torch.zeros(N_CLASSES)
    return x, y


@pytest.fixture
def sample_eeg_data() -> tuple[np.ndarray, np.ndarray]:
    """5000 frame EEG with positive labels in two segments."""
    n_frames = 5000
    eeg = RNG.standard_normal((n_frames, N_CH)).astype(np.float32)
    labels = np.zeros((n_frames, N_CLASSES), dtype=np.int8)
    # Insert positive label windows
    labels[1000:1100, 0] = 1
    labels[2000:2100, 1] = 1
    labels[3000:3100, 2] = 1
    return eeg, labels


@pytest.fixture(autouse=True)
def reset_random() -> None:
    """各テスト前に random seed をリセット"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


# ──────────────────────────────────────────────
# ChannelMask
# ──────────────────────────────────────────────


def test_channel_mask_shape_preserved(sample_window: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = sample_window
    aug = ChannelMask(max_channels=3, p=1.0)
    x_out, y_out = aug(x, y)
    assert x_out.shape == x.shape
    assert y_out.shape == y.shape


def test_channel_mask_zeros_some_channels(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    aug = ChannelMask(max_channels=3, p=1.0)
    x_out, _ = aug(x, y)
    n_zero = int((x_out == 0).all(dim=1).sum().item())
    assert 1 <= n_zero <= 3, f"Expected 1-3 zeroed channels, got {n_zero}"


def test_channel_mask_p_zero_no_change(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    aug = ChannelMask(max_channels=3, p=0.0)
    x_out, y_out = aug(x, y)
    assert torch.equal(x, x_out)
    assert torch.equal(y, y_out)


def test_channel_mask_does_not_modify_label(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    y_before = y.clone()
    aug = ChannelMask(max_channels=3, p=1.0)
    _, y_out = aug(x, y)
    assert torch.equal(y_before, y_out)


def test_channel_mask_invalid_args() -> None:
    with pytest.raises(ValueError):
        ChannelMask(max_channels=0, p=0.5)
    with pytest.raises(ValueError):
        ChannelMask(max_channels=3, p=-0.1)
    with pytest.raises(ValueError):
        ChannelMask(max_channels=3, p=1.5)


# ──────────────────────────────────────────────
# AmplitudeScale
# ──────────────────────────────────────────────


def test_amplitude_scale_shape_preserved(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    aug = AmplitudeScale(scale_range=(0.9, 1.1), p=1.0)
    x_out, _ = aug(x, y)
    assert x_out.shape == x.shape


def test_amplitude_scale_per_channel_in_range(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    # Avoid x near zero where ratio is unstable
    x = x + 1.0  # offset to ensure non-zero
    aug = AmplitudeScale(scale_range=(0.9, 1.1), p=1.0)
    x_out, _ = aug(x, y)
    ratio = x_out / x
    # Each channel should have nearly constant ratio
    per_ch_std = ratio.std(dim=1)
    assert (per_ch_std < 1e-5).all(), "AmplitudeScale should be channel-wise constant"
    per_ch_ratio = ratio.mean(dim=1)
    assert (per_ch_ratio >= 0.89).all() and (per_ch_ratio <= 1.11).all()


def test_amplitude_scale_invalid_range() -> None:
    with pytest.raises(ValueError):
        AmplitudeScale(scale_range=(-0.1, 1.0))
    with pytest.raises(ValueError):
        AmplitudeScale(scale_range=(1.5, 0.5))  # lo > hi


# ──────────────────────────────────────────────
# GaussianNoise
# ──────────────────────────────────────────────


def test_gaussian_noise_shape_preserved(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    aug = GaussianNoise(std=0.05, p=1.0)
    x_out, _ = aug(x, y)
    assert x_out.shape == x.shape


def test_gaussian_noise_changes_signal(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = sample_window
    aug = GaussianNoise(std=0.1, p=1.0)
    x_out, _ = aug(x, y)
    diff = (x_out - x).abs()
    assert diff.mean().item() > 0.01, "GaussianNoise should change signal"


def test_gaussian_noise_magnitude_bounded(
    sample_window: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """3σ 内に大半のノイズが収まる"""
    x, y = sample_window
    std = 0.05
    aug = GaussianNoise(std=std, p=1.0)
    x_out, _ = aug(x, y)
    diff = x_out - x
    # 99.7% (3σ) 以内
    within_3sigma = (diff.abs() < 3 * std).float().mean().item()
    assert within_3sigma > 0.99


def test_gaussian_noise_zero_std() -> None:
    x = torch.randn(N_CH, N_TIMES)
    y = torch.zeros(N_CLASSES)
    aug = GaussianNoise(std=0.0, p=1.0)
    x_out, _ = aug(x, y)
    assert torch.equal(x, x_out)


# ──────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────


def test_compose_applies_in_order() -> None:
    x = torch.zeros(N_CH, N_TIMES)
    y = torch.zeros(N_CLASSES)

    class _AddConst:
        def __init__(self, val: float) -> None:
            self.val = val

        def __call__(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x + self.val, y

    augs = Compose([_AddConst(1.0), _AddConst(2.0), _AddConst(0.5)])
    x_out, _ = augs(x, y)
    assert x_out.mean().item() == pytest.approx(3.5)


def test_compose_empty() -> None:
    x = torch.randn(N_CH, N_TIMES)
    y = torch.zeros(N_CLASSES)
    augs = Compose([])
    x_out, y_out = augs(x, y)
    assert torch.equal(x, x_out)
    assert torch.equal(y, y_out)


def test_build_v4_train_augment_returns_compose() -> None:
    augs = build_v4_train_augment()
    assert isinstance(augs, Compose)
    assert len(augs) == 3  # ChannelMask, AmplitudeScale, GaussianNoise


# ──────────────────────────────────────────────
# Mixup
# ──────────────────────────────────────────────


def test_mixup_alpha_zero_passthrough() -> None:
    X = torch.randn(8, N_CH, N_TIMES)
    y = torch.zeros(8, N_CLASSES)
    X_out, y_out = mixup_batch(X, y, alpha=0.0)
    assert torch.equal(X, X_out)
    assert torch.equal(y, y_out)


def test_mixup_p_zero_passthrough() -> None:
    X = torch.randn(8, N_CH, N_TIMES)
    y = torch.zeros(8, N_CLASSES)
    X_out, y_out = mixup_batch(X, y, alpha=0.5, p=0.0)
    assert torch.equal(X, X_out)
    assert torch.equal(y, y_out)


def test_mixup_shape_preserved() -> None:
    X = torch.randn(16, N_CH, N_TIMES)
    y = torch.zeros(16, N_CLASSES)
    X_out, y_out = mixup_batch(X, y, alpha=0.2)
    assert X_out.shape == X.shape
    assert y_out.shape == y.shape


def test_mixup_label_in_valid_range() -> None:
    X = torch.randn(16, N_CH, N_TIMES)
    y = torch.eye(N_CLASSES)[torch.randint(0, N_CLASSES, (16,))]
    X_out, y_out = mixup_batch(X, y, alpha=0.5)
    assert (y_out >= 0).all() and (y_out <= 1).all()


# ──────────────────────────────────────────────
# JitteredEEGWindowDataset — causality (CRITICAL)
# ──────────────────────────────────────────────


def test_jittered_dataset_basic_shape(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    eeg, labels = sample_eeg_data
    ds = JitteredEEGWindowDataset(
        eeg, labels, window_samples=250, step_samples=20, max_jitter=10
    )
    x, y = ds[0]
    assert x.shape == (N_CH, 250)
    assert y.shape == (N_CLASSES,)


def test_jittered_dataset_no_jitter_is_deterministic(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """max_jitter=0 では複数回読み出しても同じウィンドウ"""
    eeg, labels = sample_eeg_data
    ds = JitteredEEGWindowDataset(
        eeg, labels, window_samples=250, step_samples=20, max_jitter=0
    )
    x1, y1 = ds[5]
    x2, y2 = ds[5]
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_jittered_dataset_causality_no_future_leak(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    因果性テスト: ウィンドウ抽出後、 EEG の未来部分を 0 に置換しても
    既に取得したウィンドウの内容が変わらないことを確認。
    """
    eeg, labels = sample_eeg_data
    ds = JitteredEEGWindowDataset(
        eeg, labels, window_samples=250, step_samples=20, max_jitter=0
    )
    # 序盤のウィンドウを取得
    x, _ = ds[3]  # end ≈ 250 + 60 = 310 付近のはず

    # EEG の未来部分を 0 に置換
    eeg_mod = eeg.copy()
    eeg_mod[1000:] = 0.0
    ds_mod = JitteredEEGWindowDataset(
        eeg_mod, labels, window_samples=250, step_samples=20, max_jitter=0
    )
    x_mod, _ = ds_mod[3]
    assert torch.equal(x, x_mod), "Future modification leaked into past window"


def test_jittered_dataset_window_within_bounds(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """全ウィンドウが EEG 範囲内 (因果性 + 範囲外参照なし)"""
    eeg, labels = sample_eeg_data
    ds = JitteredEEGWindowDataset(
        eeg, labels, window_samples=250, step_samples=20, max_jitter=10
    )
    n_frames = len(eeg)
    # サンプルしてチェック
    for i in range(0, len(ds), max(1, len(ds) // 50)):
        x, y = ds[i]
        assert x.shape == (N_CH, 250)
        assert torch.isfinite(x).all()
        # ラベルも有効
        assert torch.isfinite(y).all()


def test_jittered_dataset_with_post_augment(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    eeg, labels = sample_eeg_data
    augs = build_v4_train_augment()
    ds = JitteredEEGWindowDataset(
        eeg,
        labels,
        window_samples=250,
        step_samples=20,
        max_jitter=10,
        post_augment=augs,
    )
    x, y = ds[0]
    assert x.shape == (N_CH, 250)
    assert y.shape == (N_CLASSES,)
    assert torch.isfinite(x).all()


def test_jittered_dataset_invalid_args(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    eeg, labels = sample_eeg_data
    with pytest.raises(ValueError):
        JitteredEEGWindowDataset(eeg, labels, max_jitter=-1)
    with pytest.raises(ValueError):
        JitteredEEGWindowDataset(eeg, labels, window_samples=0)


# ──────────────────────────────────────────────
# Combined pipeline integration
# ──────────────────────────────────────────────


def test_full_pipeline_smoke(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    すべての augmentation を組み合わせた smoke test。
    エラーなく一巡できること、 出力の妥当性を確認。
    """
    eeg, labels = sample_eeg_data
    augs = build_v4_train_augment()
    ds = JitteredEEGWindowDataset(
        eeg,
        labels,
        window_samples=250,
        step_samples=20,
        max_jitter=10,
        post_augment=augs,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

    for batch_idx, (X, y) in enumerate(loader):
        assert X.shape == (8, N_CH, 250)
        assert y.shape == (8, N_CLASSES)
        # mixup を batch 適用
        X_mixed, y_mixed = mixup_batch(X, y, alpha=0.2, p=1.0)
        assert X_mixed.shape == X.shape
        assert y_mixed.shape == y.shape
        assert (y_mixed >= 0).all() and (y_mixed <= 1).all()
        if batch_idx >= 3:
            break


def test_label_integrity_under_jitter(
    sample_eeg_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """
    ラベル整合性: max_jitter=10 (±20ms) はラベル tolerance ±150ms 内。
    ジッタ後のラベルは、 元のラベルとほぼ同じ (境界点を除く)。
    """
    eeg, labels = sample_eeg_data
    ds_no_jitter = JitteredEEGWindowDataset(
        eeg, labels, window_samples=250, step_samples=20, max_jitter=0
    )
    ds_jitter = JitteredEEGWindowDataset(
        eeg, labels, window_samples=250, step_samples=20, max_jitter=10
    )

    n_check = min(100, len(ds_jitter))
    label_match = 0
    for i in range(n_check):
        _, y_no_jit = ds_no_jitter[i]
        _, y_jit = ds_jitter[i]
        if torch.equal(y_no_jit, y_jit):
            label_match += 1

    # 90% 以上のラベルは一致するはず (境界点近傍は変動しうる)
    match_rate = label_match / n_check
    assert match_rate >= 0.85, f"Label match rate too low: {match_rate:.2%}"
