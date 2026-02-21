"""
EEG augmentation pipeline for v4 training.

すべての拡張は以下を保証する:
  - 因果性: window 内のデータが未来フレームを参照しない
  - ラベル整合性: ±150ms のラベル tolerance を超えない
  - 物理的妥当性: EEG の極性・周波数構造を破壊しない

訓練時のみ適用。Validation/test には適用しない。

設計の根拠は docs/technical_notes/v4_augmentation_design.md を参照。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


# ──────────────────────────────────────────────
# Channel-level augmentations
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class _Config:
    """共通の augmentation 設定基底（不変）"""
    p: float = 0.5


class ChannelMask:
    """
    1〜max_channels 個のランダムチャンネルを 0 化する。

    EEG-DL での効果: spatial filter 学習の汎化、 アーチファクト耐性向上
    （Mohsenvand et al. 2020, He et al. 2021）

    Args:
        max_channels: 同時にマスクする最大チャンネル数 (default: 3)
        p: 適用確率 (default: 0.5)
    """

    def __init__(self, max_channels: int = 3, p: float = 0.5) -> None:
        if max_channels < 1:
            raise ValueError(f"max_channels must be >= 1, got {max_channels}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.max_channels = max_channels
        self.p = p

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return x, y
        n_ch = x.shape[0]
        k = random.randint(1, min(self.max_channels, n_ch))
        idx = torch.randperm(n_ch)[:k]
        x = x.clone()
        x[idx, :] = 0.0
        return x, y


class AmplitudeScale:
    """
    各チャンネル独立にランダム振幅スケーリング。

    EEG-DL での効果: 個人差・電極インピーダンス変動への耐性
    （He et al. 2021）

    Args:
        scale_range: (min, max) スケーリング係数。 default (0.9, 1.1)
        p: 適用確率
    """

    def __init__(
        self, scale_range: tuple[float, float] = (0.9, 1.1), p: float = 0.5
    ) -> None:
        lo, hi = scale_range
        if lo <= 0 or hi <= 0:
            raise ValueError(f"scale_range must be positive, got {scale_range}")
        if lo > hi:
            raise ValueError(f"scale_range[0] > scale_range[1]: {scale_range}")
        self.lo = lo
        self.hi = hi
        self.p = p

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return x, y
        n_ch = x.shape[0]
        scale = torch.empty(n_ch, 1, dtype=x.dtype, device=x.device).uniform_(
            self.lo, self.hi
        )
        return x * scale, y


# ──────────────────────────────────────────────
# Time-level augmentations
# ──────────────────────────────────────────────


class GaussianNoise:
    """
    全チャンネルに小さなガウシアンノイズを加算。

    EEG-DL での効果: 過学習抑制、SNR robustness（Lashgari et al. 2020）

    Args:
        std: ノイズ標準偏差。 z-score 後の信号に対し 0.05 (5%) が安全範囲
        p: 適用確率
    """

    def __init__(self, std: float = 0.05, p: float = 0.3) -> None:
        if std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        self.std = std
        self.p = p

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return x, y
        return x + torch.randn_like(x) * self.std, y


# ──────────────────────────────────────────────
# Composition
# ──────────────────────────────────────────────


class Compose:
    """
    複数の augmentation を順に適用する。

    Args:
        transforms: 適用する augmentation のシーケンス
    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y

    def __len__(self) -> int:
        return len(self.transforms)


# ──────────────────────────────────────────────
# Batch-level augmentations
# ──────────────────────────────────────────────


def mixup_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    バッチ単位 mixup (Zhang et al. 2018).

    入力 X, y を λ 重みで他サンプルと線形補間する。
      X' = λ X + (1-λ) X[perm]
      y' = λ y + (1-λ) y[perm]
    λ ~ Beta(alpha, alpha)

    α=0 の場合は無効化（X, y そのまま返す）。

    Args:
        X: (batch, n_ch, T) 入力
        y: (batch, n_classes) ラベル (multi-label のため float)
        alpha: Beta 分布のパラメータ (default 0.2)
        p: 適用確率 (default 1.0)

    Returns:
        Mixed X, y
    """
    if alpha <= 0 or random.random() > p:
        return X, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    X_mixed = lam * X + (1.0 - lam) * X[idx]
    y_mixed = lam * y + (1.0 - lam) * y[idx]
    return X_mixed, y_mixed


# ──────────────────────────────────────────────
# Dataset with TemporalJitter integration
# ──────────────────────────────────────────────


class JitteredEEGWindowDataset(torch.utils.data.Dataset):
    """
    Temporal jitter 統合 EEG ウィンドウデータセット。

    Window 末端時刻をランダムに ±max_jitter サンプルシフトする。
    シフト後の時刻でウィンドウとラベルの両方を取得することで、
    因果性とラベル整合性 (±150ms tolerance 内) を保つ。

    Args:
        eeg: (n_frames, n_ch) 前処理済み EEG
        labels: (n_frames, n_classes) フレーム単位ラベル
        window_samples: ウィンドウ長 (default 250 = 500ms @ 500Hz)
        step_samples: スライディングステップ (default 10)
        positive_oversample_ratio: 正例オーバーサンプリング倍率 (default 3.0)
        max_jitter: 最大時間ジッタ (samples, default 0 で無効)
        post_augment: ウィンドウ抽出後に適用する augmentation の Compose
    """

    def __init__(
        self,
        eeg: np.ndarray,
        labels: np.ndarray,
        window_samples: int = 250,
        step_samples: int = 10,
        positive_oversample_ratio: float = 3.0,
        max_jitter: int = 0,
        post_augment: Compose | None = None,
    ) -> None:
        if max_jitter < 0:
            raise ValueError(f"max_jitter must be non-negative, got {max_jitter}")
        if window_samples < 1:
            raise ValueError(f"window_samples must be >= 1, got {window_samples}")

        self.eeg = torch.from_numpy(eeg.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.win = window_samples
        self.max_jitter = max_jitter
        self.post_augment = post_augment
        self._n_frames = len(eeg)

        # ジッタによる範囲外参照を防ぐマージン
        margin = max(0, max_jitter)
        all_ends = np.arange(
            window_samples - 1 + margin,
            self._n_frames - margin,
            step_samples,
            dtype=np.int32,
        )
        if len(all_ends) == 0:
            raise ValueError(
                f"No valid windows: n_frames={self._n_frames}, "
                f"window={window_samples}, jitter={max_jitter}"
            )

        # 正例・陰性例の分離とオーバーサンプリング
        is_pos = labels[all_ends].any(axis=1)
        pos_ends = all_ends[is_pos]
        neg_ends = all_ends[~is_pos]
        if positive_oversample_ratio > 1 and len(pos_ends) > 0:
            n_repeat = int(positive_oversample_ratio)
            pos_ends = np.tile(pos_ends, n_repeat)
        self.ends = np.concatenate([neg_ends, pos_ends]).astype(np.int32)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.ends[idx])
        if self.max_jitter > 0:
            jitter = random.randint(-self.max_jitter, self.max_jitter)
            # window が EEG 範囲を逸脱しないようクリップ
            end = max(self.win - 1, min(self._n_frames - 1, end + jitter))
        # window: [end - win + 1, end] — すべて end 以前 (因果的)
        x = self.eeg[end - self.win + 1 : end + 1].T.contiguous()
        y = self.labels[end]
        if self.post_augment is not None:
            x, y = self.post_augment(x, y)
        return x, y


# ──────────────────────────────────────────────
# Convenience: standard v4 pipeline
# ──────────────────────────────────────────────


def build_v4_train_augment(
    channel_mask_p: float = 0.5,
    amp_scale_p: float = 0.5,
    noise_p: float = 0.3,
) -> Compose:
    """
    v4 標準訓練用 augmentation パイプライン。

    TemporalJitter は Dataset 内で適用されるため含まれない。
    Mixup は batch レベルで適用するため含まれない。

    Returns:
        Compose: post-extraction augmentation
    """
    return Compose(
        [
            ChannelMask(max_channels=3, p=channel_mask_p),
            AmplitudeScale(scale_range=(0.9, 1.1), p=amp_scale_p),
            GaussianNoise(std=0.05, p=noise_p),
        ]
    )
