"""
CausalEEGConformer — 独自モデル
Causal Temporal Convolution + Electrode Attention + Riemann Covariance Pooling

設計方針:
- 全 temporal 処理は因果パディング (未来参照禁止)
- ElectrodeAttention: 10-20 電極座標を position encoding に使用し、
  空間的近接関係を attention の prior として組み込む
- RiemannCovPool: 共分散行列の log-Euclidean 特徴を連結 (小データで汎化)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import CHANNEL_POS_2D, CHANNELS


# ──────────────────────────────────────────────
# 因果的 Temporal Convolution Block
# ──────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """因果的拡張畳み込み (左パディングのみ)"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Temporal Convolution Block with residual connection"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        residual = self.proj(x)
        out = self.act(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        out = self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2)
        return self.act(out + residual)


# ──────────────────────────────────────────────
# Electrode Attention (空間的 Prior 付き)
# ──────────────────────────────────────────────

def _build_electrode_dist_matrix(channel_names: list[str]) -> torch.Tensor:
    """
    電極間のユークリッド距離行列を返す。
    10-20 座標を使用。存在しない電極は原点に置く。
    returns: (n_ch, n_ch) float32
    """
    pos = np.array([CHANNEL_POS_2D.get(ch, (0.0, 0.0)) for ch in channel_names], dtype=np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return torch.from_numpy(dist)


class ElectrodeAttention(nn.Module):
    """
    Multi-head self-attention over electrode (channel) 次元。
    事前計算した電極間距離を attention bias に加算し、
    空間的近接を自然に学習する。
    入力: (batch, n_ch, d_model)
    出力: (batch, n_ch, d_model)
    """

    def __init__(
        self,
        n_ch: int,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        channel_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # 電極空間 bias — 距離が近いほど attention が強くなる先天的バイアス
        if channel_names is not None:
            dist = _build_electrode_dist_matrix(channel_names)  # (n_ch, n_ch)
            # 距離を負のバイアスに変換 (学習可能スケール付き)
            self.register_buffer("dist_bias_base", -dist)
        else:
            self.register_buffer("dist_bias_base", torch.zeros(n_ch, n_ch))

        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_ch, d_model)
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)  # each: (B, N, heads, d_head)
        q = q.transpose(1, 2)  # (B, heads, N, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        # 空間 prior バイアスを全 head に追加
        spatial_bias = self.bias_scale * self.dist_bias_base  # (N, N)
        attn = attn + spatial_bias.unsqueeze(0).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return residual + self.proj(out)


# ──────────────────────────────────────────────
# Riemann Covariance Pooling
# ──────────────────────────────────────────────

class RiemannCovPool(nn.Module):
    """
    時間方向に共分散行列を計算し、log-Euclidean 空間で圧縮する。
    入力: (batch, n_ch, time)
    出力: (batch, out_dim)

    学習可能な線形射影で次元削減後、log-Euclidean 計量の上三角を vectorize。
    """

    def __init__(self, n_ch: int, proj_dim: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.eps = eps
        self.proj = nn.Linear(n_ch, proj_dim, bias=False)
        n_tri = proj_dim * (proj_dim + 1) // 2
        self.out_dim = n_tri
        # 事前登録: DMLではtriu_indices/eyeが毎回CPU→デバイスコピーになるため
        self.register_buffer("_eye", torch.eye(proj_dim).unsqueeze(0))
        self.register_buffer("_tri_r", torch.triu_indices(proj_dim, proj_dim)[0])
        self.register_buffer("_tri_c", torch.triu_indices(proj_dim, proj_dim)[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_t = x.transpose(1, 2)          # (B, T, C)
        x_p = self.proj(x_t)             # (B, T, proj_dim)
        x_p = x_p - x_p.mean(dim=1, keepdim=True)

        cov = torch.bmm(x_p.transpose(1, 2), x_p) / (T - 1 + self.eps)
        cov = cov + self.eps * self._eye  # 正則化 (DML互換)

        # linalg.eigh は DML 非対応のため CPU で計算してデバイスに戻す
        cov_cpu = cov.cpu()
        L, V = torch.linalg.eigh(cov_cpu)
        L = torch.clamp(L, min=1e-12)
        log_cov = (V @ torch.diag_embed(torch.log(L)) @ V.transpose(-2, -1)).to(x.device)

        return log_cov[:, self._tri_r, self._tri_c]  # (B, n_tri)


# ──────────────────────────────────────────────
# CausalEEGConformer 本体
# ──────────────────────────────────────────────

class CausalEEGConformer(nn.Module):
    """
    独自モデル: 因果 TCN + Electrode Attention + Riemann Cov Pooling

    入力: (batch, n_channels, n_times)
    出力: (batch, n_classes) ロジット
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = 250,
        n_classes: int = 6,
        tcn_channels: list[int] | None = None,
        tcn_kernel: int = 8,
        attn_heads: int = 4,
        attn_dim: int = 64,
        cov_proj_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        channel_names: list[str] | None = None,
    ) -> None:
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [32, 64, 128]

        if channel_names is None:
            channel_names = CHANNELS

        # --- 入力投影 ---
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, attn_dim),
            nn.LayerNorm(attn_dim),
        )

        # --- Electrode Attention (空間次元) ---
        self.elec_attn = ElectrodeAttention(
            n_ch=n_channels,
            d_model=attn_dim,
            n_heads=attn_heads,
            dropout=dropout,
            channel_names=channel_names,
        )
        # (batch, n_ch, attn_dim) → (batch, attn_dim, n_times) に変換して TCN へ
        self.spatial_to_temporal = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),       # (B, n_ch*attn_dim)
            nn.Linear(n_channels * attn_dim, attn_dim),
            nn.GELU(),
        )

        # Electrode Attention を時間軸に適用するため: 先に時間方向に処理
        # 入力 (B, C, T) → Conv で特徴抽出 → Attn で空間統合
        self.input_conv = nn.Sequential(
            CausalConv1d(n_channels, attn_dim, kernel=3),
            nn.BatchNorm1d(attn_dim),  # Conv1d output: (B, C, T) → BatchNorm1d で C 次元を正規化
            nn.GELU(),
        )

        # --- Causal TCN スタック ---
        layers = []
        in_ch = attn_dim
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, tcn_kernel, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        tcn_out_ch = tcn_channels[-1]

        # --- Riemann Covariance Pooling ---
        self.riem_pool = RiemannCovPool(n_ch=tcn_out_ch, proj_dim=cov_proj_dim)
        riem_dim = self.riem_pool.out_dim

        # --- 時間方向 Global Average Pooling ---
        # 最終フレームの特徴 (因果的; 末尾フレームが現在時刻)
        self.gap_proj = nn.Linear(tcn_out_ch, hidden_dim // 2)

        # --- 分類ヘッド ---
        combined_dim = hidden_dim // 2 + riem_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, n_times)
        returns: (batch, n_classes) ロジット
        """
        # 1. 時間方向の入力畳み込み: (B, C, T) → (B, attn_dim, T)
        feat = self.input_conv(x)

        # 2. Causal TCN: (B, attn_dim, T) → (B, tcn_out, T)
        feat = self.tcn(feat)

        # 3a. Riemann Cov Pool: (B, tcn_out, T) → (B, riem_dim)
        riem_feat = self.riem_pool(feat)

        # 3b. 最終時刻の特徴 (因果的: ウィンドウ末尾 = 現在時刻): (B, tcn_out)
        last_feat = feat[:, :, -1]
        gap_feat = self.gap_proj(last_feat)  # (B, hidden_dim//2)

        # 4. 結合して分類
        combined = torch.cat([gap_feat, riem_feat], dim=1)
        return self.classifier(combined)
