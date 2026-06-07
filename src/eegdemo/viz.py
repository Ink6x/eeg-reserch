"""
可視化モジュール — 判断のための図のみ。飾りは作らない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI なし環境でも動作
import matplotlib.pyplot as plt
import numpy as np

from .constants import CHANNEL_POS_2D, CHANNELS, EVENTS, MOTOR_CHANNELS, SAMPLING_RATE

FIG_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# トポグラフィマップ
# ──────────────────────────────────────────────

def plot_topomap(
    values: np.ndarray,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    save_name: str | None = None,
) -> plt.Figure:
    """
    32ch EEG のスカラー値をトポグラフィプロット。
    values: (32,) — チャンネル順は CHANNELS と一致
    """
    from matplotlib.patches import Circle
    from scipy.interpolate import griddata

    positions = np.array([CHANNEL_POS_2D.get(ch, (0.0, 0.0)) for ch in CHANNELS])
    x, y = positions[:, 0], positions[:, 1]

    xi = np.linspace(-1.1, 1.1, 200)
    yi = np.linspace(-1.1, 1.1, 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), values, (xi, yi), method="cubic")

    # 頭部外をマスク
    mask = np.sqrt(xi**2 + yi**2) > 1.05
    zi[mask] = np.nan

    fig, ax = plt.subplots(figsize=(4, 4))
    if vmin is None:
        vmin = np.nanpercentile(values, 5)
    if vmax is None:
        vmax = np.nanpercentile(values, 95)
    sym = max(abs(vmin), abs(vmax))

    im = ax.contourf(xi, yi, zi, levels=64, cmap=cmap, vmin=-sym, vmax=sym)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.scatter(x, y, c="k", s=20, zorder=5)
    for ch, (cx, cy) in CHANNEL_POS_2D.items():
        if ch in MOTOR_CHANNELS:
            ax.annotate(ch, (cx, cy), fontsize=6, ha="center", va="bottom")

    # 頭部の輪郭
    circle = Circle((0, 0), 1.0, color="k", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    if save_name:
        _save(fig, save_name)
    return fig


# ──────────────────────────────────────────────
# ERP 平均波形
# ──────────────────────────────────────────────

def plot_erp(
    epochs: np.ndarray,
    fs: float = SAMPLING_RATE,
    tmin_ms: float = -2000.0,
    channels: list[str] | None = None,
    event_name: str = "HandStart",
    ci: bool = True,
    save_name: str | None = None,
) -> plt.Figure:
    """
    epoch 平均 (ERP) を描画。
    epochs: (n_epochs, n_times, n_channels)
    """
    if channels is None:
        channels = MOTOR_CHANNELS

    n_times = epochs.shape[1]
    times = np.arange(n_times) / fs * 1000 + tmin_ms  # ms

    fig, ax = plt.subplots(figsize=(8, 4))
    for ch in channels:
        if ch not in CHANNELS:
            continue
        idx = CHANNELS.index(ch)
        mean_erp = epochs[:, :, idx].mean(axis=0)
        ax.plot(times, mean_erp, label=ch, linewidth=1.2)
        if ci:
            sem = epochs[:, :, idx].std(axis=0) / np.sqrt(len(epochs))
            ax.fill_between(times, mean_erp - sem, mean_erp + sem, alpha=0.2)

    ax.axvline(0, color="k", linestyle="--", linewidth=1, label="イベント")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("振幅 (a.u.)")
    ax.set_title(f"ERP — {event_name} (n={len(epochs)})")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    if save_name:
        _save(fig, save_name)
    return fig


# ──────────────────────────────────────────────
# ERD/ERS 時間周波数マップ
# ──────────────────────────────────────────────

def plot_tfr(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    channel: str = "Cz",
    event_name: str = "HandStart",
    vmin: float = -1.0,
    vmax: float = 1.0,
    save_name: str | None = None,
) -> plt.Figure:
    """
    時間周波数表現 (ERD/ERS) を描画。
    power: (n_freqs, n_times) — ベースラインで正規化済みの値 (dB or %)
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.pcolormesh(
        times, freqs, power,
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
    )
    plt.colorbar(im, ax=ax, label="ERD/ERS (dB)")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)

    # 帯域境界線
    for f in [4, 8, 13, 30]:
        ax.axhline(f, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("周波数 (Hz)")
    ax.set_title(f"時間周波数解析 — {channel} / {event_name}")

    if save_name:
        _save(fig, save_name)
    return fig


# ──────────────────────────────────────────────
# 分類結果
# ──────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC曲線",
    save_name: str | None = None,
) -> plt.Figure:
    """6イベント全ての ROC 曲線を 1 図に描画"""
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(EVENTS)))

    for i, (event, color) in enumerate(zip(EVENTS, colors, strict=True)):
        yt = y_true[:, i]
        ys = y_score[:, i]
        if yt.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{event} (AUC={auc_val:.3f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("偽陽性率")
    ax.set_ylabel("真陽性率")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    if save_name:
        _save(fig, save_name)
    return fig


def plot_subject_auc_boxplot(
    fold_results: list[dict],
    save_name: str | None = None,
) -> plt.Figure:
    """被験者ごとの AUC 箱ひげ図 (イベント別)"""
    data_per_event = {ev: [] for ev in EVENTS}
    for r in fold_results:
        for ev in EVENTS:
            v = r["roc_auc"].get(ev, float("nan"))
            data_per_event[ev].append(v)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(
        [data_per_event[ev] for ev in EVENTS],
        labels=EVENTS,
        patch_artist=True,
        medianprops={"color": "k", "linewidth": 2},
    )
    ax.axhline(0.5, color="r", linestyle="--", linewidth=1, label="チャンス水準")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("被験者横断 CV — イベント別 AUC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if save_name:
        _save(fig, save_name)
    return fig
