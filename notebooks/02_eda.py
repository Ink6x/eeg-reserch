"""
ノートブック 02: 探索的データ分析 (EDA)
- チャンネル別分布
- 周波数スペクトル (全帯域)
- ERP 平均波形 (HandStart / LiftOff)
- 被験者間ばらつき確認
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq

from src.eegdemo.font_setup import setup_japanese_font
from src.eegdemo.constants import (
    CHANNELS, EVENTS, MOTOR_CHANNELS, SAMPLING_RATE, FRONTAL_CHANNELS,
)
from src.eegdemo.io import load_raw, get_trial_boundaries
from src.eegdemo.preprocess import preprocess

setup_japanese_font()
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EDA 開始")
print("=" * 60)

# ──────────────────────────────────────────────
# データ読み込み (Subject 1-6, Series 1 のみ代表として)
# ──────────────────────────────────────────────
print("\nデータ読み込み中...")

# 代表 1 件の詳細 EDA 用
eeg_raw, labels = load_raw(1, 1, "train")
eeg_proc = preprocess(eeg_raw, SAMPLING_RATE)

motor_idx = [CHANNELS.index(ch) for ch in MOTOR_CHANNELS]

# ──────────────────────────────────────────────
# 1. チャンネル別振幅分布 (前処理前後)
# ──────────────────────────────────────────────
print("\n[1] チャンネル別振幅分布...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, eeg, title in [(axes[0], eeg_raw, "前処理前 (生信号)"),
                        (axes[1], eeg_proc, "前処理後 (CAR + filter + z-score)")]:
    ch_std = eeg.std(axis=0)
    colors = ["steelblue" if ch in MOTOR_CHANNELS else "lightgray" for ch in CHANNELS]
    ax.bar(range(len(CHANNELS)), ch_std, color=colors)
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=7)
    ax.set_ylabel("標準偏差")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("標準偏差 (ADC count)")
axes[1].set_ylabel("標準偏差 (z-score単位)")
plt.tight_layout()
fig.savefig(FIG_DIR / "02a_channel_amplitude.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: 02a_channel_amplitude.png")

# ──────────────────────────────────────────────
# 2. 周波数スペクトル (Welch PSD)
# ──────────────────────────────────────────────
print("\n[2] 周波数スペクトル...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, eeg, title in [(axes[0], eeg_raw, "生信号 PSD"),
                        (axes[1], eeg_proc, "前処理後 PSD")]:
    for ch_name in MOTOR_CHANNELS:
        idx = CHANNELS.index(ch_name)
        freqs, psd = welch(eeg[:, idx], fs=SAMPLING_RATE, nperseg=1024)
        mask = freqs <= 80
        ax.semilogy(freqs[mask], psd[mask], linewidth=0.8, label=ch_name)

    # 帯域境界
    for f, label in [(0.5, "HP"), (4, "δ"), (8, "θ"), (13, "α/μ"), (30, "β"), (45, "LP")]:
        ax.axvline(f, color="gray", linestyle=":", linewidth=0.7)
        ax.text(f, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-10, label, fontsize=6, color="gray")

    ax.set_xlabel("周波数 (Hz)")
    ax.set_ylabel("PSD (power/Hz)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(FIG_DIR / "02b_psd_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: 02b_psd_comparison.png")

# ──────────────────────────────────────────────
# 3. ERP 平均波形 (HandStart を中心に epoch 化)
# ──────────────────────────────────────────────
print("\n[3] ERP 平均波形 (HandStart)...")

def extract_epochs(eeg: np.ndarray, labels: np.ndarray, event_col: int,
                   tmin_ms: float = -2000, tmax_ms: float = 1000,
                   fs: float = SAMPLING_RATE) -> np.ndarray | None:
    """
    イベントの立ち上がり (0→1) を起点に epoch を切り出す。
    注: ERP 解析用なので causal 制約は不要 (retrospective解析)。
    """
    tmin = int(tmin_ms * fs / 1000)
    tmax = int(tmax_ms * fs / 1000)
    n_times = tmax - tmin

    event_signal = labels[:, event_col]
    onsets = np.where(np.diff(event_signal.astype(int)) == 1)[0] + 1

    epochs = []
    for onset in onsets:
        start = onset + tmin
        end = onset + tmax
        if start < 0 or end > len(eeg):
            continue
        epochs.append(eeg[start:end])

    return np.stack(epochs) if epochs else None  # (n_epochs, n_times, n_ch)


# 全被験者の HandStart ERP を収集 (Series 1 のみ代表)
print("  全被験者の ERP 収集中...")
all_erp_epochs = []  # (total_epochs, n_times, n_ch)
for subj in range(1, 13):
    eeg_s, lbl_s = load_raw(subj, 1, "train")
    eeg_p = preprocess(eeg_s, SAMPLING_RATE)
    epochs = extract_epochs(eeg_p, lbl_s, event_col=0)  # HandStart
    if epochs is not None:
        all_erp_epochs.append(epochs)
    print(f"  Subj {subj}: {len(epochs) if epochs is not None else 0} epochs")

all_erp = np.concatenate(all_erp_epochs, axis=0)  # (N, n_times, n_ch)
tmin_ms, tmax_ms = -2000, 1000
n_times = all_erp.shape[1]
times_ms = np.linspace(tmin_ms, tmax_ms, n_times)

print(f"  合計 {len(all_erp)} epochs")

# ベースライン補正 (-2000 ~ -1500 ms)
bl_start = int((tmin_ms - tmin_ms) / (tmax_ms - tmin_ms) * n_times)
bl_end = int(500 / (tmax_ms - tmin_ms) * n_times)
baseline_mean = all_erp[:, bl_start:bl_end, :].mean(axis=1, keepdims=True)
all_erp_bc = all_erp - baseline_mean

# ERP 図
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# 上段: 全運動関連チャンネル
ax = axes[0]
for ch in MOTOR_CHANNELS:
    idx = CHANNELS.index(ch)
    mean_erp = all_erp_bc[:, :, idx].mean(axis=0)
    sem = all_erp_bc[:, :, idx].std(axis=0) / np.sqrt(len(all_erp_bc))
    ax.plot(times_ms, mean_erp, label=ch, linewidth=1.2)
    ax.fill_between(times_ms, mean_erp - sem, mean_erp + sem, alpha=0.15)

ax.axvline(0, color="k", linestyle="--", linewidth=1.5, label="HandStart")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvspan(-1500, -200, alpha=0.05, color="blue", label="RP 期待域")
ax.set_ylabel("振幅 (z-score)")
ax.set_title(f"ERP — HandStart (N={len(all_erp_bc)} epochs, 全被験者 Series1)")
ax.legend(fontsize=8, ncol=4)
ax.grid(alpha=0.3)

# 下段: Cz のみ拡大
ax2 = axes[1]
cz_idx = CHANNELS.index("Cz")
mean_cz = all_erp_bc[:, :, cz_idx].mean(axis=0)
sem_cz = all_erp_bc[:, :, cz_idx].std(axis=0) / np.sqrt(len(all_erp_bc))
ax2.plot(times_ms, mean_cz, color="steelblue", linewidth=2, label="Cz")
ax2.fill_between(times_ms, mean_cz - sem_cz, mean_cz + sem_cz, alpha=0.3, color="steelblue")
ax2.axvline(0, color="k", linestyle="--", linewidth=1.5)
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.set_xlabel("イベントからの時間 (ms)")
ax2.set_ylabel("振幅 (z-score)")
ax2.set_title("Cz 単チャンネル ERP (Readiness Potential 観察)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "02c_erp_handstart.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: 02c_erp_handstart.png")

# ──────────────────────────────────────────────
# 4. 被験者間ばらつき (各被験者の Cz ERP)
# ──────────────────────────────────────────────
print("\n[4] 被験者間 Cz ERP 比較...")

fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.tab20(np.linspace(0, 1, 12))
for i, subj_epochs in enumerate(all_erp_epochs):
    # ベースライン補正
    bc = subj_epochs - subj_epochs[:, bl_start:bl_end, :].mean(axis=1, keepdims=True)
    mean_cz_subj = bc[:, :, cz_idx].mean(axis=0)
    n_ep = len(bc)
    ax.plot(times_ms, mean_cz_subj, color=cmap[i], linewidth=1.0, alpha=0.8, label=f"Subj{i+1}(n={n_ep})")

ax.axvline(0, color="k", linestyle="--", linewidth=1.5)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("イベントからの時間 (ms)")
ax.set_ylabel("振幅 (z-score)")
ax.set_title("被験者間 Cz ERP 比較 — HandStart")
ax.legend(fontsize=6, ncol=4, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "02d_erp_between_subjects.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: 02d_erp_between_subjects.png")

# ──────────────────────────────────────────────
# 5. μ/β 帯域パワー時間変化 (ERD/ERS 概観)
# ──────────────────────────────────────────────
print("\n[5] ERD/ERS 概観 (C3/C4 μ帯)...")

def band_envelope(epoch_ch: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """短時間窓のバンドパワー包絡線 (非因果: ERP 解析用)"""
    from scipy.signal import filtfilt, butter
    nyq = fs / 2
    sos_b = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    from scipy.signal import sosfiltfilt
    filtered = sosfiltfilt(sos_b, epoch_ch)
    return filtered ** 2


# HandStart 前後の μ帯パワー変化
mu_power_c3, mu_power_c4 = [], []
c3_idx, c4_idx = CHANNELS.index("C3"), CHANNELS.index("C4")

for subj_epochs in all_erp_epochs:
    for ep in subj_epochs:
        mu_power_c3.append(band_envelope(ep[:, c3_idx], SAMPLING_RATE, 8.0, 13.0))
        mu_power_c4.append(band_envelope(ep[:, c4_idx], SAMPLING_RATE, 8.0, 13.0))

mu_c3 = np.stack(mu_power_c3)  # (n_epochs, n_times)
mu_c4 = np.stack(mu_power_c4)

# ベースライン正規化 (dB)
bl_power_c3 = mu_c3[:, bl_start:bl_end].mean(axis=1, keepdims=True)
bl_power_c4 = mu_c4[:, bl_start:bl_end].mean(axis=1, keepdims=True)
erd_c3 = 10 * np.log10((mu_c3 + 1e-12) / (bl_power_c3 + 1e-12))
erd_c4 = 10 * np.log10((mu_c4 + 1e-12) / (bl_power_c4 + 1e-12))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(times_ms, erd_c3.mean(axis=0), color="blue", linewidth=1.5, label="C3 (対側)")
ax.fill_between(times_ms, erd_c3.mean(0) - erd_c3.std(0) / np.sqrt(len(erd_c3)),
                erd_c3.mean(0) + erd_c3.std(0) / np.sqrt(len(erd_c3)), alpha=0.2, color="blue")
ax.plot(times_ms, erd_c4.mean(axis=0), color="red", linewidth=1.5, label="C4 (同側)")
ax.fill_between(times_ms, erd_c4.mean(0) - erd_c4.std(0) / np.sqrt(len(erd_c4)),
                erd_c4.mean(0) + erd_c4.std(0) / np.sqrt(len(erd_c4)), alpha=0.2, color="red")

ax.axvline(0, color="k", linestyle="--", linewidth=1.5)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("イベントからの時間 (ms)")
ax.set_ylabel("ERD/ERS (dB vs baseline)")
ax.set_title("mu帯 (8-13 Hz) ERD/ERS — HandStart (C3 vs C4)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "02e_erd_mu_handstart.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: 02e_erd_mu_handstart.png")

print("\n" + "=" * 60)
print("EDA 完了")
print("=" * 60)
print("\n生成された図:")
for f in sorted(FIG_DIR.glob("02*.png")):
    print(f"  {f.name}")
