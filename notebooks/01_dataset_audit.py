"""
ノートブック 01: データセット監査
全 96 ファイルの整合性確認、基本統計、ラベル分布

このスクリプトは Jupyter Notebook の代わりに直接実行可能。
結果は reports/figures/ と reports/results/ に保存される。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.eegdemo.constants import CHANNELS, EVENTS, SAMPLING_RATE, TRAIN_DIR, TEST_DIR
from src.eegdemo.io import load_raw, get_trial_boundaries

RESULTS_DIR = Path("reports/results")
FIG_DIR = Path("reports/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("データセット監査開始")
print("=" * 60)

# ──────────────────────────────────────────────
# 1. ファイル一覧と整合性チェック
# ──────────────────────────────────────────────
print("\n■ 1. ファイル一覧確認")

records = []
for subj in range(1, 13):
    for ser in range(1, 9):
        data_path = TRAIN_DIR / f"subj{subj}_series{ser}_data.csv"
        events_path = TRAIN_DIR / f"subj{subj}_series{ser}_events.csv"

        if not data_path.exists():
            print(f"  [警告] 欠損: {data_path.name}")
            continue

        eeg, labels = load_raw(subj, ser, "train")
        n_frames = len(eeg)
        duration_sec = n_frames / SAMPLING_RATE
        n_events_per_class = labels.sum(axis=0).tolist()
        n_trials = len(get_trial_boundaries(labels)) if labels is not None else 0
        has_nan = bool(np.isnan(eeg).any())
        amp_max = float(np.abs(eeg).max())

        records.append({
            "subject": subj,
            "series": ser,
            "n_frames": n_frames,
            "duration_sec": round(duration_sec, 1),
            "n_trials": n_trials,
            "has_nan": has_nan,
            "amp_max": round(amp_max, 1),
            **{f"n_{ev}": int(v) for ev, v in zip(EVENTS, n_events_per_class)},
        })

df = pd.DataFrame(records)
print(df.to_string(index=False))
df.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False)
print(f"\nCSV保存: {RESULTS_DIR / 'dataset_audit.csv'}")

# ──────────────────────────────────────────────
# 2. 基本統計サマリー
# ──────────────────────────────────────────────
print("\n■ 2. 基本統計")
print(f"  被験者数: {df['subject'].nunique()}")
print(f"  series/被験者: 8 (series 1-8)")
print(f"  フレーム数 — 平均: {df['n_frames'].mean():.0f}, 最小: {df['n_frames'].min()}, 最大: {df['n_frames'].max()}")
print(f"  duration — 平均: {df['duration_sec'].mean():.1f}秒, 合計: {df['duration_sec'].sum():.0f}秒 ({df['duration_sec'].sum()/3600:.1f}時間)")
print(f"  trial数 — 平均: {df['n_trials'].mean():.1f}/series, 合計: {df['n_trials'].sum()}")
print(f"  NaN: {'あり (要注意)' if df['has_nan'].any() else 'なし (OK)'}")
print(f"  最大振幅: {df['amp_max'].max():.0f} ADCカウント")

# ポジ率計算
total_frames = df['n_frames'].sum()
print("\n  イベント別ポジ率 (1 の割合):")
for ev in EVENTS:
    rate = df[f'n_{ev}'].sum() / total_frames * 100
    print(f"    {ev:30s}: {rate:.2f}%")

# ──────────────────────────────────────────────
# 3. 被験者・series間の duration 分布 (図)
# ──────────────────────────────────────────────
print("\n■ 3. duration 分布 → 図作成中...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) 被験者×series の duration ヒートマップ
pivot = df.pivot(index="subject", columns="series", values="duration_sec")
im = axes[0].imshow(pivot.values, aspect="auto", cmap="Blues")
axes[0].set_xticks(range(8))
axes[0].set_xticklabels([f"S{s}" for s in range(1, 9)])
axes[0].set_yticks(range(12))
axes[0].set_yticklabels([f"Subj{s}" for s in range(1, 13)], fontsize=7)
axes[0].set_title("記録長 (秒) — 被験者×Series")
plt.colorbar(im, ax=axes[0], label="秒")

# (b) trial 数の分布
pivot_trials = df.pivot(index="subject", columns="series", values="n_trials")
im2 = axes[1].imshow(pivot_trials.values, aspect="auto", cmap="Greens")
axes[1].set_xticks(range(8))
axes[1].set_xticklabels([f"S{s}" for s in range(1, 9)])
axes[1].set_yticks(range(12))
axes[1].set_yticklabels([f"Subj{s}" for s in range(1, 13)], fontsize=7)
axes[1].set_title("Trial 数 — 被験者×Series")
plt.colorbar(im2, ax=axes[1], label="trial数")

plt.tight_layout()
fig.savefig(FIG_DIR / "01a_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: {FIG_DIR / '01a_dataset_overview.png'}")

# ──────────────────────────────────────────────
# 4. ラベル分布の可視化
# ──────────────────────────────────────────────
print("\n■ 4. ラベル分布 → 図作成中...")

fig, ax = plt.subplots(figsize=(9, 4))
event_counts = [df[f'n_{ev}'].sum() for ev in EVENTS]
neg_counts = [total_frames - c for c in event_counts]
x = range(len(EVENTS))
ax.bar(x, event_counts, label="陽性 (1)", color="steelblue")
ax.bar(x, neg_counts, bottom=event_counts, label="陰性 (0)", color="lightgray")
ax.set_xticks(list(x))
ax.set_xticklabels(EVENTS, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("フレーム数")
ax.set_title("ラベル分布 — 全 train データ")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
for i, c in enumerate(event_counts):
    rate = c / total_frames * 100
    ax.text(i, c + total_frames * 0.002, f"{rate:.1f}%", ha="center", fontsize=8, color="steelblue")

plt.tight_layout()
fig.savefig(FIG_DIR / "01b_label_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: {FIG_DIR / '01b_label_distribution.png'}")

# ──────────────────────────────────────────────
# 5. EEG 時系列スニペット (Subject 1, Series 1)
# ──────────────────────────────────────────────
print("\n■ 5. EEG 時系列スニペット → 図作成中...")

eeg_sample, labels_sample = load_raw(1, 1, "train")
n_show = int(10 * SAMPLING_RATE)  # 最初の 10 秒
t_axis = np.arange(n_show) / SAMPLING_RATE

motor_ch_indices = [CHANNELS.index(ch) for ch in ["C3", "Cz", "C4", "FC1", "FC2"]]
motor_ch_names = ["C3", "Cz", "C4", "FC1", "FC2"]

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

ax_eeg = fig.add_subplot(gs[0])
offset = 0
offsets = []
for idx, name in zip(motor_ch_indices, motor_ch_names):
    sig = eeg_sample[:n_show, idx]
    sig_normalized = (sig - sig.mean()) / (sig.std() + 1e-8)
    ax_eeg.plot(t_axis, sig_normalized + offset, linewidth=0.7, label=name)
    offsets.append(offset)
    offset += 6

ax_eeg.set_yticks(offsets)
ax_eeg.set_yticklabels(motor_ch_names)
ax_eeg.set_ylabel("チャンネル (正規化)")
ax_eeg.set_title("EEG 時系列 — Subj1 Series1 最初の10秒 (運動関連チャンネル)")
ax_eeg.grid(alpha=0.3)
ax_eeg.set_xlim(0, 10)

ax_ev = fig.add_subplot(gs[1], sharex=ax_eeg)
colors_ev = plt.cm.tab10(np.linspace(0, 0.7, len(EVENTS)))
for i, (ev, col) in enumerate(zip(EVENTS, colors_ev)):
    ev_sig = labels_sample[:n_show, i].astype(float) * (i + 1)
    ax_ev.fill_between(t_axis, i, i + labels_sample[:n_show, i].astype(float), color=col, alpha=0.7, label=ev)

ax_ev.set_yticks(range(len(EVENTS)))
ax_ev.set_yticklabels(EVENTS, fontsize=7)
ax_ev.set_xlabel("時間 (秒)")
ax_ev.set_title("イベントラベル (±150ms窓)")
ax_ev.set_xlim(0, 10)

plt.tight_layout()
fig.savefig(FIG_DIR / "01c_eeg_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: {FIG_DIR / '01c_eeg_timeseries.png'}")

# ──────────────────────────────────────────────
# 6. line noise 確認
# ──────────────────────────────────────────────
print("\n■ 6. Line noise (50/60Hz) 確認...")

from src.eegdemo.preprocess import detect_line_noise
noise_info = detect_line_noise(eeg_sample, SAMPLING_RATE)
print(f"  50Hz SNR: {noise_info['50hz_snr_db']:.1f} dB  {'→ ノッチ要検討' if noise_info['50hz_snr_db'] > 10 else '→ 問題なし'}")
print(f"  60Hz SNR: {noise_info['60hz_snr_db']:.1f} dB  {'→ ノッチ要検討' if noise_info['60hz_snr_db'] > 10 else '→ 問題なし'}")

# ──────────────────────────────────────────────
# 7. サマリー保存
# ──────────────────────────────────────────────
summary = {
    "total_subjects": int(df["subject"].nunique()),
    "total_files": len(df),
    "total_frames": int(total_frames),
    "total_duration_hours": round(df["duration_sec"].sum() / 3600, 2),
    "total_trials": int(df["n_trials"].sum()),
    "has_nan": bool(df["has_nan"].any()),
    "positive_rate_pct": {ev: round(df[f'n_{ev}'].sum() / total_frames * 100, 3) for ev in EVENTS},
    "line_noise_snr_db": noise_info,
}
with open(RESULTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n  JSON保存: {RESULTS_DIR / 'dataset_summary.json'}")
print("\n" + "=" * 60)
print("データセット監査完了")
print("=" * 60)
