"""
ノートブック 04: CausalEEGConformer 訓練
- AMD GPU (DirectML) または CPU で動作
- 被験者ごとに within-subject モデルを訓練
- 全被験者合算の cross-subject モデルも訓練
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from src.eegdemo.constants import CHANNELS, EVENTS, SAMPLING_RATE
from src.eegdemo.io import load_raw
from src.eegdemo.preprocess import preprocess
from src.eegdemo.models import CausalEEGConformer, EEGNet
from src.eegdemo.train import fit, get_device
from src.eegdemo.eval import column_wise_auc, bootstrap_ci, cohens_d_paired, permutation_test_paired

RESULTS_DIR = Path("reports/results")
MODEL_DIR = Path("reports/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/model.yaml", encoding="utf-8") as f:
    cfg_full = yaml.safe_load(f)

cfg = cfg_full["causal_eeg_conformer"]
N_CHANNELS = cfg_full["n_channels"]
N_CLASSES = cfg_full["n_classes"]
VAL_SERIES = [7, 8]
WINDOW_SAMPLES = cfg.get("window_samples", 250)

device = get_device(cfg_full.get("device", "dml"))
print("=" * 60)
print(f"CausalEEGConformer 訓練 — device: {device}")
print("=" * 60)

# ──────────────────────────────────────────────
# データ準備
# ──────────────────────────────────────────────
print("\nデータ読み込み & 前処理中...")

subject_data: dict[int, dict] = {}
for subj in range(1, 13):
    train_eegs, train_lbls = [], []
    val_eegs, val_lbls = [], []
    for ser in range(1, 9):
        eeg_raw, lbl = load_raw(subj, ser, "train")
        eeg_proc = preprocess(eeg_raw, SAMPLING_RATE)
        if ser in VAL_SERIES:
            val_eegs.append(eeg_proc)
            val_lbls.append(lbl)
        else:
            train_eegs.append(eeg_proc)
            train_lbls.append(lbl)

    subject_data[subj] = {
        "train_eeg": np.concatenate(train_eegs, axis=0),
        "train_lbl": np.concatenate(train_lbls, axis=0),
        "val_eeg": np.concatenate(val_eegs, axis=0),
        "val_lbl": np.concatenate(val_lbls, axis=0),
    }
    n_tr = len(subject_data[subj]["train_eeg"])
    n_val = len(subject_data[subj]["val_eeg"])
    print(f"  Subj {subj:2d}: train={n_tr:7d}frames, val={n_val:7d}frames")

# ──────────────────────────────────────────────
# 1. EEGNet ベースライン (DL)
# ──────────────────────────────────────────────
print("\n[1] EEGNet within-subject 訓練...")

eegnet_within_aucs = []
eegnet_results = []

eegnet_cfg = {
    "window_samples": WINDOW_SAMPLES,
    "batch_size": 256,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "n_epochs": 30,
    "patience": 8,
    "focal_gamma": 2.0,
    "label_smoothing": 0.05,
    "device": str(device),
}

for subj in range(1, 13):
    print(f"\n  Subj {subj}...")
    model = EEGNet(n_channels=N_CHANNELS, n_times=WINDOW_SAMPLES, n_classes=N_CLASSES)
    save_path = MODEL_DIR / f"eegnet_subj{subj}.pt"

    result = fit(
        model=model,
        train_eeg=subject_data[subj]["train_eeg"],
        train_labels=subject_data[subj]["train_lbl"],
        val_eeg=subject_data[subj]["val_eeg"],
        val_labels=subject_data[subj]["val_lbl"],
        cfg=eegnet_cfg,
        save_path=save_path,
        device=device,
    )

    best_auc = result["best_val_auc"]
    eegnet_within_aucs.append(best_auc)
    eegnet_results.append({"subject": subj, "best_val_auc": best_auc})
    print(f"  Subj {subj} | best val AUC = {best_auc:.4f}")

print(f"\n  EEGNet Within mean AUC: {np.nanmean(eegnet_within_aucs):.4f} ± {np.nanstd(eegnet_within_aucs):.4f}")

# ──────────────────────────────────────────────
# 2. CausalEEGConformer (独自モデル) within-subject
# ──────────────────────────────────────────────
print("\n[2] CausalEEGConformer within-subject 訓練...")

conformer_within_aucs = []
conformer_results = []

conformer_cfg = {
    "window_samples": WINDOW_SAMPLES,
    "batch_size": cfg.get("batch_size", 512),
    "lr": cfg.get("lr", 1e-3),
    "weight_decay": cfg.get("weight_decay", 1e-4),
    "n_epochs": cfg.get("n_epochs", 100),
    "patience": cfg.get("patience", 15),
    "focal_gamma": cfg.get("focal_gamma", 2.0),
    "label_smoothing": cfg.get("label_smoothing", 0.05),
    "device": str(device),
}

for subj in range(1, 13):
    print(f"\n  Subj {subj}...")
    model = CausalEEGConformer(
        n_channels=N_CHANNELS,
        n_times=WINDOW_SAMPLES,
        n_classes=N_CLASSES,
        tcn_channels=cfg.get("tcn_channels", [32, 64, 128]),
        tcn_kernel=cfg.get("tcn_kernel_size", 8),
        attn_heads=cfg.get("electrode_attn_heads", 4),
        attn_dim=cfg.get("electrode_attn_dim", 64),
        cov_proj_dim=cfg.get("cov_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.3),
        channel_names=CHANNELS,
    )
    save_path = MODEL_DIR / f"conformer_subj{subj}.pt"

    result = fit(
        model=model,
        train_eeg=subject_data[subj]["train_eeg"],
        train_labels=subject_data[subj]["train_lbl"],
        val_eeg=subject_data[subj]["val_eeg"],
        val_labels=subject_data[subj]["val_lbl"],
        cfg=conformer_cfg,
        save_path=save_path,
        device=device,
    )

    best_auc = result["best_val_auc"]
    conformer_within_aucs.append(best_auc)
    conformer_results.append({
        "subject": subj,
        "best_val_auc": best_auc,
        "history": result["history"],
    })
    print(f"  Subj {subj} | best val AUC = {best_auc:.4f}")

print(f"\n  Conformer Within mean AUC: {np.nanmean(conformer_within_aucs):.4f} ± {np.nanstd(conformer_within_aucs):.4f}")

# ──────────────────────────────────────────────
# 3. モデル比較
# ──────────────────────────────────────────────
print("\n[3] モデル比較...")

# ベースライン結果の読み込み
try:
    with open(RESULTS_DIR / "03_baseline_results.json", encoding="utf-8") as f:
        baseline_res = json.load(f)
    lr_within_aucs = baseline_res["within_subject"]["per_subject"]
    has_baseline = True
except FileNotFoundError:
    lr_within_aucs = None
    has_baseline = False
    print("  ベースライン結果が見つかりません。LR との比較をスキップ。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.eegdemo.font_setup import setup_japanese_font
FIG_DIR = Path("reports/figures")
setup_japanese_font()

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(12)
w = 0.25

if has_baseline:
    ax.bar(x - w, lr_within_aucs, w, label=f"LR+BandPower ({np.mean(lr_within_aucs):.3f})", color="lightgray")
ax.bar(x, eegnet_within_aucs, w, label=f"EEGNet ({np.nanmean(eegnet_within_aucs):.3f})", color="steelblue")
ax.bar(x + w, conformer_within_aucs, w, label=f"CausalConformer ({np.nanmean(conformer_within_aucs):.3f})", color="darkorange")

ax.set_xticks(list(x))
ax.set_xticklabels([f"S{i+1}" for i in range(12)])
ax.set_ylabel("Best Val AUC (mean 6 events)")
ax.set_title("Within-Subject モデル比較")
ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="chance")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
fig.savefig(FIG_DIR / "04_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  保存: 04_model_comparison.png")

# ──────────────────────────────────────────────
# 4. 結果保存
# ──────────────────────────────────────────────
dl_results = {
    "eegnet_within": {
        "mean_auc": round(float(np.nanmean(eegnet_within_aucs)), 4),
        "std_auc": round(float(np.nanstd(eegnet_within_aucs)), 4),
        "per_subject": [round(v, 4) for v in eegnet_within_aucs],
    },
    "conformer_within": {
        "mean_auc": round(float(np.nanmean(conformer_within_aucs)), 4),
        "std_auc": round(float(np.nanstd(conformer_within_aucs)), 4),
        "per_subject": [round(v, 4) for v in conformer_within_aucs],
    },
}

with open(RESULTS_DIR / "04_dl_results.json", "w", encoding="utf-8") as f:
    json.dump(dl_results, f, ensure_ascii=False, indent=2)

print(f"\n  JSON保存: {RESULTS_DIR / '04_dl_results.json'}")
print("\n" + "=" * 60)
print("DL 訓練完了")
print("=" * 60)
print(f"\n  EEGNet       mean AUC: {np.nanmean(eegnet_within_aucs):.4f}")
print(f"  CausalConformer AUC: {np.nanmean(conformer_within_aucs):.4f}")
