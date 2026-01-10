"""
ノートブック 04 (再開版): DL モデル訓練
- EEGNet subjects 1-3: 保存済みモデルを検証セットで再評価
- EEGNet subjects 4-12: 高速設定で新規訓練
- CausalEEGConformer subjects 1-12: 高速設定で訓練
- step_samples=100 (元の10から10倍高速化)
"""
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from src.eegdemo.constants import CHANNELS, SAMPLING_RATE
from src.eegdemo.io import load_raw
from src.eegdemo.preprocess import preprocess
from src.eegdemo.models import CausalEEGConformer, EEGNet
from src.eegdemo.train import fit, get_device, EEGWindowDataset, FocalLoss, eval_epoch
from src.eegdemo.eval import column_wise_auc
from torch.utils.data import DataLoader

RESULTS_DIR = Path("reports/results")
MODEL_DIR = Path("reports/models")
FIG_DIR = Path("reports/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/model.yaml", encoding="utf-8") as f:
    cfg_full = yaml.safe_load(f)

cfg = cfg_full["causal_eeg_conformer"]
N_CHANNELS = cfg_full["n_channels"]
N_CLASSES = cfg_full["n_classes"]
VAL_SERIES = [7, 8]
WINDOW_SAMPLES = cfg.get("window_samples", 250)

device = get_device(cfg_full.get("device", "dml"))
print("=" * 60)
print(f"DL 訓練再開 — device: {device}")
print("=" * 60)

# ──────────────────────────────────────────────
# データ準備
# ──────────────────────────────────────────────
print("\nデータ読み込み & 前処理中...")
t_load = time.time()

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
        "val_eeg":   np.concatenate(val_eegs,   axis=0),
        "val_lbl":   np.concatenate(val_lbls,   axis=0),
    }
    n_tr  = len(subject_data[subj]["train_eeg"])
    n_val = len(subject_data[subj]["val_eeg"])
    print(f"  Subj {subj:2d}: train={n_tr:7,}frames  val={n_val:7,}frames")

print(f"  前処理完了: {time.time()-t_load:.1f}s\n")

# ──────────────────────────────────────────────
# 共通設定
# ──────────────────────────────────────────────
TRAIN_STEP  = 100   # 高速化: 元の10から10倍
VAL_STEP    = 25
BATCH       = 256

eegnet_cfg = {
    "window_samples":      WINDOW_SAMPLES,
    "train_step_samples":  TRAIN_STEP,
    "val_step_samples":    VAL_STEP,
    "batch_size":          BATCH,
    "lr":                  5e-4,
    "weight_decay":        1e-4,
    "n_epochs":            20,
    "patience":            6,
    "focal_gamma":         2.0,
    "label_smoothing":     0.05,
}

conformer_cfg = {
    "window_samples":      WINDOW_SAMPLES,
    "train_step_samples":  TRAIN_STEP,
    "val_step_samples":    VAL_STEP,
    "batch_size":          BATCH,
    "lr":                  cfg.get("lr", 1e-3),
    "weight_decay":        cfg.get("weight_decay", 1e-4),
    "n_epochs":            30,
    "patience":            8,
    "focal_gamma":         cfg.get("focal_gamma", 2.0),
    "label_smoothing":     cfg.get("label_smoothing", 0.05),
}


def _eval_saved_model(model: torch.nn.Module, save_path: Path, cfg_eval: dict, subj_id: int) -> float:
    """保存済みモデルをロードして検証 AUC を返す。ファイルがなければ nan。"""
    if not save_path.exists():
        return float("nan")
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    model = model.to(device)
    val_ds = EEGWindowDataset(
        subject_data[subj_id]["val_eeg"],
        subject_data[subj_id]["val_lbl"],
        window_samples=cfg_eval["window_samples"],
        step_samples=cfg_eval.get("val_step_samples", 25),
        positive_oversample_ratio=1.0,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
    criterion = FocalLoss(
        gamma=cfg_eval.get("focal_gamma", 2.0),
        label_smoothing=cfg_eval.get("label_smoothing", 0.05),
    )
    _, y_true, y_score = eval_epoch(model, val_loader, criterion, device)
    auc_dict = column_wise_auc(y_true, y_score)
    return auc_dict["mean_auc"]


# ──────────────────────────────────────────────
# 1. EEGNet
# ──────────────────────────────────────────────
print("[1] EEGNet within-subject 訓練...")
eegnet_within_aucs = []

for subj in range(1, 13):
    save_path = MODEL_DIR / f"eegnet_subj{subj}.pt"
    t_subj = time.time()

    if subj <= 3 and save_path.exists():
        # 保存済みモデルを評価のみ
        model = EEGNet(n_channels=N_CHANNELS, n_times=WINDOW_SAMPLES, n_classes=N_CLASSES)
        auc = _eval_saved_model(model, save_path, eegnet_cfg, subj)
        print(f"  Subj {subj:2d} [loaded ] val AUC = {auc:.4f}  ({time.time()-t_subj:.0f}s)")
    else:
        model = EEGNet(n_channels=N_CHANNELS, n_times=WINDOW_SAMPLES, n_classes=N_CLASSES)
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
        auc = result["best_val_auc"]
        print(f"  Subj {subj:2d} [trained] val AUC = {auc:.4f}  ({time.time()-t_subj:.0f}s)")

    eegnet_within_aucs.append(auc)

print(f"\n  EEGNet mean AUC: {np.nanmean(eegnet_within_aucs):.4f} ± {np.nanstd(eegnet_within_aucs):.4f}\n")

# ──────────────────────────────────────────────
# 2. CausalEEGConformer
# ──────────────────────────────────────────────
print("[2] CausalEEGConformer within-subject 訓練...")
conformer_within_aucs = []
conformer_histories = []

for subj in range(1, 13):
    save_path = MODEL_DIR / f"conformer_subj{subj}.pt"

    # 既に完成済みなら評価のみ
    if save_path.exists():
        model = CausalEEGConformer(
            n_channels=N_CHANNELS, n_times=WINDOW_SAMPLES, n_classes=N_CLASSES,
            tcn_channels=cfg.get("tcn_channels", [32, 64, 128]),
            tcn_kernel=cfg.get("tcn_kernel_size", 8),
            attn_heads=cfg.get("electrode_attn_heads", 4),
            attn_dim=cfg.get("electrode_attn_dim", 64),
            cov_proj_dim=cfg.get("cov_dim", 32),
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=cfg.get("dropout", 0.3),
            channel_names=CHANNELS,
        )
        auc = _eval_saved_model(model, save_path, conformer_cfg, subj)
        print(f"  Subj {subj:2d} [loaded ] val AUC = {auc:.4f}")
        conformer_within_aucs.append(auc)
        conformer_histories.append(None)
        continue

    t_subj = time.time()
    model = CausalEEGConformer(
        n_channels=N_CHANNELS, n_times=WINDOW_SAMPLES, n_classes=N_CLASSES,
        tcn_channels=cfg.get("tcn_channels", [32, 64, 128]),
        tcn_kernel=cfg.get("tcn_kernel_size", 8),
        attn_heads=cfg.get("electrode_attn_heads", 4),
        attn_dim=cfg.get("electrode_attn_dim", 64),
        cov_proj_dim=cfg.get("cov_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.3),
        channel_names=CHANNELS,
    )
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
    auc = result["best_val_auc"]
    conformer_within_aucs.append(auc)
    conformer_histories.append(result["history"])
    print(f"  Subj {subj:2d} [trained] val AUC = {auc:.4f}  ({time.time()-t_subj:.0f}s)")

print(f"\n  Conformer mean AUC: {np.nanmean(conformer_within_aucs):.4f} ± {np.nanstd(conformer_within_aucs):.4f}\n")

# ──────────────────────────────────────────────
# 3. モデル比較図
# ──────────────────────────────────────────────
print("[3] モデル比較図を生成中...")

try:
    with open(RESULTS_DIR / "03_baseline_results.json", encoding="utf-8") as f:
        baseline_res = json.load(f)
    lr_within_aucs = baseline_res["within_subject"]["per_subject"]
    has_baseline = True
except FileNotFoundError:
    lr_within_aucs = None
    has_baseline = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.eegdemo.font_setup import setup_japanese_font
    setup_japanese_font()
except Exception:
    pass

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(12)
w = 0.25

if has_baseline:
    ax.bar(x - w, lr_within_aucs, w,
           label=f"LR+BandPower (μ={np.mean(lr_within_aucs):.3f})",
           color="lightgray", edgecolor="gray")
ax.bar(x,     eegnet_within_aucs,    w,
       label=f"EEGNet (μ={np.nanmean(eegnet_within_aucs):.3f})",
       color="steelblue", edgecolor="navy")
ax.bar(x + w, conformer_within_aucs, w,
       label=f"CausalConformer (μ={np.nanmean(conformer_within_aucs):.3f})",
       color="darkorange", edgecolor="saddlebrown")

ax.set_xticks(list(x))
ax.set_xticklabels([f"S{i+1}" for i in range(12)])
ax.set_ylabel("Best Val AUC (mean 6 events)")
ax.set_title("Within-Subject Model Comparison (series 7-8 holdout)")
ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="chance")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
fig.savefig(FIG_DIR / "04_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  保存: 04_model_comparison.png")

# ──────────────────────────────────────────────
# 4. 結果保存
# ──────────────────────────────────────────────
dl_results = {
    "train_step_samples": TRAIN_STEP,
    "eegnet_within": {
        "mean_auc": round(float(np.nanmean(eegnet_within_aucs)), 4),
        "std_auc":  round(float(np.nanstd(eegnet_within_aucs)), 4),
        "ci_95": [
            round(float(np.nanmean(eegnet_within_aucs) - 1.96 * np.nanstd(eegnet_within_aucs) / np.sqrt(12)), 4),
            round(float(np.nanmean(eegnet_within_aucs) + 1.96 * np.nanstd(eegnet_within_aucs) / np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in eegnet_within_aucs],
    },
    "conformer_within": {
        "mean_auc": round(float(np.nanmean(conformer_within_aucs)), 4),
        "std_auc":  round(float(np.nanstd(conformer_within_aucs)), 4),
        "ci_95": [
            round(float(np.nanmean(conformer_within_aucs) - 1.96 * np.nanstd(conformer_within_aucs) / np.sqrt(12)), 4),
            round(float(np.nanmean(conformer_within_aucs) + 1.96 * np.nanstd(conformer_within_aucs) / np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in conformer_within_aucs],
    },
}

out_path = RESULTS_DIR / "04_dl_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dl_results, f, ensure_ascii=False, indent=2)

print(f"\n  結果保存: {out_path}")
print("\n" + "=" * 60)
print("DL 訓練完了")
print("=" * 60)
print(f"\n  LR baseline      : {np.mean(lr_within_aucs):.4f}" if has_baseline else "")
print(f"  EEGNet           : {np.nanmean(eegnet_within_aucs):.4f}")
print(f"  CausalConformer  : {np.nanmean(conformer_within_aucs):.4f}")
