# ============================================================
# Kaggle Notebook v4 sub-experiment: Loss Function Comparison
# Dataset: Grasp-and-Lift EEG Detection
#
# 目的: 仮説 A1「Focal loss の 0.5 局所安定性」を検証
#       3 つの損失関数で比較
#
# 比較対象:
#   (a) Focal loss (γ=2.0, α=0.25, label_smoothing=0.05) — v2 までの設定
#   (b) BCE + pos_weight (class imbalance 対策)
#   (c) BCE のみ (baseline)
#
# 設計:
#   - 3 被験者 × 3 損失 = 9 訓練
#   - EEGNet, window=500ms 固定 (ablation の中庸値)
#   - 同条件 (step=20, lr=5e-4, 50 epoch, augmentation なし)
#   - 学習曲線 + 予測確率の分布を保存
#
# 診断指標:
#   - val_auc 学習曲線 (崩壊 / 平坦化 / 改善のパターン分類)
#   - 予測確率の分布 ヒストグラム (0.5 付近に集中するか)
#   - per-event AUC (どのイベントでどの損失が強いか)
# ============================================================

# ─── セル 0: train.zip 解凍 (初回のみ) ──────────────────────
import zipfile, os
zip_path   = "/kaggle/input/competitions/grasp-and-lift-eeg-detection/train.zip"
extract_to = "/kaggle/working/data"
if not os.path.exists(extract_to + "/train"):
    print("解凍中...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print("解凍完了")
else:
    print("解凍済みスキップ")

# ─── セル 1: 環境確認 ────────────────────────────────────────
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── セル 2: インポート ──────────────────────────────────────
import json, math, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ─── セル 3: 定数 ───────────────────────────────────────────
TRAIN_DIR = Path("/kaggle/working/data/train")
WORK_DIR  = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "models_v4_loss"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SAMPLING_RATE = 500
N_CHANNELS    = 32
N_CLASSES     = 6
VAL_SERIES    = [7, 8]
WINDOW_SAMPLES = 500   # 1 秒。 ablation の中庸値

# 実験対象
LOSS_TYPES = ["focal", "bce_weighted", "bce_plain"]
SUBJECTS   = [1, 5, 9]

EVENTS = ["HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased"]

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8",
    "TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8",
    "PO9","O1","Oz","O2","PO10",
]

# ─── セル 4: 前処理 ─────────────────────────────────────────
def load_raw(subject: int, series: int):
    eeg = pd.read_csv(TRAIN_DIR / f"subj{subject}_series{series}_data.csv",
                      index_col="id")[CHANNELS].values.astype(np.float64)
    lbl = pd.read_csv(TRAIN_DIR / f"subj{subject}_series{series}_events.csv",
                      index_col="id")[EVENTS].values.astype(np.int8)
    return eeg, lbl

def causal_highpass(eeg, fs=500.0, cutoff=0.5, order=4):
    sos = butter(order, cutoff / (fs / 2), btype="high", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)

def causal_lowpass(eeg, fs=500.0, cutoff=45.0, order=4):
    sos = butter(order, cutoff / (fs / 2), btype="low", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)

def apply_car(eeg):
    return (eeg - eeg.mean(axis=1, keepdims=True)).astype(np.float32)

def causal_running_zscore(eeg, fs=500.0):
    n, c     = eeg.shape
    eeg64    = eeg.astype(np.float64)
    cum_sum  = np.cumsum(eeg64, axis=0)
    cum_sq   = np.cumsum(eeg64 ** 2, axis=0)
    counts   = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    sum_prev = np.vstack([np.zeros((1, c)), cum_sum[:-1]])
    sq_prev  = np.vstack([np.zeros((1, c)), cum_sq[:-1]])
    cnt_prev = np.maximum(counts - 1, 1)
    mean_p   = sum_prev / cnt_prev
    var_p    = np.maximum(sq_prev / cnt_prev - mean_p ** 2, 1e-16)
    out      = (eeg64 - mean_p) / (np.sqrt(var_p) + 1e-8)
    out[0]   = 0.0
    return out.astype(np.float32)

def preprocess(eeg, fs=500.0):
    eeg = causal_highpass(eeg, fs)
    eeg = causal_lowpass(eeg, fs)
    eeg = apply_car(eeg)
    eeg = causal_running_zscore(eeg, fs)
    return eeg

# ─── セル 5: EEGNet (完全 causal padding) ────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_times=500, n_classes=6,
                 f1=8, d=2, f2=16, kernel_len=64, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.ZeroPad2d((kernel_len - 1, 0, 0, 0)),
            nn.Conv2d(1, f1, (1, kernel_len), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f1*d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1*d), nn.GELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        sep_k = 16
        self.separable_conv = nn.Sequential(
            nn.ZeroPad2d((sep_k - 1, 0, 0, 0)),
            nn.Conv2d(f1*d, f1*d, (1, sep_k), groups=f1*d, bias=False),
            nn.Conv2d(f1*d, f2, 1, bias=False),
            nn.BatchNorm2d(f2), nn.GELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2 * (n_times // 32), n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x.flatten(1))

# ─── セル 6: 損失関数 3 種 ────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) — 不均衡対策"""
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05):
        super().__init__()
        self.gamma, self.alpha, self.ls = gamma, alpha, label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets >= 0.5, prob, 1 - prob)
        return (torch.where(targets >= 0.5, self.alpha, 1 - self.alpha) *
                (1 - pt) ** self.gamma * bce).mean()


class WeightedBCELoss(nn.Module):
    """BCE + per-class pos_weight (class imbalance 対策の標準手法)"""
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )


class PlainBCELoss(nn.Module):
    """BCE のみ (control group)"""
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def build_loss(loss_type: str, train_labels: np.ndarray) -> nn.Module:
    """損失関数を名前で構築"""
    if loss_type == "focal":
        return FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.05)
    elif loss_type == "bce_weighted":
        # 各イベントの pos_weight = neg/pos (class imbalance ratio)
        pos_count = train_labels.sum(axis=0)
        neg_count = len(train_labels) - pos_count
        pos_weight = torch.tensor(
            np.maximum(neg_count / np.maximum(pos_count, 1), 1.0),
            dtype=torch.float32,
        )
        return WeightedBCELoss(pos_weight)
    elif loss_type == "bce_plain":
        return PlainBCELoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

# ─── セル 7: 訓練ユーティリティ ─────────────────────────────
class EEGWindowDataset(Dataset):
    def __init__(self, eeg, labels, window_samples=500, step_samples=20,
                 positive_oversample_ratio=3.0):
        self.eeg    = torch.from_numpy(eeg.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.win    = window_samples
        all_ends = np.arange(window_samples - 1, len(eeg), step_samples, dtype=np.int32)
        is_pos   = labels[all_ends].any(axis=1)
        pos_ends = all_ends[is_pos]
        neg_ends = all_ends[~is_pos]
        if positive_oversample_ratio > 1 and len(pos_ends) > 0:
            pos_ends = np.tile(pos_ends, int(positive_oversample_ratio))
        self.ends = np.concatenate([neg_ends, pos_ends]).astype(np.int32)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, idx):
        end = int(self.ends[idx])
        return (self.eeg[end - self.win + 1:end + 1].T.contiguous(), self.labels[end])


def column_wise_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_score[:, i]
        if y_true[:, i].sum() == 0 or not np.isfinite(col).all():
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y_true[:, i], col)))
    return aucs  # 配列で返す (per-event 比較のため)


def train_one(model, train_eeg, train_labels, val_eeg, val_labels,
              loss_type, cfg, save_path):
    """単一モデル訓練 — 学習曲線と最終予測の分布を返す"""
    model = model.to(DEVICE)

    train_ds = EEGWindowDataset(train_eeg, train_labels,
                                cfg["window_samples"], cfg["train_step"], 3.0)
    val_ds   = EEGWindowDataset(val_eeg, val_labels,
                                cfg["window_samples"], cfg["val_step"], 1.0)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    criterion = build_loss(loss_type, train_labels).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"])

    best_auc, patience_cnt = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_per_event": []}
    final_pred_dist = None  # 最終 epoch の予測確率ヒストグラム

    for epoch in range(1, cfg["n_epochs"] + 1):
        # train
        model.train()
        total = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(X)
        tr_loss = total / max(len(train_ds), 1)
        scheduler.step()

        # eval
        model.eval()
        v_total, all_logits, all_y = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                v_loss = criterion(logits, y)
                if torch.isfinite(v_loss):
                    v_total += v_loss.item() * len(X)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                all_logits.append(logits.cpu().float())
                all_y.append(y.cpu())
        val_loss = v_total / max(len(val_ds), 1)
        y_score  = torch.sigmoid(torch.cat(all_logits)).numpy()
        y_true   = torch.cat(all_y).numpy().astype(np.int8)
        per_event = column_wise_auc(y_true, y_score)
        valid_aucs = [a for a in per_event if not np.isnan(a)]
        val_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_per_event"].append([float(a) for a in per_event])

        # 最終 epoch の予測確率ヒストグラム保存 (0.5 局所安定性診断用)
        # bins: [0.0, 0.1, ..., 1.0] = 11 edges, 10 bins
        hist_bins = np.linspace(0.0, 1.0, 11)
        final_pred_dist = {
            "bin_edges": hist_bins.tolist(),
            "counts": [
                int(np.histogram(y_score[:, i], bins=hist_bins)[0].sum())
                for i in range(N_CLASSES)
            ],
            "per_event_hist": [
                np.histogram(y_score[:, i], bins=hist_bins)[0].tolist()
                for i in range(N_CLASSES)
            ],
            "near_half_ratio": float(((y_score >= 0.4) & (y_score <= 0.6)).mean()),
        }

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1

        if patience_cnt >= cfg["patience"]:
            break

    return {
        "best_val_auc": best_auc,
        "history":      history,
        "n_epochs_run": epoch,
        "final_pred_dist": final_pred_dist,
    }

# ─── セル 8: データ準備 ───────────────────────────────────
print("=" * 60)
print(f"対象被験者: {SUBJECTS}, 損失: {LOSS_TYPES}, window={WINDOW_SAMPLES}")
print("データ読み込み & 前処理中...")
t_start = time.time()

subject_data = {}
for subj in SUBJECTS:
    tr_eegs, tr_lbls, va_eegs, va_lbls = [], [], [], []
    for ser in range(1, 9):
        eeg, lbl = load_raw(subj, ser)
        eeg = preprocess(eeg)
        if ser in VAL_SERIES:
            va_eegs.append(eeg); va_lbls.append(lbl)
        else:
            tr_eegs.append(eeg); tr_lbls.append(lbl)

    subject_data[subj] = {
        "tr_eeg": np.concatenate(tr_eegs), "tr_lbl": np.concatenate(tr_lbls),
        "va_eeg": np.concatenate(va_eegs), "va_lbl": np.concatenate(va_lbls),
    }
    print(f"  Subj {subj:2d}: train={len(subject_data[subj]['tr_eeg']):7,}  "
          f"val={len(subject_data[subj]['va_eeg']):7,} frames")

print(f"前処理完了: {time.time()-t_start:.1f}s\n")

# ─── セル 9: Loss Comparison 実行 ────────────────────────
print("=" * 60)
print("Loss Function Comparison 実行")
print("=" * 60)

BASE_CFG = {
    "window_samples": WINDOW_SAMPLES,
    "train_step":     20,
    "val_step":       25,
    "batch_size":     512,
    "lr":             5e-4,
    "weight_decay":   1e-4,
    "n_epochs":       50,
    "patience":       12,
}

results = {}
total_start = time.time()

for loss_type in LOSS_TYPES:
    print(f"\n--- Loss = {loss_type} ---")
    results[loss_type] = {}

    for subj in SUBJECTS:
        print(f"\n  Subj {subj:2d}, loss={loss_type}")
        t0 = time.time()
        save_path = MODEL_DIR / f"eegnet_{loss_type}_subj{subj}.pt"
        model = EEGNet(N_CHANNELS, WINDOW_SAMPLES, N_CLASSES)
        res = train_one(
            model,
            subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
            subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
            loss_type, BASE_CFG, save_path,
        )
        elapsed = time.time() - t0
        results[loss_type][subj] = {
            "best_val_auc":     res["best_val_auc"],
            "history":          res["history"],
            "n_epochs_run":     res["n_epochs_run"],
            "final_pred_dist":  res["final_pred_dist"],
            "time_sec":         elapsed,
        }
        near_half = res["final_pred_dist"]["near_half_ratio"]
        print(f"    >>> AUC={res['best_val_auc']:.4f}  near_0.5={near_half:.3f}  "
              f"({elapsed:.0f}s, {res['n_epochs_run']} ep)")

print(f"\n総実行時間: {(time.time() - total_start) / 60:.1f}分")

# ─── セル 10: 結果保存 ────────────────────────────────────
out_path = WORK_DIR / "04_loss_comparison.json"

serializable = {
    "config_base":  BASE_CFG,
    "subjects":     SUBJECTS,
    "loss_types":   LOSS_TYPES,
    "events":       EVENTS,
    "results": {
        loss_type: {
            str(s): {
                "best_val_auc":    float(results[loss_type][s]["best_val_auc"]),
                "history": {
                    "train_loss":    [float(v) for v in results[loss_type][s]["history"]["train_loss"]],
                    "val_loss":      [float(v) for v in results[loss_type][s]["history"]["val_loss"]],
                    "val_auc":       [float(v) for v in results[loss_type][s]["history"]["val_auc"]],
                    "val_per_event": results[loss_type][s]["history"]["val_per_event"],
                },
                "n_epochs_run":     int(results[loss_type][s]["n_epochs_run"]),
                "final_pred_dist":  results[loss_type][s]["final_pred_dist"],
                "time_sec":         float(results[loss_type][s]["time_sec"]),
            }
            for s in SUBJECTS
        }
        for loss_type in LOSS_TYPES
    },
}

with open(out_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\n結果保存: {out_path}")

# ─── セル 11: サマリー表示 ────────────────────────────────
print("\n" + "=" * 60)
print("Loss Comparison Summary")
print("=" * 60)
print(f"\n{'Subj':>4}  " + "  ".join(f"{lt:>13}" for lt in LOSS_TYPES))
print("-" * (8 + 15 * len(LOSS_TYPES)))
for subj in SUBJECTS:
    aucs = [results[lt][subj]["best_val_auc"] for lt in LOSS_TYPES]
    print(f"  {subj:2d}    " + "  ".join(f"{a:>13.4f}" for a in aucs))

print("-" * (8 + 15 * len(LOSS_TYPES)))
mean_aucs = [
    np.nanmean([results[lt][s]["best_val_auc"] for s in SUBJECTS])
    for lt in LOSS_TYPES
]
print(f"  Mean  " + "  ".join(f"{a:>13.4f}" for a in mean_aucs))

# 0.5 局所安定性診断
print(f"\n予測確率の 0.5 周辺 (0.4-0.6) への集中率:")
print(f"{'Subj':>4}  " + "  ".join(f"{lt:>13}" for lt in LOSS_TYPES))
print("-" * (8 + 15 * len(LOSS_TYPES)))
for subj in SUBJECTS:
    ratios = [results[lt][subj]["final_pred_dist"]["near_half_ratio"]
              for lt in LOSS_TYPES]
    print(f"  {subj:2d}    " + "  ".join(f"{r:>13.4f}" for r in ratios))

print("-" * (8 + 15 * len(LOSS_TYPES)))
mean_ratios = [
    np.mean([results[lt][s]["final_pred_dist"]["near_half_ratio"] for s in SUBJECTS])
    for lt in LOSS_TYPES
]
print(f"  Mean  " + "  ".join(f"{r:>13.4f}" for r in mean_ratios))

# 最良損失の特定
best_loss_idx = int(np.argmax(mean_aucs))
best_loss = LOSS_TYPES[best_loss_idx]
print(f"\n最良損失関数: {best_loss}")
print(f"平均 AUC: {mean_aucs[best_loss_idx]:.4f}")

# 仮説 A1 検証: focal の near_0.5 比率が他より高いか
focal_idx = LOSS_TYPES.index("focal") if "focal" in LOSS_TYPES else None
if focal_idx is not None:
    if mean_ratios[focal_idx] > max(r for i, r in enumerate(mean_ratios) if i != focal_idx):
        print("\n仮説 A1 検証: ✓ Focal loss は他損失より 0.5 周辺に予測が集中する傾向")
    else:
        print("\n仮説 A1 検証: ✗ Focal loss と他損失で 0.5 集中率に有意差なし")

print(f"\nモデル保存先: {MODEL_DIR}")
print("Output > ダウンロードで 04_loss_comparison.json と models_v4_loss/*.pt を取得してください。")
