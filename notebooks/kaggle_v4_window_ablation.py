# ============================================================
# Kaggle Notebook v4 sub-experiment: Window Size Ablation
# Dataset: Grasp-and-Lift EEG Detection
#
# 目的: 仮説 S2「受容野と RP 時間スケール不整合」を検証
#       window_samples を 250 → 750 まで変化させ AUC を比較
#
# 設計:
#   - 3 被験者 × 4 window サイズ = 12 訓練
#   - EEGNet のみ (Conformer は時間がかかるので window 確定後に)
#   - 同条件 (step=20, lr=5e-4, 50 epoch, focal loss)
#   - augmentation なし (window 効果を isolation するため)
#
# 期待結果:
#   - 仮説が正しいなら window 拡大で +0.02 〜 +0.05 改善
#   - とくに S9 (LR で 0.673 → DL v2 で 0.546) で顕著改善
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
MODEL_DIR = WORK_DIR / "models_v4_ablation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SAMPLING_RATE = 500
N_CHANNELS    = 32
N_CLASSES     = 6
VAL_SERIES    = [7, 8]

# 実験対象
WINDOW_SIZES = [250, 375, 500, 750]    # 500ms, 750ms, 1.0s, 1.5s
SUBJECTS     = [1, 5, 9]                # representative: high/medium/low LR baseline

EVENTS = ["HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased"]

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8",
    "TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8",
    "PO9","O1","Oz","O2","PO10",
]

# ─── セル 4: 前処理 (v2 と同じ) ───────────────────────────
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

# ─── セル 5: EEGNet (n_times パラメータ化済み) ──────────────
class EEGNet(nn.Module):
    """因果的 EEGNet。n_times に応じて分類ヘッドが自動調整される。"""
    def __init__(self, n_channels=32, n_times=250, n_classes=6,
                 f1=8, d=2, f2=16, kernel_len=64, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.ZeroPad2d((kernel_len - 1, 0, 0, 0)),  # 完全 causal pad
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
            nn.ZeroPad2d((sep_k - 1, 0, 0, 0)),  # 完全 causal pad
            nn.Conv2d(f1*d, f1*d, (1, sep_k), groups=f1*d, bias=False),
            nn.Conv2d(f1*d, f2, 1, bias=False),
            nn.BatchNorm2d(f2), nn.GELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        # n_times は 32 で割り切れる前提。 250 → 7, 375 → 11, 500 → 15, 750 → 23
        self.classifier = nn.Linear(f2 * (n_times // 32), n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x.flatten(1))

# ─── セル 6: 訓練ユーティリティ ─────────────────────────────
class EEGWindowDataset(Dataset):
    def __init__(self, eeg, labels, window_samples=250, step_samples=20,
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


class FocalLoss(nn.Module):
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


def column_wise_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_score[:, i]
        if y_true[:, i].sum() == 0 or not np.isfinite(col).all():
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y_true[:, i], col)))
    valid = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


def train_one(model, train_eeg, train_labels, val_eeg, val_labels, cfg, save_path):
    """単一モデル訓練"""
    model = model.to(DEVICE)
    win = cfg["window_samples"]

    train_ds = EEGWindowDataset(train_eeg, train_labels, win, cfg["train_step"], 3.0)
    val_ds   = EEGWindowDataset(val_eeg,   val_labels,   win, cfg["val_step"],   1.0)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"])

    best_auc, patience_cnt = -1.0, 0
    history = {"train_loss": [], "val_auc": []}

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
        all_logits, all_y = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = torch.nan_to_num(model(X), nan=0.0, posinf=10.0, neginf=-10.0)
                all_logits.append(logits.cpu().float())
                all_y.append(y.cpu())
        y_score = torch.sigmoid(torch.cat(all_logits)).numpy()
        y_true  = torch.cat(all_y).numpy().astype(np.int8)
        val_auc = column_wise_auc(y_true, y_score)

        history["train_loss"].append(tr_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1

        if patience_cnt >= cfg["patience"]:
            break

    return {"best_val_auc": best_auc, "history": history, "n_epochs_run": epoch}

# ─── セル 7: データ準備 (3 被験者のみ) ────────────────────
print("=" * 60)
print(f"対象被験者: {SUBJECTS}, window サイズ: {WINDOW_SIZES}")
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

# ─── セル 8: Window Ablation 実行 ─────────────────────────
print("=" * 60)
print("Window Size Ablation 実行")
print("=" * 60)

BASE_CFG = {
    "train_step":   20,
    "val_step":     25,
    "batch_size":   512,
    "lr":           5e-4,
    "weight_decay": 1e-4,
    "n_epochs":     50,
    "patience":     12,
}

results = {}  # results[window][subject] = {best_val_auc, history, n_epochs_run, time_sec}
total_start = time.time()

for window_samples in WINDOW_SIZES:
    print(f"\n--- Window = {window_samples} samples ({window_samples * 1000 / SAMPLING_RATE:.0f}ms) ---")
    results[window_samples] = {}
    cfg = {**BASE_CFG, "window_samples": window_samples}

    for subj in SUBJECTS:
        print(f"\n  Subj {subj:2d}, window={window_samples}")
        t0 = time.time()
        save_path = MODEL_DIR / f"eegnet_w{window_samples}_subj{subj}.pt"
        model = EEGNet(N_CHANNELS, window_samples, N_CLASSES)
        res = train_one(
            model,
            subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
            subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
            cfg, save_path,
        )
        elapsed = time.time() - t0
        results[window_samples][subj] = {
            "best_val_auc": res["best_val_auc"],
            "history":      res["history"],
            "n_epochs_run": res["n_epochs_run"],
            "time_sec":     elapsed,
        }
        print(f"    >>> AUC={res['best_val_auc']:.4f}  ({elapsed:.0f}s, {res['n_epochs_run']} ep)")

print(f"\n総実行時間: {(time.time() - total_start) / 60:.1f}分")

# ─── セル 9: 結果保存 + サマリー ──────────────────────────
out_path = WORK_DIR / "04_window_ablation.json"

# JSON シリアライズ可能な形に変換
serializable = {
    "config_base": BASE_CFG,
    "subjects": SUBJECTS,
    "window_sizes": WINDOW_SIZES,
    "results": {
        str(w): {
            str(s): {
                "best_val_auc": float(results[w][s]["best_val_auc"]),
                "history":      {k: [float(v) for v in vs] for k, vs in results[w][s]["history"].items()},
                "n_epochs_run": int(results[w][s]["n_epochs_run"]),
                "time_sec":     float(results[w][s]["time_sec"]),
            }
            for s in SUBJECTS
        }
        for w in WINDOW_SIZES
    },
}

with open(out_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\n結果保存: {out_path}")

# ─── セル 10: 可視化サマリー ──────────────────────────────
print("\n" + "=" * 60)
print("Window Ablation Summary")
print("=" * 60)
print(f"\n{'Subj':>4}  " + "  ".join(f"w={w:>4}" for w in WINDOW_SIZES))
print("-" * (8 + 8 * len(WINDOW_SIZES)))
for subj in SUBJECTS:
    aucs = [results[w][subj]["best_val_auc"] for w in WINDOW_SIZES]
    print(f"  {subj:2d}    " + "  ".join(f"{a:.4f}" for a in aucs))

print("-" * (8 + 8 * len(WINDOW_SIZES)))
mean_aucs = [
    np.nanmean([results[w][s]["best_val_auc"] for s in SUBJECTS])
    for w in WINDOW_SIZES
]
print(f"  Mean  " + "  ".join(f"{a:.4f}" for a in mean_aucs))

# 最良 window の特定
best_window_idx = int(np.argmax(mean_aucs))
best_window = WINDOW_SIZES[best_window_idx]
print(f"\n最良 window サイズ: {best_window} samples ({best_window * 1000 / SAMPLING_RATE:.0f}ms)")
print(f"平均 AUC: {mean_aucs[best_window_idx]:.4f}")

# 訓練時間サマリー
print("\n訓練時間 (秒):")
print(f"{'Subj':>4}  " + "  ".join(f"w={w:>4}" for w in WINDOW_SIZES))
for subj in SUBJECTS:
    times = [results[w][subj]["time_sec"] for w in WINDOW_SIZES]
    print(f"  {subj:2d}    " + "  ".join(f"{t:>6.0f}" for t in times))

print(f"\nモデル保存先: {MODEL_DIR}")
print("Output > ダウンロードで 04_window_ablation.json と models_v4_ablation/*.pt を取得してください。")
