# ============================================================
# Kaggle Notebook v2: EEGNet + CausalEEGConformer (改善版)
# Dataset: Grasp-and-Lift EEG Detection
#
# v1からの改善点:
#   1. RiemannCovPool: eigh → Cholesky (NaN勾配問題を修正)
#   2. train_step: 50 → 20 (訓練窓数2.5倍)
#   3. Conformer LR: 1e-3 → 3e-4 + Linear warmup
#   4. EEGNet: 50 epochs + patience=10
#
# 根本原因: torch.linalg.eigh の backward がCUDA上で縮退固有値のとき
# NaN勾配を発生させ、モデルが崩壊していた (Brooks et al. NeurIPS 2019)
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

# ─── セル 3: パス・定数 ──────────────────────────────────────
TRAIN_DIR = Path("/kaggle/working/data/train")
WORK_DIR  = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "models_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SAMPLING_RATE = 500
N_CHANNELS    = 32
N_CLASSES     = 6
VAL_SERIES    = [7, 8]

EVENTS = ["HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased"]

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8",
    "TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8",
    "PO9","O1","Oz","O2","PO10",
]

CHANNEL_POS_2D = {
    "Fp1":(-0.18,0.85),"Fp2":(0.18,0.85),
    "F7":(-0.71,0.55),"F3":(-0.37,0.55),"Fz":(0.0,0.55),"F4":(0.37,0.55),"F8":(0.71,0.55),
    "FC5":(-0.60,0.27),"FC1":(-0.22,0.27),"FC2":(0.22,0.27),"FC6":(0.60,0.27),
    "T7":(-1.0,0.0),"C3":(-0.5,0.0),"Cz":(0.0,0.0),"C4":(0.5,0.0),"T8":(1.0,0.0),
    "TP9":(-0.95,-0.31),"CP5":(-0.60,-0.27),"CP1":(-0.22,-0.27),
    "CP2":(0.22,-0.27),"CP6":(0.60,-0.27),"TP10":(0.95,-0.31),
    "P7":(-0.71,-0.55),"P3":(-0.37,-0.55),"Pz":(0.0,-0.55),"P4":(0.37,-0.55),"P8":(0.71,-0.55),
    "PO9":(-0.50,-0.78),"O1":(-0.18,-0.85),"Oz":(0.0,-0.85),"O2":(0.18,-0.85),"PO10":(0.50,-0.78),
}

# ─── セル 4: データ読み込み・前処理 ──────────────────────────
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
    """Expanding window z-score — 完全因果的 (cumsum ベクトル化)"""
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

# ─── セル 5: モデル定義 ──────────────────────────────────────

# ---- EEGNet ----
class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_times=250, n_classes=6,
                 f1=8, d=2, f2=16, kernel_len=64, dropout=0.5):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.ZeroPad2d((kernel_len // 2 - 1, kernel_len // 2, 0, 0)),
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
            nn.ZeroPad2d((sep_k // 2 - 1, sep_k // 2, 0, 0)),
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


# ---- CausalEEGConformer ----
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, bias=False)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self.proj  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.act(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        out = self.drop(out)
        out = self.norm2(self.conv2(out).transpose(1,2)).transpose(1,2)
        return self.act(out + residual)


def _build_dist_matrix(channel_names):
    pos  = np.array([CHANNEL_POS_2D.get(ch, (0.0,0.0)) for ch in channel_names], dtype=np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    return torch.from_numpy(np.sqrt((diff**2).sum(-1)))


class ElectrodeAttention(nn.Module):
    def __init__(self, n_ch, d_model, n_heads=4, dropout=0.1, channel_names=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
        self.qkv  = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        dist = _build_dist_matrix(channel_names or CHANNELS)
        self.register_buffer("dist_bias_base", -dist)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, N, D = x.shape
        residual = x
        x   = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) / self.scale
        attn = attn + (self.bias_scale * self.dist_bias_base).unsqueeze(0).unsqueeze(0)
        attn = self.drop(attn.softmax(dim=-1))
        return residual + self.proj((attn @ v).transpose(1,2).reshape(B, N, D))


class CholeskyPool(nn.Module):
    """
    SPD共分散プーリング — Cholesky分解版 (v2: NaN安定)

    v1のlog-Euclidean (eigh) は CUDA上で固有値縮退時にNaN勾配が発生
    (Brooks et al. NeurIPS 2019)。Cholesky分解は正定値行列で常に安定。

    入力: (batch, n_ch, time)
    出力: (batch, proj_dim*(proj_dim+1)//2)
    """
    def __init__(self, n_ch, proj_dim=32, eps=1e-4):
        super().__init__()
        self.proj_dim = proj_dim
        self.eps      = eps
        self.proj     = nn.Linear(n_ch, proj_dim, bias=False)
        self.out_dim  = proj_dim * (proj_dim + 1) // 2
        # 正則化用単位行列とインデックスを事前登録
        self.register_buffer("_eye",   torch.eye(proj_dim).unsqueeze(0))
        self.register_buffer("_tri_r", torch.tril_indices(proj_dim, proj_dim)[0])
        self.register_buffer("_tri_c", torch.tril_indices(proj_dim, proj_dim)[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_p = self.proj(x.transpose(1, 2))            # (B, T, proj_dim)
        x_p = x_p - x_p.mean(dim=1, keepdim=True)

        # 共分散行列 + 強めの正則化で正定値性を保証
        cov = torch.bmm(x_p.transpose(1,2), x_p) / (T - 1 + self.eps)
        cov = cov + self.eps * self._eye               # eps=1e-4 (v1の1e-6より大)

        # Cholesky分解: cov = L L^T, L は下三角 (eigh より数値安定)
        try:
            L = torch.linalg.cholesky(cov)             # (B, proj_dim, proj_dim)
        except Exception:
            # Cholesky失敗 (非正定値) → 単位行列でフォールバック
            L = self._eye.expand(B, -1, -1)

        return L[:, self._tri_r, self._tri_c]          # (B, out_dim)


class CausalEEGConformer(nn.Module):
    """
    CausalEEGConformer v2
    変更: RiemannCovPool(eigh) → CholeskyPool (NaN安定版)
    """
    def __init__(self, n_channels=32, n_times=250, n_classes=6,
                 tcn_channels=None, tcn_kernel=8, attn_heads=4, attn_dim=64,
                 cov_proj_dim=32, hidden_dim=256, dropout=0.3, channel_names=None):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64, 128]

        self.input_conv = nn.Sequential(
            CausalConv1d(n_channels, attn_dim, kernel=3),
            nn.BatchNorm1d(attn_dim),
            nn.GELU(),
        )

        layers, in_ch = [], attn_dim
        for i, out_ch in enumerate(tcn_channels):
            layers.append(TCNBlock(in_ch, out_ch, tcn_kernel, 2**i, dropout))
            in_ch = out_ch
        self.tcn     = nn.Sequential(*layers)
        tcn_out      = tcn_channels[-1]

        self.cov_pool  = CholeskyPool(n_ch=tcn_out, proj_dim=cov_proj_dim)
        self.gap_proj  = nn.Linear(tcn_out, hidden_dim // 2)

        combined_dim = hidden_dim // 2 + self.cov_pool.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        feat = self.input_conv(x)        # (B, attn_dim, T)
        feat = self.tcn(feat)            # (B, tcn_out, T)
        cov  = self.cov_pool(feat)       # (B, cov_dim)
        gap  = self.gap_proj(feat[:,:,-1])  # (B, hidden//2)
        return self.classifier(torch.cat([gap, cov], dim=1))


# ─── セル 6: 訓練ユーティリティ ─────────────────────────────

class EEGWindowDataset(Dataset):
    def __init__(self, eeg, labels, window_samples=250, step_samples=20,
                 positive_oversample_ratio=3.0):
        self.eeg    = torch.from_numpy(eeg.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.win    = window_samples

        all_ends = np.arange(window_samples-1, len(eeg), step_samples, dtype=np.int32)
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
        return (self.eeg[end-self.win+1:end+1].T.contiguous(), self.labels[end])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05):
        super().__init__()
        self.gamma, self.alpha, self.ls = gamma, alpha, label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets >= 0.5, probs, 1 - probs)
        return (torch.where(targets >= 0.5, self.alpha, 1-self.alpha) *
                (1-pt)**self.gamma * bce).mean()


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup → CosineAnnealingLR"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            scale = (ep + 1) / self.warmup_epochs
        else:
            progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


def column_wise_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_score[:, i]
        if y_true[:, i].sum() == 0 or not np.isfinite(col).all():
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y_true[:, i], col)))
    valid = [a for a in aucs if not np.isnan(a)]
    return {"mean_auc": float(np.mean(valid)) if valid else float("nan"), "per_event": aucs}


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total, all_logits, all_y = 0.0, [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)
        if torch.isfinite(loss):
            total += loss.item() * len(X)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        all_logits.append(logits.cpu().float())
        all_y.append(y.cpu())
    y_score = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_true  = torch.cat(all_y).numpy().astype(np.int8)
    return total / max(len(loader.dataset), 1), y_true, y_score


def fit(model, train_eeg, train_labels, val_eeg, val_labels, cfg, save_path):
    model = model.to(DEVICE)
    win   = cfg["window_samples"]

    train_ds = EEGWindowDataset(train_eeg, train_labels, win, cfg.get("train_step", 20), 3.0)
    val_ds   = EEGWindowDataset(val_eeg,   val_labels,   win, cfg.get("val_step",   25), 1.0)

    n_tr, n_va = len(train_ds), len(val_ds)
    print(f"    dataset: train={n_tr:,} val={n_va:,} windows")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"]*2,
                              shuffle=False, num_workers=2, pin_memory=True)

    criterion = FocalLoss(cfg.get("focal_gamma", 2.0), label_smoothing=cfg.get("label_smoothing", 0.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=cfg.get("warmup_epochs", 5),
                                      total_epochs=cfg["n_epochs"])

    best_auc, patience_cnt = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        val_loss, y_true, y_score = eval_epoch(model, val_loader, criterion)
        val_auc  = column_wise_auc(y_true, y_score)["mean_auc"]

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"    Ep{epoch:3d}/{cfg['n_epochs']} | tr={tr_loss:.4f} val={val_loss:.4f} "
              f"auc={val_auc:.4f} lr={lr_now:.2e} | {time.time()-t0:.1f}s")

        if patience_cnt >= cfg["patience"]:
            print(f"    Early stop (patience={cfg['patience']})")
            break

    return {"best_val_auc": best_auc, "history": history}


# ─── セル 7: データ準備 ──────────────────────────────────────
print("=" * 60)
print("データ読み込み & 前処理中...")
t_start = time.time()

subject_data = {}
for subj in range(1, 13):
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

# ─── セル 8: EEGNet 訓練 ─────────────────────────────────────
EEGNET_CFG = {
    "window_samples": 250, "train_step": 20, "val_step": 25,
    "batch_size": 512,  "lr": 5e-4, "weight_decay": 1e-4,
    "n_epochs": 50, "patience": 10, "warmup_epochs": 5,
    "focal_gamma": 2.0, "label_smoothing": 0.05,
}

print("=" * 60)
print("[1] EEGNet v1 (within-subject, step=20)")
print("=" * 60)

eegnet_aucs = []
for subj in range(1, 13):
    print(f"\nSubj {subj:2d}")
    t0  = time.time()
    res = fit(EEGNet(N_CHANNELS, 250, N_CLASSES),
              subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
              subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
              EEGNET_CFG, MODEL_DIR / f"eegnet_subj{subj}.pt")
    eegnet_aucs.append(res["best_val_auc"])
    print(f"  >>> Subj {subj:2d}  AUC={res['best_val_auc']:.4f}  ({time.time()-t0:.0f}s)")

print(f"\nEEGNet  mean AUC: {np.nanmean(eegnet_aucs):.4f} ± {np.nanstd(eegnet_aucs):.4f}")

# ─── セル 9: CausalEEGConformer v2 訓練 ──────────────────────
CONFORMER_CFG = {
    "window_samples": 250, "train_step": 20, "val_step": 25,
    "batch_size": 512, "lr": 3e-4, "weight_decay": 1e-4,
    "n_epochs": 80, "patience": 12, "warmup_epochs": 5,
    "focal_gamma": 2.0, "label_smoothing": 0.05,
}

print("\n" + "=" * 60)
print("[2] CausalEEGConformer v2 (CholeskyPool, LR warmup)")
print("=" * 60)

conformer_aucs      = []
conformer_histories = []
for subj in range(1, 13):
    print(f"\nSubj {subj:2d}")
    t0    = time.time()
    model = CausalEEGConformer(
        n_channels=N_CHANNELS, n_times=250, n_classes=N_CLASSES,
        tcn_channels=[32, 64, 128], tcn_kernel=8, attn_heads=4, attn_dim=64,
        cov_proj_dim=32, hidden_dim=256, dropout=0.3, channel_names=CHANNELS,
    )
    res = fit(model,
              subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
              subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
              CONFORMER_CFG, MODEL_DIR / f"conformer_subj{subj}.pt")
    conformer_aucs.append(res["best_val_auc"])
    conformer_histories.append(res["history"])
    print(f"  >>> Subj {subj:2d}  AUC={res['best_val_auc']:.4f}  ({time.time()-t0:.0f}s)")

print(f"\nConformer mean AUC: {np.nanmean(conformer_aucs):.4f} ± {np.nanstd(conformer_aucs):.4f}")

# ─── セル 10: 結果保存・サマリー ─────────────────────────────
# ベースライン（LR）との比較用（なければスキップ）
lr_aucs = [0.7739,0.8050,0.7152,0.7870,0.6670,0.7436,0.7680,0.7719,0.6729,0.7510,0.6981,0.7019]

results = {
    "version": "v2",
    "changes": "RiemannCovPool(eigh) -> CholeskyPool, train_step=20, lr=3e-4, warmup",
    "lr_baseline":   {"mean_auc": round(np.mean(lr_aucs), 4), "per_subject": lr_aucs},
    "eegnet_within": {
        "mean_auc": round(float(np.nanmean(eegnet_aucs)), 4),
        "std_auc":  round(float(np.nanstd(eegnet_aucs)), 4),
        "ci_95": [
            round(float(np.nanmean(eegnet_aucs) - 1.96*np.nanstd(eegnet_aucs)/np.sqrt(12)), 4),
            round(float(np.nanmean(eegnet_aucs) + 1.96*np.nanstd(eegnet_aucs)/np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in eegnet_aucs],
    },
    "conformer_within": {
        "mean_auc": round(float(np.nanmean(conformer_aucs)), 4),
        "std_auc":  round(float(np.nanstd(conformer_aucs)), 4),
        "ci_95": [
            round(float(np.nanmean(conformer_aucs) - 1.96*np.nanstd(conformer_aucs)/np.sqrt(12)), 4),
            round(float(np.nanmean(conformer_aucs) + 1.96*np.nanstd(conformer_aucs)/np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in conformer_aucs],
        "histories": [{"val_auc": h["val_auc"]} if h else None for h in conformer_histories],
    },
}

with open(WORK_DIR / "04_dl_results_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n結果保存: {WORK_DIR}/04_dl_results_v2.json")

# ─── セル 11: サマリー表示 ───────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS v2  (within-subject, val series 7-8 holdout)")
print("=" * 60)
print(f"\n{'Subj':>4}  {'LR base':>8}  {'EEGNet':>8}  {'Conformer':>10}  {'Conf-LR':>8}")
print("-" * 48)
for i in range(12):
    e, c, lr = eegnet_aucs[i], conformer_aucs[i], lr_aucs[i]
    print(f"  {i+1:2d}   {lr:.4f}    {e:.4f}    {c:.4f}    {c-lr:+.4f}")
print("-" * 48)
print(f"  Mean {np.mean(lr_aucs):.4f}    {np.nanmean(eegnet_aucs):.4f}    "
      f"{np.nanmean(conformer_aucs):.4f}    {np.nanmean(conformer_aucs)-np.mean(lr_aucs):+.4f}")

print(f"\nモデル保存先: {MODEL_DIR}")
print("Output > ダウンロードで 04_dl_results_v2.json と models_v2/*.pt を取得してください。")
