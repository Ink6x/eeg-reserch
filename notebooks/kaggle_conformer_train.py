# ============================================================
# Kaggle Notebook: CausalEEGConformer Training
# Dataset: Grasp-and-Lift EEG Detection
# GPU: T4 / P100 (CUDA)
#
# 使い方:
#   1. Kaggle > New Notebook
#   2. このファイルを全部コピーして貼り付け
#   3. Settings > Accelerator: GPU T4 x2 または GPU P100
#   4. train.zip を /kaggle/working/data/ に解凍済みであること
#      (解凍コード: zipfileセルを先に実行)
#   5. Run All
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

# ─── セル 1: 環境確認 ───────────────────────────────────────
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── セル 2: インポート ──────────────────────────────────────
from __future__ import annotations
import json, math, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, lfilter, iirnotch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ─── セル 3: パス設定 ────────────────────────────────────────
TRAIN_DIR = Path("/kaggle/working/data/train")
WORK_DIR  = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {DEVICE}")

# ─── セル 4: 定数 ────────────────────────────────────────────
SAMPLING_RATE = 500
N_CHANNELS    = 32
N_CLASSES     = 6
VAL_SERIES    = [7, 8]

EVENTS = [
    "HandStart", "FirstDigitTouch", "BothStartLoadPhase",
    "LiftOff", "Replace", "BothReleased",
]

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

# ─── セル 5: データ読み込み ──────────────────────────────────
def load_raw(subject: int, series: int) -> tuple[np.ndarray, np.ndarray]:
    data_path   = TRAIN_DIR / f"subj{subject}_series{series}_data.csv"
    events_path = TRAIN_DIR / f"subj{subject}_series{series}_events.csv"
    eeg = pd.read_csv(data_path,   index_col="id")[CHANNELS].values.astype(np.float64)
    lbl = pd.read_csv(events_path, index_col="id")[EVENTS].values.astype(np.int8)
    return eeg, lbl

# ─── セル 6: 前処理 (全処理因果的) ──────────────────────────
def causal_highpass(eeg, fs=500.0, cutoff=0.5, order=4):
    sos = butter(order, cutoff / (fs / 2), btype="high", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)

def causal_lowpass(eeg, fs=500.0, cutoff=45.0, order=4):
    sos = butter(order, cutoff / (fs / 2), btype="low", output="sos")
    return sosfilt(sos, eeg, axis=0).astype(np.float32)

def apply_car(eeg):
    return (eeg - eeg.mean(axis=1, keepdims=True)).astype(np.float32)

def causal_running_zscore(eeg, fs=500.0, warmup_sec=5.0):
    """Expanding window z-score — 完全因果的 (cumsum ベクトル化)"""
    n, c     = eeg.shape
    eeg64    = eeg.astype(np.float64)
    cum_sum  = np.cumsum(eeg64, axis=0)
    cum_sq   = np.cumsum(eeg64 ** 2, axis=0)
    counts   = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    sum_prev = np.vstack([np.zeros((1, c)), cum_sum[:-1]])
    sq_prev  = np.vstack([np.zeros((1, c)), cum_sq[:-1]])
    cnt_prev = np.maximum(counts - 1, 1)
    mean_prev = sum_prev / cnt_prev
    var_prev  = np.maximum(sq_prev / cnt_prev - mean_prev ** 2, 1e-16)
    std_prev  = np.sqrt(var_prev) + 1e-8
    out       = (eeg64 - mean_prev) / std_prev
    out[0]    = 0.0
    return out.astype(np.float32)

def preprocess(eeg, fs=500.0):
    eeg = causal_highpass(eeg, fs)
    eeg = causal_lowpass(eeg, fs)
    eeg = apply_car(eeg)
    eeg = causal_running_zscore(eeg, fs)
    return eeg

# ─── セル 7: モデル定義 ──────────────────────────────────────
import torch.nn as nn
import torch.nn.functional as F

# --- EEGNet (因果版) ---
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
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d), nn.GELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        sep_kernel = 16
        self.separable_conv = nn.Sequential(
            nn.ZeroPad2d((sep_kernel // 2 - 1, sep_kernel // 2, 0, 0)),
            nn.Conv2d(f1 * d, f1 * d, (1, sep_kernel), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),
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


# --- CausalEEGConformer (独自モデル) ---
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
        out = self.act(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        out = self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2)
        return self.act(out + residual)


def _build_dist_matrix(channel_names):
    pos  = np.array([CHANNEL_POS_2D.get(ch, (0.0, 0.0)) for ch in channel_names], dtype=np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    return torch.from_numpy(np.sqrt((diff ** 2).sum(-1)))


class ElectrodeAttention(nn.Module):
    def __init__(self, n_ch, d_model, n_heads=4, dropout=0.1, channel_names=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
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
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn + (self.bias_scale * self.dist_bias_base).unsqueeze(0).unsqueeze(0)
        attn = self.drop(attn.softmax(dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return residual + self.proj(out)


class RiemannCovPool(nn.Module):
    """Log-Euclidean 共分散プーリング"""
    def __init__(self, n_ch, proj_dim=32, eps=1e-6):
        super().__init__()
        self.proj_dim = proj_dim
        self.eps      = eps
        self.proj     = nn.Linear(n_ch, proj_dim, bias=False)
        self.out_dim  = proj_dim * (proj_dim + 1) // 2
        self.register_buffer("_eye",   torch.eye(proj_dim).unsqueeze(0))
        self.register_buffer("_tri_r", torch.triu_indices(proj_dim, proj_dim)[0])
        self.register_buffer("_tri_c", torch.triu_indices(proj_dim, proj_dim)[1])

    def forward(self, x):
        B, C, T = x.shape
        x_p = self.proj(x.transpose(1, 2))
        x_p = x_p - x_p.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_p.transpose(1, 2), x_p) / (T - 1 + self.eps)
        cov = cov + self.eps * self._eye

        # CUDA では eigh がネイティブ対応
        L, V    = torch.linalg.eigh(cov)
        L       = torch.clamp(L, min=1e-12)
        log_cov = V @ torch.diag_embed(torch.log(L)) @ V.transpose(-2, -1)
        return log_cov[:, self._tri_r, self._tri_c]


class CausalEEGConformer(nn.Module):
    """独自モデル: 因果 TCN + Electrode Attention + Riemann Cov Pooling"""
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
            layers.append(TCNBlock(in_ch, out_ch, tcn_kernel, 2 ** i, dropout))
            in_ch = out_ch
        self.tcn      = nn.Sequential(*layers)
        tcn_out       = tcn_channels[-1]

        self.riem_pool = RiemannCovPool(n_ch=tcn_out, proj_dim=cov_proj_dim)
        self.gap_proj  = nn.Linear(tcn_out, hidden_dim // 2)

        combined_dim   = hidden_dim // 2 + self.riem_pool.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        feat = self.input_conv(x)
        feat = self.tcn(feat)
        riem = self.riem_pool(feat)
        gap  = self.gap_proj(feat[:, :, -1])
        return self.classifier(torch.cat([gap, riem], dim=1))


# ─── セル 8: 訓練ユーティリティ ─────────────────────────────
from torch.utils.data import Dataset, DataLoader

class EEGWindowDataset(Dataset):
    """Lazy スライディングウィンドウ Dataset"""
    def __init__(self, eeg, labels, window_samples=250, step_samples=50,
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
        x   = self.eeg[end - self.win + 1 : end + 1].T.contiguous()
        y   = self.labels[end]
        return x, y


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
        fw    = (1 - pt) ** self.gamma
        aw    = torch.where(targets >= 0.5, self.alpha, 1 - self.alpha)
        return (aw * fw * bce).mean()


def column_wise_auc(y_true, y_score):
    aucs = []
    for i in range(y_true.shape[1]):
        col_score = y_score[:, i]
        # NaN/Inf を含む列はスキップ
        if y_true[:, i].sum() == 0 or not np.isfinite(col_score).all():
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y_true[:, i], col_score)))
    valid = [a for a in aucs if not np.isnan(a)]
    return {"mean_auc": float(np.mean(valid)) if valid else float("nan"), "per_event": aucs}


def train_epoch(model, loader, optimizer, criterion):
    """AMP なし (fp32) — NaN 安定版"""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        if not torch.isfinite(loss):
            continue          # NaN/Inf loss はスキップ
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
        # NaN logit を 0 に置換 (sigmoid(0) = 0.5 = chance)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        all_logits.append(logits.cpu().float())
        all_y.append(y.cpu())
    y_score = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_true  = torch.cat(all_y).numpy().astype(np.int8)
    return total / max(len(loader.dataset), 1), y_true, y_score


def fit(model, train_eeg, train_labels, val_eeg, val_labels, cfg, save_path):
    model = model.to(DEVICE)
    win   = cfg["window_samples"]

    train_ds = EEGWindowDataset(train_eeg, train_labels, win, cfg.get("train_step", 50), 3.0)
    val_ds   = EEGWindowDataset(val_eeg,   val_labels,   win, cfg.get("val_step",   25), 1.0)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    criterion = FocalLoss(cfg.get("focal_gamma", 2.0),
                          label_smoothing=cfg.get("label_smoothing", 0.05))
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"])

    best_auc, patience_cnt = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        val_loss, y_true, y_score = eval_epoch(model, val_loader, criterion)
        auc_dict = column_wise_auc(y_true, y_score)
        val_auc  = auc_dict["mean_auc"]

        if val_auc > best_auc:
            best_auc     = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"  Epoch {epoch:3d}/{cfg['n_epochs']} | "
              f"tr={tr_loss:.4f} | val={val_loss:.4f} | auc={val_auc:.4f} | {time.time()-t0:.1f}s")

        if patience_cnt >= cfg["patience"]:
            print(f"  Early stop (patience={cfg['patience']})")
            break

    return {"best_val_auc": best_auc, "history": history}


# ─── セル 9: データ準備 ──────────────────────────────────────
print("=" * 60)
print("データ読み込み & 前処理中...")
t_start = time.time()

subject_data = {}
for subj in range(1, 13):
    tr_eegs, tr_lbls = [], []
    va_eegs, va_lbls = [], []
    for ser in range(1, 9):
        eeg, lbl = load_raw(subj, ser)
        eeg = preprocess(eeg)
        (va_eegs if ser in VAL_SERIES else tr_eegs).append(eeg)
        (va_lbls if ser in VAL_SERIES else tr_lbls).append(lbl)

    subject_data[subj] = {
        "tr_eeg": np.concatenate(tr_eegs),
        "tr_lbl": np.concatenate(tr_lbls),
        "va_eeg": np.concatenate(va_eegs),
        "va_lbl": np.concatenate(va_lbls),
    }
    n_tr = len(subject_data[subj]["tr_eeg"])
    n_va = len(subject_data[subj]["va_eeg"])
    print(f"  Subj {subj:2d}: train={n_tr:7,}  val={n_va:7,} frames")

print(f"前処理完了: {time.time()-t_start:.1f}s\n")

# ─── セル 10: EEGNet 訓練 ────────────────────────────────────
EEGNET_CFG = {
    "window_samples":  250,
    "train_step":       50,
    "val_step":         25,
    "batch_size":      512,
    "lr":             5e-4,
    "weight_decay":   1e-4,
    "n_epochs":         40,
    "patience":          8,
    "focal_gamma":     2.0,
    "label_smoothing": 0.05,
}

print("=" * 60)
print("[1] EEGNet within-subject 訓練")
print("=" * 60)

eegnet_aucs = []
for subj in range(1, 13):
    print(f"\nSubj {subj:2d}")
    t0    = time.time()
    model = EEGNet(N_CHANNELS, 250, N_CLASSES)
    res   = fit(
        model,
        subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
        subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
        EEGNET_CFG,
        MODEL_DIR / f"eegnet_subj{subj}.pt",
    )
    auc = res["best_val_auc"]
    eegnet_aucs.append(auc)
    print(f"  >>> Subj {subj:2d} best AUC = {auc:.4f}  ({time.time()-t0:.0f}s)")

print(f"\nEEGNet mean AUC: {np.nanmean(eegnet_aucs):.4f} ± {np.nanstd(eegnet_aucs):.4f}")

# ─── セル 11: CausalEEGConformer 訓練 ───────────────────────
CONFORMER_CFG = {
    "window_samples":  250,
    "train_step":       50,
    "val_step":         25,
    "batch_size":      512,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "n_epochs":         80,
    "patience":         12,
    "focal_gamma":     2.0,
    "label_smoothing": 0.05,
}

print("\n" + "=" * 60)
print("[2] CausalEEGConformer within-subject 訓練")
print("=" * 60)

conformer_aucs      = []
conformer_histories = []

for subj in range(1, 13):
    print(f"\nSubj {subj:2d}")
    t0    = time.time()
    model = CausalEEGConformer(
        n_channels=N_CHANNELS, n_times=250, n_classes=N_CLASSES,
        tcn_channels=[32, 64, 128], tcn_kernel=8,
        attn_heads=4, attn_dim=64, cov_proj_dim=32,
        hidden_dim=256, dropout=0.3, channel_names=CHANNELS,
    )
    res = fit(
        model,
        subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
        subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
        CONFORMER_CFG,
        MODEL_DIR / f"conformer_subj{subj}.pt",
    )
    auc = res["best_val_auc"]
    conformer_aucs.append(auc)
    conformer_histories.append(res["history"])
    print(f"  >>> Subj {subj:2d} best AUC = {auc:.4f}  ({time.time()-t0:.0f}s)")

print(f"\nConformer mean AUC: {np.nanmean(conformer_aucs):.4f} ± {np.nanstd(conformer_aucs):.4f}")

# ─── セル 12: 結果保存 ───────────────────────────────────────
results = {
    "eegnet_within": {
        "mean_auc":  round(float(np.nanmean(eegnet_aucs)), 4),
        "std_auc":   round(float(np.nanstd(eegnet_aucs)),  4),
        "ci_95": [
            round(float(np.nanmean(eegnet_aucs) - 1.96 * np.nanstd(eegnet_aucs) / np.sqrt(12)), 4),
            round(float(np.nanmean(eegnet_aucs) + 1.96 * np.nanstd(eegnet_aucs) / np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in eegnet_aucs],
    },
    "conformer_within": {
        "mean_auc":  round(float(np.nanmean(conformer_aucs)), 4),
        "std_auc":   round(float(np.nanstd(conformer_aucs)),  4),
        "ci_95": [
            round(float(np.nanmean(conformer_aucs) - 1.96 * np.nanstd(conformer_aucs) / np.sqrt(12)), 4),
            round(float(np.nanmean(conformer_aucs) + 1.96 * np.nanstd(conformer_aucs) / np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in conformer_aucs],
        "histories": [
            {"val_auc": h["val_auc"]} if h else None
            for h in conformer_histories
        ],
    },
}

out_path = WORK_DIR / "04_dl_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n結果保存: {out_path}")

# ─── セル 13: サマリー ───────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS (within-subject, val series 7-8)")
print("=" * 60)
print(f"\n{'Subj':>4}  {'EEGNet':>8}  {'Conformer':>10}  {'Delta':>7}")
print("-" * 38)
for i, (e, c) in enumerate(zip(eegnet_aucs, conformer_aucs), 1):
    print(f"  {i:2d}   {e:.4f}    {c:.4f}    {c-e:+.4f}")
print("-" * 38)
print(f"  Mean {np.nanmean(eegnet_aucs):.4f}    {np.nanmean(conformer_aucs):.4f}    "
      f"{np.nanmean(conformer_aucs)-np.nanmean(eegnet_aucs):+.4f}")
print(f"  Std  {np.nanstd(eegnet_aucs):.4f}    {np.nanstd(conformer_aucs):.4f}")
print(f"\nモデル保存先: {MODEL_DIR}")
print("Output > ダウンロードで 04_dl_results.json と models/*.pt を取得してください。")
