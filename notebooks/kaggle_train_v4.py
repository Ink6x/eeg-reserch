# ============================================================
# Kaggle Notebook v4: EEGNet v4 + CausalEEGConformer v4 (本番訓練)
# Dataset: Grasp-and-Lift EEG Detection
#
# v3/v2 からの統合改善:
#   1. EEGNet padding: 完全 causal (kernel-1, 0)
#   2. TCNBlock: BatchNorm1d (v3 修正)
#   3. CholeskyPool → LogVarPool (v3 修正)
#   4. データ拡張パイプライン (TemporalJitter, ChannelMask,
#      AmplitudeScale, GaussianNoise, Mixup)
#   5. Window 拡大 (2.2 ablation の最良値を採用)
#   6. 最良損失関数 (2.3 比較の最良値を採用)
#
# !!! 重要 !!!
# 以下のハイパーパラメータは Phase 2.2 と 2.3 の結果を反映して更新する:
#   WINDOW_SAMPLES   = 500   ← Phase 2.2 の最良値に置換
#   LOSS_TYPE        = "focal" ← Phase 2.3 の最良値に置換
#   それ以外は v3 設計に基づく既定値
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
import json, math, random, time, warnings
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
MODEL_DIR = WORK_DIR / "models_v4"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SAMPLING_RATE = 500
N_CHANNELS    = 32
N_CLASSES     = 6
VAL_SERIES    = [7, 8]

# Phase 2.2 結果反映 (2026-05-07): 750 sample (1500ms) が平均最良
# S1/S9 で大幅改善 (+0.067, +0.078), S5 のみ反証 (個人差)
WINDOW_SAMPLES = 750

# Phase 2.3 結果反映 (2026-05-07): bce_weighted が平均最良
# 仮説 A1 (focal 0.5 局所安定) は反証 → focal は実は問題なかった
# bce_weighted は pos_weight ≈ 37 (neg/pos ratio) で正例の損失寄与を強化
LOSS_TYPE = "bce_weighted"

# Augmentation パラメータ (Phase 2.0 設計に基づく)
AUG_TEMPORAL_JITTER  = 10        # ±20ms (max_jitter samples)
AUG_CHANNEL_MASK_K   = 3         # max channels to mask
AUG_CHANNEL_MASK_P   = 0.5
AUG_AMP_SCALE_RANGE  = (0.9, 1.1)
AUG_AMP_SCALE_P      = 0.5
AUG_NOISE_STD        = 0.05
AUG_NOISE_P          = 0.3
AUG_MIXUP_ALPHA      = 0.2
AUG_MIXUP_P          = 0.5

EVENTS = ["HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased"]

CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8",
    "TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8",
    "PO9","O1","Oz","O2","PO10",
]

print(f"WINDOW_SAMPLES = {WINDOW_SAMPLES} ({WINDOW_SAMPLES * 1000 / SAMPLING_RATE:.0f}ms)")
print(f"LOSS_TYPE = {LOSS_TYPE}")

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

# ─── セル 5: モデル定義 ──────────────────────────────────────

# ---- EEGNet v4 (完全 causal padding) ----
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


# ---- CausalEEGConformer v4 (= v3: BatchNorm1d + LogVarPool) ----
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, bias=False)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """v3 修正: LayerNorm(transpose) → BatchNorm1d"""
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self.proj  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class LogVarPool(nn.Module):
    """v3 設計: mean + log_var の連結 (有界スケール)"""
    def __init__(self, n_ch, proj_dim=64, eps=1e-8):
        super().__init__()
        self.proj    = nn.Linear(n_ch, proj_dim, bias=False)
        self.eps     = eps
        self.out_dim = proj_dim * 2

    def forward(self, x):
        x_p   = self.proj(x.transpose(1, 2))
        mean  = x_p.mean(dim=1)
        var   = x_p.var(dim=1, unbiased=False)
        log_v = torch.log(var + self.eps)
        return torch.cat([mean, log_v], dim=1)


class CausalEEGConformer(nn.Module):
    """v4 = v3 改善版"""
    def __init__(self, n_channels=32, n_times=500, n_classes=6,
                 tcn_channels=None, tcn_kernel=8, attn_dim=64,
                 stat_proj_dim=64, hidden_dim=256, dropout=0.2):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 64, 128]

        self.input_conv = nn.Sequential(
            CausalConv1d(n_channels, attn_dim, kernel=3),
            nn.BatchNorm1d(attn_dim),
            nn.GELU(),
        )

        layers, in_ch = [], attn_dim
        for i, out_ch in enumerate(tcn_channels):
            layers.append(TCNBlock(in_ch, out_ch, tcn_kernel, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn      = nn.Sequential(*layers)
        tcn_out       = tcn_channels[-1]

        self.stat_pool   = LogVarPool(n_ch=tcn_out, proj_dim=stat_proj_dim)
        combined_dim     = self.stat_pool.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        feat  = self.input_conv(x)
        feat  = self.tcn(feat)
        stats = self.stat_pool(feat)
        return self.classifier(stats)


# ─── セル 6: Augmentation (Kaggle 自己完結のため inline) ─────
class ChannelMask:
    def __init__(self, max_channels=3, p=0.5):
        self.max_channels, self.p = max_channels, p
    def __call__(self, x, y):
        if random.random() > self.p:
            return x, y
        n_ch = x.shape[0]
        k = random.randint(1, min(self.max_channels, n_ch))
        idx = torch.randperm(n_ch)[:k]
        x = x.clone()
        x[idx, :] = 0.0
        return x, y


class AmplitudeScale:
    def __init__(self, scale_range=(0.9, 1.1), p=0.5):
        self.lo, self.hi = scale_range
        self.p = p
    def __call__(self, x, y):
        if random.random() > self.p:
            return x, y
        scale = torch.empty(x.shape[0], 1, dtype=x.dtype).uniform_(self.lo, self.hi)
        return x * scale, y


class GaussianNoise:
    def __init__(self, std=0.05, p=0.3):
        self.std, self.p = std, p
    def __call__(self, x, y):
        if random.random() > self.p:
            return x, y
        return x + torch.randn_like(x) * self.std, y


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)
    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


def mixup_batch(X, y, alpha=0.2, p=1.0):
    if alpha <= 0 or random.random() > p:
        return X, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], lam * y + (1 - lam) * y[idx]


# ─── セル 7: Dataset (TemporalJitter 統合) ───────────────────
class JitteredEEGWindowDataset(Dataset):
    """v4: window 抽出時に TemporalJitter, post-extraction で残りの拡張"""
    def __init__(self, eeg, labels, window_samples=500, step_samples=10,
                 positive_oversample_ratio=3.0, max_jitter=0, post_augment=None):
        self.eeg    = torch.from_numpy(eeg.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.win    = window_samples
        self.max_jitter = max_jitter
        self.post_augment = post_augment
        self._n_frames = len(eeg)

        margin = max(0, max_jitter)
        all_ends = np.arange(window_samples - 1 + margin, self._n_frames - margin,
                             step_samples, dtype=np.int32)
        is_pos = labels[all_ends].any(axis=1)
        pos_ends = all_ends[is_pos]
        neg_ends = all_ends[~is_pos]
        if positive_oversample_ratio > 1 and len(pos_ends) > 0:
            pos_ends = np.tile(pos_ends, int(positive_oversample_ratio))
        self.ends = np.concatenate([neg_ends, pos_ends]).astype(np.int32)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, idx):
        end = int(self.ends[idx])
        if self.max_jitter > 0:
            jitter = random.randint(-self.max_jitter, self.max_jitter)
            end = max(self.win - 1, min(self._n_frames - 1, end + jitter))
        x = self.eeg[end - self.win + 1:end + 1].T.contiguous()
        y = self.labels[end]
        if self.post_augment is not None:
            x, y = self.post_augment(x, y)
        return x, y


# ─── セル 8: 損失関数 + 訓練ユーティリティ ──────────────────
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


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="mean"
        )


class PlainBCELoss(nn.Module):
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def build_loss(loss_type, train_labels):
    if loss_type == "focal":
        return FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.05)
    if loss_type == "bce_weighted":
        pos_count = train_labels.sum(axis=0)
        neg_count = len(train_labels) - pos_count
        return WeightedBCELoss(torch.tensor(
            np.maximum(neg_count / np.maximum(pos_count, 1), 1.0), dtype=torch.float32
        ))
    if loss_type == "bce_plain":
        return PlainBCELoss()
    raise ValueError(f"Unknown loss_type: {loss_type}")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
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
    return {"mean_auc": float(np.mean(valid)) if valid else float("nan"),
            "per_event": aucs}


def fit(model, train_eeg, train_labels, val_eeg, val_labels, cfg, save_path,
        use_augment=True, use_mixup=True):
    """v4 訓練エントリポイント — augmentation + mixup 対応"""
    model = model.to(DEVICE)
    win = cfg["window_samples"]

    # 拡張パイプライン構築
    if use_augment:
        post_augs = Compose([
            ChannelMask(AUG_CHANNEL_MASK_K, AUG_CHANNEL_MASK_P),
            AmplitudeScale(AUG_AMP_SCALE_RANGE, AUG_AMP_SCALE_P),
            GaussianNoise(AUG_NOISE_STD, AUG_NOISE_P),
        ])
        max_jitter = AUG_TEMPORAL_JITTER
    else:
        post_augs  = None
        max_jitter = 0

    train_ds = JitteredEEGWindowDataset(
        train_eeg, train_labels, win, cfg["train_step"], 3.0,
        max_jitter=max_jitter, post_augment=post_augs,
    )
    # validation には augmentation 適用しない (max_jitter=0, post_augment=None)
    val_ds = JitteredEEGWindowDataset(
        val_eeg, val_labels, win, cfg["val_step"], 1.0,
        max_jitter=0, post_augment=None,
    )

    print(f"    dataset: train={len(train_ds):,} val={len(val_ds):,} windows")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    criterion = build_loss(LOSS_TYPE, train_labels).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=cfg.get("warmup_epochs", 5),
                                      total_epochs=cfg["n_epochs"])

    best_auc, patience_cnt = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()
        # train
        model.train()
        total = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if use_mixup:
                X, y = mixup_batch(X, y, alpha=AUG_MIXUP_ALPHA, p=AUG_MIXUP_P)
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
        auc_result = column_wise_auc(y_true, y_score)
        val_auc = auc_result["mean_auc"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"    Ep{epoch:3d}/{cfg['n_epochs']} | tr={tr_loss:.4f} val={val_loss:.4f} "
              f"auc={val_auc:.4f} lr={lr_now:.2e} | {time.time()-t0:.1f}s")

        if patience_cnt >= cfg["patience"]:
            print(f"    Early stop (patience={cfg['patience']})")
            break

    return {"best_val_auc": best_auc, "history": history}


# ─── セル 9: データ準備 (12 被験者) ───────────────────────
print("=" * 60)
print(f"v4 設定: window={WINDOW_SAMPLES}, loss={LOSS_TYPE}, augmentation=ON")
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

# ─── セル 10: EEGNet v4 訓練 ────────────────────────────────
EEGNET_CFG = {
    "window_samples": WINDOW_SAMPLES,
    "train_step":     10,    # step=10 (118K 窓、データ量改善)
    "val_step":       25,
    "batch_size":     512,
    "lr":             5e-4,
    "weight_decay":   1e-4,
    "n_epochs":       50,
    "patience":       12,
    "warmup_epochs":  5,
}

print("=" * 60)
print(f"[1] EEGNet v4 (causal pad fix + augmentation + step=10)")
print("=" * 60)

eegnet_aucs, eegnet_histories = [], []
for subj in range(1, 13):
    print(f"\nSubj {subj:2d}")
    t0  = time.time()
    res = fit(EEGNet(N_CHANNELS, WINDOW_SAMPLES, N_CLASSES),
              subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
              subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
              EEGNET_CFG, MODEL_DIR / f"eegnet_v4_subj{subj}.pt",
              use_augment=True, use_mixup=True)
    eegnet_aucs.append(res["best_val_auc"])
    eegnet_histories.append(res["history"])
    print(f"  >>> Subj {subj:2d}  AUC={res['best_val_auc']:.4f}  ({time.time()-t0:.0f}s)")

print(f"\nEEGNet v4 mean AUC: {np.nanmean(eegnet_aucs):.4f} ± {np.nanstd(eegnet_aucs):.4f}")

# ─── セル 11: CausalEEGConformer v4 訓練 ────────────────────
CONFORMER_CFG = {
    "window_samples": WINDOW_SAMPLES,
    "train_step":     20,     # Conformer は重いので step=20
    "val_step":       25,
    "batch_size":     512,
    "lr":             1e-4,   # v3 と同じ保守的設定
    "weight_decay":   1e-4,
    "n_epochs":       80,
    "patience":       15,
    "warmup_epochs":  10,
}

print("\n" + "=" * 60)
print("[2] CausalEEGConformer v4 (BatchNorm1d + LogVarPool + augmentation)")
print("=" * 60)

conformer_aucs, conformer_histories = [], []
for subj in range(1, 13):
    print(f"\nSubj {subj:2d}")
    t0    = time.time()
    model = CausalEEGConformer(
        n_channels=N_CHANNELS, n_times=WINDOW_SAMPLES, n_classes=N_CLASSES,
        tcn_channels=[64, 64, 128], tcn_kernel=8, attn_dim=64,
        stat_proj_dim=64, hidden_dim=256, dropout=0.2,
    )
    res = fit(model,
              subject_data[subj]["tr_eeg"], subject_data[subj]["tr_lbl"],
              subject_data[subj]["va_eeg"], subject_data[subj]["va_lbl"],
              CONFORMER_CFG, MODEL_DIR / f"conformer_v4_subj{subj}.pt",
              use_augment=True, use_mixup=True)
    conformer_aucs.append(res["best_val_auc"])
    conformer_histories.append(res["history"])
    print(f"  >>> Subj {subj:2d}  AUC={res['best_val_auc']:.4f}  ({time.time()-t0:.0f}s)")

print(f"\nConformer v4 mean AUC: {np.nanmean(conformer_aucs):.4f} ± {np.nanstd(conformer_aucs):.4f}")

# ─── セル 12: 結果保存 ───────────────────────────────────────
lr_aucs = [0.7739,0.8050,0.7152,0.7870,0.6670,0.7436,0.7680,0.7719,0.6729,0.7510,0.6981,0.7019]

results = {
    "version": "v4",
    "config": {
        "window_samples": WINDOW_SAMPLES,
        "loss_type":      LOSS_TYPE,
        "augmentation": {
            "temporal_jitter": AUG_TEMPORAL_JITTER,
            "channel_mask_k": AUG_CHANNEL_MASK_K,
            "channel_mask_p": AUG_CHANNEL_MASK_P,
            "amp_scale": list(AUG_AMP_SCALE_RANGE),
            "amp_p":     AUG_AMP_SCALE_P,
            "noise_std": AUG_NOISE_STD,
            "noise_p":   AUG_NOISE_P,
            "mixup_alpha": AUG_MIXUP_ALPHA,
            "mixup_p":     AUG_MIXUP_P,
        },
        "eegnet_cfg":    EEGNET_CFG,
        "conformer_cfg": CONFORMER_CFG,
    },
    "lr_baseline": {"mean_auc": round(np.mean(lr_aucs), 4), "per_subject": lr_aucs},
    "eegnet_v4": {
        "mean_auc": round(float(np.nanmean(eegnet_aucs)), 4),
        "std_auc":  round(float(np.nanstd(eegnet_aucs)), 4),
        "ci_95": [
            round(float(np.nanmean(eegnet_aucs) - 1.96 * np.nanstd(eegnet_aucs) / np.sqrt(12)), 4),
            round(float(np.nanmean(eegnet_aucs) + 1.96 * np.nanstd(eegnet_aucs) / np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in eegnet_aucs],
        "histories":   [{"val_auc": h["val_auc"]} for h in eegnet_histories],
    },
    "conformer_v4": {
        "mean_auc": round(float(np.nanmean(conformer_aucs)), 4),
        "std_auc":  round(float(np.nanstd(conformer_aucs)), 4),
        "ci_95": [
            round(float(np.nanmean(conformer_aucs) - 1.96 * np.nanstd(conformer_aucs) / np.sqrt(12)), 4),
            round(float(np.nanmean(conformer_aucs) + 1.96 * np.nanstd(conformer_aucs) / np.sqrt(12)), 4),
        ],
        "per_subject": [round(v, 4) for v in conformer_aucs],
        "histories":   [{"val_auc": h["val_auc"]} for h in conformer_histories],
    },
}

out_path = WORK_DIR / "04_dl_results_v4.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n結果保存: {out_path}")

# ─── セル 13: サマリー表示 ───────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS v4  (within-subject, val series 7-8 holdout)")
print("=" * 60)

print(f"\n{'Subj':>4}  {'LR':>7}  {'EEGNet_v4':>10}  {'Δ(LR)':>7}  "
      f"{'Conf_v4':>8}  {'Δ(LR)':>7}")
print("-" * 56)
for i in range(12):
    e4, c4, lr = eegnet_aucs[i], conformer_aucs[i], lr_aucs[i]
    print(f"  {i+1:2d}   {lr:.4f}   {e4:.4f}   {e4-lr:+.3f}   "
          f"{c4:.4f}   {c4-lr:+.3f}")

print("-" * 56)
print(f"  Mean {np.mean(lr_aucs):.4f}   {np.nanmean(eegnet_aucs):.4f}   "
      f"{np.nanmean(eegnet_aucs)-np.mean(lr_aucs):+.3f}   "
      f"{np.nanmean(conformer_aucs):.4f}   "
      f"{np.nanmean(conformer_aucs)-np.mean(lr_aucs):+.3f}")

print(f"\nモデル保存先: {MODEL_DIR}")
print("Output > ダウンロードで 04_dl_results_v4.json と models_v4/*.pt を取得してください。")
