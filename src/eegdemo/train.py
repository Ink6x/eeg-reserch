"""
DL モデル訓練ループ (因果制約 + AMD GPU DirectML 対応)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# AMD GPU: torch-directml で DML デバイスを使用
try:
    import torch_directml
    _DML_AVAILABLE = True
except ImportError:
    _DML_AVAILABLE = False


def get_device(preferred: str = "dml") -> torch.device:
    if preferred == "dml" and _DML_AVAILABLE and torch_directml.device_count() > 0:
        return torch_directml.device()
    return torch.device("cpu")


# ──────────────────────────────────────────────
# データセット
# ──────────────────────────────────────────────

class EEGWindowDataset(Dataset):
    """
    Lazy スライディングウィンドウ EEG データセット。
    全ウィンドウを事前確保せず、__getitem__ 時に切り出す。
    メモリ使用量: O(n_frames) のみ。
    """

    def __init__(
        self,
        eeg: np.ndarray,
        labels: np.ndarray,
        window_samples: int = 250,
        step_samples: int = 10,
        positive_oversample_ratio: float = 3.0,
    ) -> None:
        # EEG を共有メモリとして保持 (コピーしない)
        self.eeg = torch.from_numpy(eeg.astype(np.float32))       # (n_frames, n_ch)
        self.labels = torch.from_numpy(labels.astype(np.float32)) # (n_frames, 6)
        self.win = window_samples

        # ウィンドウの末尾フレームインデックス一覧
        n_frames = len(eeg)
        all_ends = np.arange(window_samples - 1, n_frames, step_samples, dtype=np.int32)

        # 正例・陰性例のインデックスを分離
        all_labels_at_end = labels[all_ends]  # (n_windows, 6)
        is_positive = all_labels_at_end.any(axis=1)  # (n_windows,)

        pos_ends = all_ends[is_positive]
        neg_ends = all_ends[~is_positive]

        # 正例をオーバーサンプリング
        if positive_oversample_ratio > 1 and len(pos_ends) > 0:
            n_repeat = int(positive_oversample_ratio)
            pos_ends = np.tile(pos_ends, n_repeat)

        self.ends = np.concatenate([neg_ends, pos_ends]).astype(np.int32)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.ends[idx])
        start = end - self.win + 1
        # (n_ch, window_samples) に変換 (Conv1d 入力形式)
        x = self.eeg[start: end + 1].T.contiguous()  # (n_ch, win)
        y = self.labels[end]                           # (6,)
        return x, y


# ──────────────────────────────────────────────
# Focal Loss (不均衡対策)
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-label focal loss"""

    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets >= 0.5, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ──────────────────────────────────────────────
# 訓練ループ
# ──────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """loss, y_true (n, 6), y_score (n, 6) を返す"""
    model.eval()
    total_loss = 0.0
    all_logits, all_y = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(X)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    y_score = torch.sigmoid(torch.cat(all_logits)).numpy()
    y_true = torch.cat(all_y).numpy().astype(np.int8)
    return total_loss / len(loader.dataset), y_true, y_score


def fit(
    model: nn.Module,
    train_eeg: np.ndarray,
    train_labels: np.ndarray,
    val_eeg: np.ndarray | None,
    val_labels: np.ndarray | None,
    cfg: dict,
    save_path: Path | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    モデル訓練のメインエントリポイント。

    Parameters
    ----------
    cfg : model.yaml の causal_eeg_conformer セクションに対応する dict
    save_path : ベストモデルの保存先
    """
    if device is None:
        device = get_device(cfg.get("device", "dml"))

    model = model.to(device)

    train_ds = EEGWindowDataset(
        train_eeg, train_labels,
        window_samples=cfg.get("window_samples", 250),
        step_samples=cfg.get("train_step_samples", 10),
        positive_oversample_ratio=3.0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 512),
        shuffle=True, num_workers=0, pin_memory=False,
    )

    val_loader = None
    if val_eeg is not None and val_labels is not None:
        val_ds = EEGWindowDataset(
            val_eeg, val_labels,
            window_samples=cfg.get("window_samples", 250),
            step_samples=cfg.get("val_step_samples", 25),
            positive_oversample_ratio=1.0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.get("batch_size", 512) * 2,
            shuffle=False, num_workers=0,
        )

    criterion = FocalLoss(
        gamma=cfg.get("focal_gamma", 2.0),
        label_smoothing=cfg.get("label_smoothing", 0.05),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("n_epochs", 100),
    )

    best_val_auc = -1.0
    patience = cfg.get("patience", 15)
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, cfg.get("n_epochs", 100) + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_auc_mean = float("nan")
        val_loss = float("nan")

        if val_loader is not None:
            from .eval import column_wise_auc
            val_loss, y_true, y_score = eval_epoch(model, val_loader, criterion, device)
            auc_dict = column_wise_auc(y_true, y_score)
            val_auc_mean = auc_dict["mean_auc"]

            if val_auc_mean > best_val_auc:
                best_val_auc = val_auc_mean
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.get('n_epochs', 100)} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc_mean:.4f} | {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc_mean)

        if patience_counter >= patience:
            print(f"Early stopping (patience={patience})")
            break

    return {"history": history, "best_val_auc": best_val_auc}
