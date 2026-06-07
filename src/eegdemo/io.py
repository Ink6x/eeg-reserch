"""
データ読み込みモジュール
因果制約: test データ予測時は絶対に未来フレームを参照しない
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CACHE_DIR, CHANNELS, EVENTS, TEST_DIR, TRAIN_DIR


def _parse_filename(path: Path) -> tuple[int, int, str]:
    """ファイル名から subject番号, series番号, 種別を取得"""
    m = re.match(r"subj(\d+)_series(\d+)_(data|events)", path.stem)
    if not m:
        raise ValueError(f"予期しないファイル名: {path.name}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def load_raw(
    subject: int, series: int, split: str = "train"
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    1 ファイルペアを読み込む。

    Returns
    -------
    eeg : ndarray, shape (n_frames, 32)  float64, ADCカウント値
    labels : ndarray or None, shape (n_frames, 6)  int8  (test は None)
    """
    base = TRAIN_DIR if split == "train" else TEST_DIR
    data_path = base / f"subj{subject}_series{series}_data.csv"
    events_path = base / f"subj{subject}_series{series}_events.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")

    data_df = pd.read_csv(data_path, index_col="id", dtype={ch: np.float64 for ch in CHANNELS})
    eeg = data_df[CHANNELS].values  # (n_frames, 32)

    labels = None
    if events_path.exists():
        events_df = pd.read_csv(events_path, index_col="id", dtype={ev: np.int8 for ev in EVENTS})
        labels = events_df[EVENTS].values  # (n_frames, 6)

    return eeg, labels


def load_subject_series(
    subject: int, series_list: list[int], split: str = "train"
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    複数 series を時系列順に結合する。
    series 境界は事前に記録しておく (必要なら分割に使う)。
    """
    eegs, labels_list = [], []
    for s in sorted(series_list):
        eeg, lbl = load_raw(subject, s, split)
        eegs.append(eeg)
        if lbl is not None:
            labels_list.append(lbl)

    eeg_concat = np.concatenate(eegs, axis=0)
    labels_concat = np.concatenate(labels_list, axis=0) if labels_list else None
    return eeg_concat, labels_concat


def load_all_train(val_series: list[int] | None = None) -> dict:
    """
    全 train データを読み込み、辞書で返す。

    Returns
    -------
    dict with keys: "eeg", "labels", "subject_ids", "series_ids", "is_val"
    各要素は frame 単位で対応している
    """
    eeg_all, labels_all, subj_ids, ser_ids = [], [], [], []

    for subj in range(1, 13):
        for ser in range(1, 9):
            eeg, lbl = load_raw(subj, ser, "train")
            n = len(eeg)
            eeg_all.append(eeg)
            labels_all.append(lbl)
            subj_ids.extend([subj] * n)
            ser_ids.extend([ser] * n)

    eeg_concat = np.concatenate(eeg_all, axis=0).astype(np.float32)
    labels_concat = np.concatenate(labels_all, axis=0)
    subj_arr = np.array(subj_ids, dtype=np.int8)
    ser_arr = np.array(ser_ids, dtype=np.int8)

    is_val = np.zeros(len(eeg_concat), dtype=bool)
    if val_series:
        for s in val_series:
            is_val |= (ser_arr == s)

    return {
        "eeg": eeg_concat,
        "labels": labels_concat,
        "subject_ids": subj_arr,
        "series_ids": ser_arr,
        "is_val": is_val,
    }


def get_trial_boundaries(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    HandStart (列 0) の立ち上がりエッジから trial 境界を推定する。
    Returns list of (start_frame, end_frame) for each trial.
    """
    handstart = labels[:, 0]
    # 0→1 の遷移を検出
    edges = np.where(np.diff(handstart.astype(int)) == 1)[0] + 1
    boundaries = []
    for i, start in enumerate(edges):
        end = edges[i + 1] if i + 1 < len(edges) else len(labels)
        boundaries.append((int(start), int(end)))
    return boundaries


def cache_exists(name: str) -> bool:
    return (CACHE_DIR / f"{name}.npz").exists()


def save_cache(name: str, **arrays: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_DIR / f"{name}.npz", **arrays)


def load_cache(name: str) -> dict[str, np.ndarray]:
    data = np.load(CACHE_DIR / f"{name}.npz")
    return {k: data[k] for k in data.files}
