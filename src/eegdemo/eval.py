"""
評価モジュール: subject-wise CV, AUC, 統計検定
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

from .constants import EVENTS, SAMPLING_RATE


# ──────────────────────────────────────────────
# 指標計算
# ──────────────────────────────────────────────

def column_wise_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """
    6 イベント各列の ROC-AUC を計算し、平均 AUC も返す。
    Kaggle public LB に相当する指標。
    """
    results = {}
    aucs = []
    for i, event in enumerate(EVENTS):
        yt = y_true[:, i]
        ys = y_score[:, i]
        if yt.sum() == 0:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(yt, ys))
        results[event] = auc
        aucs.append(auc)

    valid_aucs = [a for a in aucs if not np.isnan(a)]
    results["mean_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return results


def multi_label_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | dict]:
    """
    AUC + PR-AUC + Brier を一括計算。
    """
    auc_dict = column_wise_auc(y_true, y_score)
    pr_aucs = {}
    briers = {}

    for i, event in enumerate(EVENTS):
        yt = y_true[:, i]
        ys = y_score[:, i]
        if yt.sum() == 0:
            pr_aucs[event] = float("nan")
            briers[event] = float("nan")
        else:
            pr_aucs[event] = float(average_precision_score(yt, ys))
            briers[event] = float(brier_score_loss(yt, ys))

    return {
        "roc_auc": auc_dict,
        "pr_auc": pr_aucs,
        "brier": briers,
    }


# ──────────────────────────────────────────────
# Subject-wise Cross Validation
# ──────────────────────────────────────────────

def leave_one_subject_out_cv(
    eeg_by_subject: dict[int, np.ndarray],
    labels_by_subject: dict[int, np.ndarray],
    model_factory,
    fs: float = SAMPLING_RATE,
) -> list[dict]:
    """
    Leave-One-Subject-Out CV。
    各 fold で 1 被験者を test、残りを train とする。

    Parameters
    ----------
    eeg_by_subject : {subject_id: (n_frames, n_ch)}
    labels_by_subject : {subject_id: (n_frames, 6)}
    model_factory : callable() → fitted model with .predict_proba(eeg) method

    Returns
    -------
    fold_results : list of metric dicts (1 per subject)
    """
    subjects = sorted(eeg_by_subject.keys())
    fold_results = []

    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        # train データ結合
        train_eeg = np.concatenate([eeg_by_subject[s] for s in train_subjs])
        train_labels = np.concatenate([labels_by_subject[s] for s in train_subjs])

        test_eeg = eeg_by_subject[test_subj]
        test_labels = labels_by_subject[test_subj]

        # モデル訓練
        model = model_factory()
        model.fit(train_eeg, train_labels, fs=fs)

        # 予測
        y_score = model.predict_proba(test_eeg, fs=fs)

        # 評価 — フレームとウィンドウのサイズ差を補正
        n_score = len(y_score)
        test_labels_aligned = test_labels[-n_score:]

        metrics = multi_label_metrics(test_labels_aligned, y_score)
        metrics["test_subject"] = test_subj
        fold_results.append(metrics)
        print(f"Subject {test_subj:2d} | mean AUC = {metrics['roc_auc']['mean_auc']:.4f}")

    return fold_results


def summarize_cv_results(fold_results: list[dict]) -> dict:
    """fold_results を被験者平均・標準偏差で集計する"""
    mean_aucs = [r["roc_auc"]["mean_auc"] for r in fold_results]
    per_event = {ev: [] for ev in EVENTS}
    for r in fold_results:
        for ev in EVENTS:
            v = r["roc_auc"].get(ev, float("nan"))
            per_event[ev].append(v)

    return {
        "mean_auc_mean": float(np.nanmean(mean_aucs)),
        "mean_auc_std": float(np.nanstd(mean_aucs)),
        "per_event_mean": {ev: float(np.nanmean(v)) for ev, v in per_event.items()},
        "per_event_std": {ev: float(np.nanstd(v)) for ev, v in per_event.items()},
        "n_subjects": len(fold_results),
    }


# ──────────────────────────────────────────────
# 統計検定ユーティリティ
# ──────────────────────────────────────────────

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """対応ありデータの Cohen's d"""
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_iter: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """BCa に近い percentile bootstrap 信頼区間"""
    rng = np.random.default_rng(seed)
    stats = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_iter)]
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(np.quantile(stats, lo)), float(np.quantile(stats, hi))


def permutation_test_paired(
    a: np.ndarray,
    b: np.ndarray,
    n_iter: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    対応あり被験者間 permutation test。
    H0: mean(a - b) = 0
    Returns: observed_stat, p_value (両側)
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    observed = float(np.abs(diff.mean()))

    null_dist = []
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=len(diff))
        null_dist.append(np.abs((diff * signs).mean()))

    p_value = float((np.array(null_dist) >= observed).mean())
    return observed, p_value
