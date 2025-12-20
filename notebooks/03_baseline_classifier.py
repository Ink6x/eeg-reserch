"""
ノートブック 03: ベースライン分類器
- 帯域パワー特徴 + LogisticRegression (被験者内 / 被験者横断)
- Subject-wise LOSO CV で AUC を評価
- 結果を reports/results/ に保存
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eegdemo.font_setup import setup_japanese_font
from src.eegdemo.constants import CHANNELS, EVENTS, SAMPLING_RATE
from src.eegdemo.io import load_raw
from src.eegdemo.preprocess import preprocess
from src.eegdemo.features import extract_band_power_features
from src.eegdemo.eval import column_wise_auc, summarize_cv_results, multi_label_metrics
from src.eegdemo.viz import plot_roc_curves, plot_subject_auc_boxplot

setup_japanese_font()
RESULTS_DIR = Path("reports/results")
FIG_DIR = Path("reports/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
WINDOW_MS = 500.0
STEP_MS = 10.0   # 10ms step (全フレームは重すぎるため間引き)
VAL_SERIES = [7, 8]   # self hold-out

print("=" * 60)
print("ベースライン分類器 — 帯域パワー + LogisticRegression")
print("=" * 60)

# ──────────────────────────────────────────────
# 1. データ準備 (被験者ごとに前処理 + 特徴抽出)
# ──────────────────────────────────────────────
print("\n[1] 前処理 & 特徴抽出中 (各被験者 series 1-8)...")

subject_features: dict[int, dict[str, np.ndarray]] = {}

for subj in range(1, 13):
    t0 = time.time()
    eegs_train, labels_train = [], []
    eegs_val, labels_val = [], []

    for ser in range(1, 9):
        eeg_raw, lbl = load_raw(subj, ser, "train")
        eeg_proc = preprocess(eeg_raw, SAMPLING_RATE)
        X, centers = extract_band_power_features(eeg_proc, SAMPLING_RATE, WINDOW_MS, STEP_MS)
        y = lbl[centers]

        if ser in VAL_SERIES:
            eegs_val.append(X)
            labels_val.append(y)
        else:
            eegs_train.append(X)
            labels_train.append(y)

    subject_features[subj] = {
        "X_train": np.concatenate(eegs_train),
        "y_train": np.concatenate(labels_train),
        "X_val": np.concatenate(eegs_val),
        "y_val": np.concatenate(labels_val),
    }
    elapsed = time.time() - t0
    n_train = len(subject_features[subj]["X_train"])
    n_val = len(subject_features[subj]["X_val"])
    print(f"  Subj {subj:2d}: train={n_train:7d}窓, val={n_val:7d}窓  ({elapsed:.1f}s)")

print(f"  特徴次元: {subject_features[1]['X_train'].shape[1]}")

# ──────────────────────────────────────────────
# 2. 被験者内評価 (Within-Subject)
# ──────────────────────────────────────────────
print("\n[2] 被験者内評価 (train=Series1-6, val=Series7-8)...")

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

within_results = []

for subj in range(1, 13):
    X_tr = subject_features[subj]["X_train"]
    y_tr = subject_features[subj]["y_train"]
    X_val = subject_features[subj]["X_val"]
    y_val = subject_features[subj]["y_val"]

    base_clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MultiOutputClassifier(base_clf, n_jobs=-1)),
    ])
    clf.fit(X_tr, y_tr)

    proba_list = clf.predict_proba(X_val)
    y_score = np.stack([p[:, 1] for p in proba_list], axis=1)

    auc_dict = column_wise_auc(y_val, y_score)
    print(f"  Subj {subj:2d} | mean AUC = {auc_dict['mean_auc']:.4f}  "
          f"| HandStart={auc_dict['HandStart']:.4f}, LiftOff={auc_dict['LiftOff']:.4f}")

    within_results.append({
        "subject": subj,
        "y_val": y_val,
        "y_score": y_score,
        "roc_auc": auc_dict,
    })

within_mean_aucs = [r["roc_auc"]["mean_auc"] for r in within_results]
print(f"\n  Within-Subject mean AUC: {np.mean(within_mean_aucs):.4f} ± {np.std(within_mean_aucs):.4f}")

# ──────────────────────────────────────────────
# 3. 被験者横断評価 (Cross-Subject LOSO)
# ──────────────────────────────────────────────
print("\n[3] 被験者横断 Leave-One-Subject-Out CV...")

cross_results = []

MAX_LOSO_TRAIN = 150_000  # 速度のためランダムサンプリング (LR はデータ量 >> 品質)

for test_subj in range(1, 13):
    train_subjs = [s for s in range(1, 13) if s != test_subj]

    X_tr_all = np.concatenate([subject_features[s]["X_train"] for s in train_subjs])
    y_tr_all = np.concatenate([subject_features[s]["y_train"] for s in train_subjs])

    # サブサンプリング (クラス比率を保持)
    rng = np.random.default_rng(42)
    n = len(X_tr_all)
    if n > MAX_LOSO_TRAIN:
        idx = rng.choice(n, size=MAX_LOSO_TRAIN, replace=False)
        idx.sort()
        X_tr = X_tr_all[idx]
        y_tr = y_tr_all[idx]
    else:
        X_tr, y_tr = X_tr_all, y_tr_all

    X_te = np.concatenate([
        subject_features[test_subj]["X_train"],
        subject_features[test_subj]["X_val"],
    ])
    y_te = np.concatenate([
        subject_features[test_subj]["y_train"],
        subject_features[test_subj]["y_val"],
    ])

    base_clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MultiOutputClassifier(base_clf, n_jobs=-1)),
    ])
    clf.fit(X_tr, y_tr)

    proba_list = clf.predict_proba(X_te)
    y_score = np.stack([p[:, 1] for p in proba_list], axis=1)

    auc_dict = column_wise_auc(y_te, y_score)
    print(f"  Subj {test_subj:2d} (test) | mean AUC = {auc_dict['mean_auc']:.4f}")

    cross_results.append({
        "subject": test_subj,
        "y_val": y_te,
        "y_score": y_score,
        "roc_auc": auc_dict,
    })

cross_mean_aucs = [r["roc_auc"]["mean_auc"] for r in cross_results]
print(f"\n  Cross-Subject mean AUC: {np.mean(cross_mean_aucs):.4f} ± {np.std(cross_mean_aucs):.4f}")

# ──────────────────────────────────────────────
# 4. Within vs Cross 比較 (H4 の予備検定)
# ──────────────────────────────────────────────
print("\n[4] Within vs Cross 比較...")

from src.eegdemo.eval import cohens_d_paired, permutation_test_paired, bootstrap_ci

d = cohens_d_paired(within_mean_aucs, cross_mean_aucs)
obs, p_val = permutation_test_paired(within_mean_aucs, cross_mean_aucs)
ci_within = bootstrap_ci(np.array(within_mean_aucs))
ci_cross = bootstrap_ci(np.array(cross_mean_aucs))

print(f"  Within  AUC: {np.mean(within_mean_aucs):.4f} (95% CI: {ci_within[0]:.4f}-{ci_within[1]:.4f})")
print(f"  Cross   AUC: {np.mean(cross_mean_aucs):.4f} (95% CI: {ci_cross[0]:.4f}-{ci_cross[1]:.4f})")
print(f"  Cohen's d:   {d:.3f}")
print(f"  Permutation test: obs={obs:.4f}, p={p_val:.4f} ({'有意' if p_val < 0.05 else '非有意'})")

# ──────────────────────────────────────────────
# 5. 可視化
# ──────────────────────────────────────────────
print("\n[5] 図の生成...")

# 5a. ROC 曲線 (全被験者集約)
y_true_all = np.concatenate([r["y_val"] for r in within_results])
y_score_all = np.concatenate([r["y_score"] for r in within_results])
fig_roc = plot_roc_curves(y_true_all, y_score_all, "ROC曲線 — Within-Subject ベースライン", "03a_roc_within")
print("  保存: 03a_roc_within.png")

# 5b. 被験者別 AUC 箱ひげ図
fig_box = plot_subject_auc_boxplot(within_results, "03b_auc_boxplot_within")
print("  保存: 03b_auc_boxplot_within.png")

# 5c. Within vs Cross 比較
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(12)
ax.bar(x - 0.2, within_mean_aucs, 0.4, label=f"Within (mean={np.mean(within_mean_aucs):.3f})", color="steelblue")
ax.bar(x + 0.2, cross_mean_aucs, 0.4, label=f"Cross (mean={np.mean(cross_mean_aucs):.3f})", color="salmon")
ax.set_xticks(list(x))
ax.set_xticklabels([f"S{i+1}" for i in range(12)])
ax.set_ylabel("Mean ROC-AUC (6 events)")
ax.set_title(f"Within vs Cross-Subject AUC (d={d:.2f}, p={p_val:.3f})")
ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="chance")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
fig.savefig(FIG_DIR / "03c_within_vs_cross.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  保存: 03c_within_vs_cross.png")

# ──────────────────────────────────────────────
# 6. 結果保存
# ──────────────────────────────────────────────
results_summary = {
    "within_subject": {
        "mean_auc": round(float(np.mean(within_mean_aucs)), 4),
        "std_auc": round(float(np.std(within_mean_aucs)), 4),
        "ci_95": [round(ci_within[0], 4), round(ci_within[1], 4)],
        "per_subject": [round(v, 4) for v in within_mean_aucs],
        "per_event_mean": {
            ev: round(float(np.mean([r["roc_auc"][ev] for r in within_results])), 4)
            for ev in EVENTS
        },
    },
    "cross_subject": {
        "mean_auc": round(float(np.mean(cross_mean_aucs)), 4),
        "std_auc": round(float(np.std(cross_mean_aucs)), 4),
        "ci_95": [round(ci_cross[0], 4), round(ci_cross[1], 4)],
        "per_subject": [round(v, 4) for v in cross_mean_aucs],
    },
    "within_vs_cross": {
        "cohens_d": round(d, 3),
        "permutation_p": round(p_val, 4),
        "significant": p_val < 0.05,
    },
}

with open(RESULTS_DIR / "03_baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n  JSON保存: {RESULTS_DIR / '03_baseline_results.json'}")

print("\n" + "=" * 60)
print("ベースライン分類器 完了")
print("=" * 60)
print(f"\n  Within-Subject  mean AUC: {np.mean(within_mean_aucs):.4f}")
print(f"  Cross-Subject   mean AUC: {np.mean(cross_mean_aucs):.4f}")
print(f"  H4 検定 (within > cross): p = {p_val:.4f}, d = {d:.3f}")
