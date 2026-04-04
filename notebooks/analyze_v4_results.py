"""
v4 訓練結果の分析スクリプト (ローカル実行用)

Kaggle から `04_dl_results_v4.json` を取得した後、
このスクリプトで詳細分析を行う。

使い方:
    python notebooks/analyze_v4_results.py reports/results/04_dl_results_v4.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# v3/v2 ベンチマーク (model_iteration_log.md より)
LR_AUCS = [0.7739, 0.8050, 0.7152, 0.7870, 0.6670, 0.7436,
           0.7680, 0.7719, 0.6729, 0.7510, 0.6981, 0.7019]

V2_EEGNET = [0.7621, 0.7780, 0.7337, 0.7860, 0.7107, 0.7242,
             0.7192, 0.7569, 0.5460, 0.5732, 0.7394, 0.7608]

V2_CONFORMER = [0.6280, 0.5560, 0.5720, 0.6360, 0.6050, 0.6360,
                0.5910, 0.6230, 0.6180, 0.5520, 0.6160, 0.6640]


def load_results(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_per_subject(results: dict) -> None:
    """被験者別 AUC と LR 比較を出力"""
    eegnet = results["eegnet_v4"]["per_subject"]
    conformer = results["conformer_v4"]["per_subject"]

    print("\n" + "=" * 78)
    print("v4 Per-Subject Comparison")
    print("=" * 78)
    print(f"{'Subj':>4}  {'LR':>7}  {'v2_EEG':>8}  {'v4_EEG':>8}  "
          f"{'Δ_EEG':>7}  {'v2_Conf':>8}  {'v4_Conf':>8}  {'Δ_Conf':>7}")
    print("-" * 78)

    for i in range(12):
        lr = LR_AUCS[i]
        v2e, v4e = V2_EEGNET[i], eegnet[i]
        v2c, v4c = V2_CONFORMER[i], conformer[i]
        delta_e = v4e - lr
        delta_c = v4c - lr
        print(f"  {i+1:2d}    {lr:.4f}   {v2e:.4f}   {v4e:.4f}   {delta_e:+.3f}   "
              f"{v2c:.4f}   {v4c:.4f}   {delta_c:+.3f}")

    print("-" * 78)
    lr_mean = np.mean(LR_AUCS)
    v2e_mean, v4e_mean = np.mean(V2_EEGNET), np.mean(eegnet)
    v2c_mean, v4c_mean = np.mean(V2_CONFORMER), np.mean(conformer)
    print(f"  Mean  {lr_mean:.4f}   {v2e_mean:.4f}   {v4e_mean:.4f}   {v4e_mean-lr_mean:+.3f}   "
          f"{v2c_mean:.4f}   {v4c_mean:.4f}   {v4c_mean-lr_mean:+.3f}")


def analyze_lr_comparison(results: dict) -> None:
    """LR ベースラインを超えた被験者数"""
    eegnet = results["eegnet_v4"]["per_subject"]
    conformer = results["conformer_v4"]["per_subject"]

    print("\n" + "=" * 50)
    print("LR Baseline 超え被験者数")
    print("=" * 50)

    eegnet_wins = sum(1 for i in range(12) if eegnet[i] > LR_AUCS[i])
    conformer_wins = sum(1 for i in range(12) if conformer[i] > LR_AUCS[i])
    print(f"  EEGNet v4:    {eegnet_wins}/12 被験者で LR を超えた")
    print(f"  Conformer v4: {conformer_wins}/12 被験者で LR を超えた")

    # 大幅勝利・大幅敗北 (|Δ| > 0.05)
    print("\n大幅な改善 (Δ > +0.05):")
    for i in range(12):
        if eegnet[i] - LR_AUCS[i] > 0.05:
            print(f"  EEGNet S{i+1}: {LR_AUCS[i]:.3f} → {eegnet[i]:.3f}  "
                  f"(+{eegnet[i] - LR_AUCS[i]:.3f})")
        if conformer[i] - LR_AUCS[i] > 0.05:
            print(f"  Conformer S{i+1}: {LR_AUCS[i]:.3f} → {conformer[i]:.3f}  "
                  f"(+{conformer[i] - LR_AUCS[i]:.3f})")

    print("\n大幅な敗北 (Δ < -0.05):")
    for i in range(12):
        if eegnet[i] - LR_AUCS[i] < -0.05:
            print(f"  EEGNet S{i+1}: {LR_AUCS[i]:.3f} → {eegnet[i]:.3f}  "
                  f"({eegnet[i] - LR_AUCS[i]:+.3f})")
        if conformer[i] - LR_AUCS[i] < -0.05:
            print(f"  Conformer S{i+1}: {LR_AUCS[i]:.3f} → {conformer[i]:.3f}  "
                  f"({conformer[i] - LR_AUCS[i]:+.3f})")


def analyze_paired_test(results: dict) -> None:
    """LR vs v4 の paired permutation test"""
    print("\n" + "=" * 50)
    print("統計検定: LR vs v4 (paired permutation test)")
    print("=" * 50)

    rng = np.random.default_rng(42)

    for model_name, key in [("EEGNet v4", "eegnet_v4"), ("Conformer v4", "conformer_v4")]:
        v4_aucs = np.array(results[key]["per_subject"])
        lr_arr = np.array(LR_AUCS)
        diff = v4_aucs - lr_arr
        observed_mean = diff.mean()

        # 符号の入れ替えで permutation
        n_perm = 10000
        perm_means = np.empty(n_perm)
        for k in range(n_perm):
            signs = rng.choice([-1, 1], size=12)
            perm_means[k] = (signs * diff).mean()
        p_value = (np.abs(perm_means) >= np.abs(observed_mean)).mean()

        # Cohen's d (paired)
        d = observed_mean / max(diff.std(ddof=1), 1e-12)

        print(f"\n  {model_name}:")
        print(f"    観測差分 (mean):  {observed_mean:+.4f}")
        print(f"    Cohen's d:       {d:+.3f}")
        print(f"    permutation p:   {p_value:.4f}")
        if p_value < 0.05:
            print(f"    結論: [significant] LR ベースラインに対して有意 (p<0.05)")
        else:
            print(f"    結論: [n.s.]        LR との有意差なし (p>=0.05)")


def analyze_learning_curves(results: dict) -> None:
    """各被験者の学習曲線パターン分析"""
    print("\n" + "=" * 50)
    print("学習曲線パターン分析 (Conformer v4)")
    print("=" * 50)

    histories = results["conformer_v4"]["histories"]
    print(f"\n{'Subj':>4}  {'best_ep':>7}  {'best_auc':>9}  {'last_auc':>9}  "
          f"{'pattern':>10}")
    print("-" * 50)

    for i, h in enumerate(histories):
        if not h or "val_auc" not in h:
            continue
        aucs = h["val_auc"]
        best_ep = int(np.argmax(aucs)) + 1
        best_auc = max(aucs)
        last_auc = aucs[-1]

        # Pattern 分類
        if best_ep <= 3 and last_auc < best_auc - 0.05:
            pattern = "early-collapse"
        elif best_ep >= len(aucs) - 3:
            pattern = "still-improving"
        elif abs(last_auc - best_auc) < 0.02:
            pattern = "stable"
        else:
            pattern = "oscillating"

        print(f"  {i+1:2d}    {best_ep:>7}  {best_auc:>9.4f}  {last_auc:>9.4f}  "
              f"{pattern:>10}")


def main():
    if len(sys.argv) < 2:
        path = Path("reports/results/04_dl_results_v4.json")
    else:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: {path} not found")
        print("Kaggle から 04_dl_results_v4.json を取得して reports/results/ に置いてください")
        sys.exit(1)

    print(f"Loading: {path}")
    results = load_results(path)
    print(f"Version: {results.get('version', 'unknown')}")
    print(f"Config: window={results['config']['window_samples']}, "
          f"loss={results['config']['loss_type']}")

    analyze_per_subject(results)
    analyze_lr_comparison(results)
    analyze_paired_test(results)
    analyze_learning_curves(results)

    print("\n" + "=" * 50)
    print("Done.  Phase 2.5 のさらなる分析にはこの結果を使ってください。")


if __name__ == "__main__":
    main()
