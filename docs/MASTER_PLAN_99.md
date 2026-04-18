# EEG-99: Grasp-and-Lift 99% AUC Master Plan

**プロジェクト名**: `eeg-99` (competition-grade EEG decoding)
**作成日**: 2026-05-14
**目標**: Kaggle Grasp-and-Lift Detection で AUC **0.99** 到達
**現状**: v4 EEGNet 0.7335 / Conformer v4 0.5532（受賞水準は 0.98+）
**ギャップ**: +0.26 AUC（単一モデルでは不可能、 競技グレード ensemble が必須）
**期間**: 17 週 (~4 ヶ月)
**リソース**: paid GPU（Kaggle paid $10/月 または Colab Pro+ $50/月）

---

## 0. 哲学の転換

### v1-v4 までの考え方（捨てる）
- 単一モデルを丁寧にチューニングして性能を出す
- アーキテクチャの優劣で勝負
- 1 モデル = 1 結果

### v5 (eeg-99) の考え方（採用）
- **Diversity が王様**: 多様なモデルの組合せで境界を埋める
- **手作り特徴 + DL を併用**: 受賞者全員が手作り特徴を使った
- **被験者ごと、 イベントごとに最適化**: 単一 universal model は不可能
- **Ensemble + Stacking + Calibration + TTA**: 競技グレードの 4 段重ね
- **失敗を許容する設計**: 100 model 訓練、 上位 30-50 を選抜

### 設計原則
1. **因果性は絶対**: 全モデル causal padding、 未来漏れゼロ
2. **被験者ヘテロ性を構造的に扱う**: subject embedding を全モデルに
3. **モジュラー設計**: 各 component が独立に開発・評価可能
4. **再現性**: seed 固定、 設定 YAML 化、 model registry
5. **ポートフォリオ的明朗性**: 5 つの "named original algorithms" を作る

---

## 1. アーキテクチャの全体像

### 1.1 4-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 0: 因果前処理                                          │
│ - causal HP/LP filter (0.5-45 Hz)                          │
│ - CAR re-reference                                          │
│ - causal running z-score                                    │
│ - filterbank decomposition (5 帯域)                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Multi-Stream Feature Extraction (parallel)         │
│ ┌──────────────────┬──────────────────┬───────────────────┐│
│ │ A: 古典特徴 stream│ B: DL stream     │ C: SSL stream     ││
│ │ - FBCSP          │ - MultiScaleNet  │ - BENDR-style     ││
│ │ - Riemannian Cov │ - SubjAdaptNet   │   pre-trained     ││
│ │ - Time-domain    │ - CausalConformer│   encoder         ││
│ │ - Band power     │ - TCN ensemble   │                   ││
│ └──────────────────┴──────────────────┴───────────────────┘│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Base Model Pool (50-100 models)                    │
│ - 5 architecture × 4 window × 5 seed = 100 candidates       │
│ - diversity-based selection → top 30-50                     │
│ - per-event specialization (6 events × top arch)            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Ensemble Strategy                                   │
│ - Stage 1: Geometric mean of base models (per event)        │
│ - Stage 2: Stacking meta-learner (LightGBM)                 │
│   on 5-fold OOF predictions                                  │
│ - Stage 3: Per-subject Platt scaling calibration            │
│ - Stage 4: Test-time augmentation (5 aug × average)         │
└─────────────────────────────────────────────────────────────┘
                          ↓
                   Final Prediction
                  Target: AUC ≥ 0.99
```

---

## 2. 10 のコンポーネント設計

各コンポーネントは独立開発可能、 失敗しても他から得るものあり。

### Component 1: Multi-Scale Causal CNN
**目的**: 異なる時間スケールの情報を同時抽出

```python
class MultiScaleCausalNet(nn.Module):
    def __init__(self):
        self.branches = [
            CausalBranch(window=125),   # 250ms — 即時応答
            CausalBranch(window=250),   # 500ms — 後期 RP
            CausalBranch(window=500),   # 1.0s  — 中期文脈
            CausalBranch(window=1000),  # 2.0s  — 早期 RP
            CausalBranch(window=2000),  # 4.0s  — 長期文脈
        ]
        self.fusion = CrossScaleAttention()
```

**期待寄与**: 単独で +0.05-0.08 AUC

### Component 2: Filterbank Common Spatial Pattern (FBCSP)
**目的**: 古典 BCI 手法の DL 統合

5-6 帯域に分解、 各帯域で CSP spatial filter 学習、 log-variance を特徴に。

参考: Ang et al. (2008) "Filter Bank Common Spatial Pattern"

**期待寄与**: hybrid model で +0.03-0.05 AUC

### Component 3: Causal Riemannian Tangent Space Network
**目的**: SPD 多様体上の幾何学的特徴

```python
class CausalRiemannianBlock(nn.Module):
    """causal な共分散計算 → tangent space 投影 → MLP"""
```

参考: Barachant et al. (2013) "Classification of covariance matrices using a Riemannian-based kernel"

**期待寄与**: hybrid model で +0.02-0.04 AUC

### Component 4: Subject-Adaptive FiLM Encoding（独自アルゴリズム #1）
**目的**: 被験者ヘテロ性を全層で吸収

```python
class SubjectAdaptiveFiLM(nn.Module):
    """各 BatchNorm の γ, β を subject embedding で modulate"""
    def __init__(self, n_subjects=12, embed_dim=16):
        self.subject_emb = nn.Embedding(n_subjects, embed_dim)
        self.gamma_net = nn.Linear(embed_dim, n_features)
        self.beta_net = nn.Linear(embed_dim, n_features)
```

参考: Perez et al. (2018) "FiLM"、 我々の独自適用

**期待寄与**: 全 model で +0.04-0.06 AUC

### Component 5: Hierarchical Causal Conformer（独自アルゴリズム #2）
**目的**: v4 で崩壊した Conformer を multi-resolution attention で救済

```python
class HierarchicalCausalConformer(nn.Module):
    """3 階層の attention: short/medium/long
    各階層で causal mask、 異なる解像度で操作"""
```

**期待寄与**: 単独で 0.80-0.85

### Component 6: BENDR-Style SSL Pre-training（独自アルゴリズム #3）
**目的**: ラベルなしデータ活用

```python
class CausalMaskedEEGEncoder(nn.Module):
    """50% time masking, MAE 風復元 + contrastive auxiliary"""
```

訓練データ: 全 12 被験者 × 全 10 series（test 含む、 ラベル使わず）
参考: Kostas et al. (2021) "BENDR"、 Yang et al. (2023) "BIOT"

**期待寄与**: fine-tune で +0.03-0.06 AUC

### Component 7: Event-Order Consistency Loss（独自アルゴリズム #4）
**目的**: 6 イベントの固定順序を auxiliary supervision に

```python
class EventOrderLoss(nn.Module):
    """HandStart → FirstDigitTouch → ... の順序違反にペナルティ"""
    def forward(self, predictions, targets):
        # 予測確率の順序関係を制約
```

**期待寄与**: +0.02-0.03 AUC、 特に LiftOff/Replace で

### Component 8: Massive Bagging Ensemble
**目的**: 多様性で精度の天井を破る

| 軸 | バリエーション | 数 |
|---|---|---|
| Architecture | 5 (EEGNet+, Conformer, TCN, Hybrid, SSL-FT) | 5 |
| Window | 4 (250, 500, 1000, 2000) | 4 |
| Seed | 5 | 5 |
| **計** | | **100** |

OOF AUC で上位 30-50 を選抜（diversity-based, e.g. greedy forward selection）

**期待寄与**: +0.05-0.10 AUC over best single

### Component 9: Stacking Meta-Learner
**目的**: ensemble の最終統合

- 5-fold subject-stratified CV
- OOF predictions を入力に LightGBM meta-learner
- per-event meta-model

**期待寄与**: +0.02-0.04 AUC over geometric mean

### Component 10: Bayesian Per-Subject Calibration（独自アルゴリズム #5）
**目的**: 被験者ごとの予測確率のずれを補正

```python
class HierarchicalPlattScaling(nn.Module):
    """Platt scaling with hierarchical Bayes prior across subjects"""
```

参考: Niculescu-Mizil & Caruana (2005) Platt scaling、 hierarchical model は独自

**期待寄与**: +0.01-0.02 AUC + 較正性改善

### Component Test-Time Augmentation (TTA)
**目的**: 推論の robustness

5 augmentation × 平均 → 各サンプル 5× 推論
- temporal jitter
- channel mask (control)
- amplitude scale
- noise injection
- mixup-test

**期待寄与**: +0.005-0.015 AUC

---

## 3. 5 つの独自アルゴリズム（ポートフォリオの目玉）

GitHub README で強調する **named original contributions**:

1. **SubjectAdaptiveFiLM** — 被験者条件付き全層 modulation
2. **HierarchicalCausalConformer** — 多解像度因果 attention
3. **CausalBENDR** — 因果 SSL pre-training for EEG (BENDR の causal 派生)
4. **EventOrderConsistency Loss** — 順序 prior を auxiliary loss に
5. **HierarchicalBayesianCalibration** — 被験者間階層 prior 付き Platt scaling

各アルゴリズムは:
- 別ファイルで実装
- 数式 + 設計理由を `docs/technical_notes/algorithm_NAME.md` に
- 単独 ablation で寄与定量化

---

## 4. Phase 計画（17 週）

### Phase 0: 基盤構築 + 文献集中レビュー（1 週、 W1）

- 新 codebase `src/eeg99/` セットアップ
- 受賞者ソリューションのコード解読（GitHub に Cat & Dog 公開あり）
- 必須文献 15 本精読:
  - Cat & Dog の Kaggle writeup
  - BENDR, BIOT, BIOT-2
  - FBCSP, Riemannian BCI
  - FiLM, AdaIN
  - 競技 ML サーベイ (Pavlyshenko 2018)
- master plan refinement
- **成果物**: `docs/literature_review_master.md`

### Phase 1: 因果前処理 + データパイプライン v2（1 週、 W2）

- 因果 filterbank（5 帯域並列）
- multi-scale window dataset
- 効率的キャッシュ（preprocessed parquet）
- subject_id 統合
- **成果物**: `src/eeg99/data/`, テストカバレッジ 90%+

### Phase 2: 古典特徴 + DL ベースの単一モデル群（3 週、 W3-W5）

#### Week 3: 古典特徴 (Component 2, 3)
- FBCSP 実装 + 評価
- Riemannian tangent space network
- Baseline: 単独 LR > 0.74、 EEGNet > 0.74 を上回ること

#### Week 4: DL 単一モデル群 (Component 1, 5)
- MultiScaleCausalNet
- HierarchicalCausalConformer (Component 5)
- SubjectAdaptiveFiLM (Component 4) を両方に統合
- 目標: 各単一モデルで AUC > 0.80

#### Week 5: Hybrid model (古典 + DL fusion)
- Concat features → MLP head
- 目標: AUC > 0.85

**成果物**: 5 種類のアーキテクチャ実装 + 単一モデル benchmark

### Phase 3: SSL Pre-training (Component 6)（2 週、 W6-W7）

- BENDR-style masked EEG modeling
- 全 12 subj × 10 series 訓練（test ラベルなし区間も）
- contrastive auxiliary loss
- Linear probing で SSL 表現の質を評価
- Fine-tune → 単一モデルで AUC +0.03 向上目標

**成果物**: 事前学習済 encoder + fine-tune パイプライン

### Phase 4: Bagging Ensemble 構築 (Component 8)（3 週、 W8-W10）

#### Week 8: Bagging 訓練インフラ
- 5 architecture × 4 window × 5 seed = 100 model training script
- model registry, OOF predictions 保存
- Kaggle paid GPU で並列実行

#### Week 9: 100 model 訓練（GPU 重作業）
- 自動化スクリプト走らせる
- 失敗監視
- 推定 GPU 60-80 時間

#### Week 10: Diversity-based selection
- OOF AUC ranking
- pairwise correlation 計算
- greedy forward selection で top 30-50

**成果物**: model pool + selection 済 ensemble の base predictions

### Phase 5: Stacking + Per-Event Models + Order Loss（2 週、 W11-W12）

#### Week 11: Stacking meta-learner (Component 9)
- 5-fold subject-stratified CV
- LightGBM meta-learner per event
- 目標: ensemble AUC > 0.92

#### Week 12: Event-specific + Order loss (Component 7)
- 6 events × top-3 arch を per-event 訓練
- Event-Order Consistency Loss 統合
- 目標: AUC > 0.94

**成果物**: stacking + per-event models

### Phase 6: Calibration + TTA + Final Polish（2 週、 W13-W14）

- HierarchicalBayesianCalibration (Component 10)
- TTA pipeline
- 全要素統合最終 inference
- 目標: AUC > 0.97

**成果物**: 最終 inference pipeline + final results

### Phase 7: Stretch — 99% 到達努力（2 週、 W15-W16）

時間が許す限り:
- 追加 model architecture 試験
- hyperparameter Bayesian optimization
- 特徴量の追加発掘
- 失敗モード分析
- Per-subject hyperparameter tuning

### Phase 8: ポートフォリオ整備（1 週、 W17）

- 統計検定 (paired permutation, Cohen's d, 多重比較補正)
- 全可視化 (学習曲線、 AUC 進化グラフ、 topomap、 t-SNE)
- README 完全版
- ケーススタディドキュメント
- 動画デモ（オプション）
- 過剰主張なしの "limitations" 章

---

## 5. 新 Codebase 構造

```
src/eeg99/
├── __init__.py
├── data/
│   ├── loader.py          # 高速 IO
│   ├── preprocess.py      # 因果前処理 v2
│   ├── augmentation.py    # 既存をリファクタ
│   ├── filterbank.py      # 5 帯域分解
│   ├── dataset.py         # multi-subject, multi-window
│   └── cache.py           # parquet キャッシュ
├── features/
│   ├── fbcsp.py           # Filter Bank CSP
│   ├── riemannian.py      # Riemannian tangent space
│   ├── time_domain.py     # 統計特徴
│   └── band_power.py      # マルチ帯域 power
├── models/
│   ├── base.py            # 共通インターフェース
│   ├── eegnet_plus.py     # MultiScale EEGNet
│   ├── conformer_hier.py  # HierarchicalCausalConformer
│   ├── tcn_multi.py       # Multi-scale TCN
│   ├── hybrid.py          # 古典 + DL fusion
│   ├── ssl_bendr.py       # SSL pre-train
│   └── adapter.py         # SubjectAdaptiveFiLM
├── losses/
│   ├── focal.py
│   ├── bce_weighted.py
│   ├── event_order.py     # Component 7
│   └── ssl_losses.py
├── ensemble/
│   ├── bagging.py
│   ├── stacking.py
│   ├── selection.py       # diversity-based selection
│   ├── tta.py
│   └── calibration.py     # HierarchicalBayesianCalibration
├── training/
│   ├── trainer.py
│   ├── schedulers.py
│   ├── optimizers.py
│   └── metrics.py
├── pipeline/
│   ├── stage1_preprocess.py
│   ├── stage2_ssl.py
│   ├── stage3_supervised.py
│   ├── stage4_bagging.py
│   ├── stage5_stacking.py
│   └── stage6_inference.py
├── utils/
│   ├── config.py
│   ├── registry.py        # model registry
│   ├── seed.py
│   └── logging.py
└── configs/
    ├── default.yaml
    ├── eegnet_plus.yaml
    ├── conformer_hier.yaml
    └── ensemble.yaml

tests/
├── test_data.py
├── test_features.py
├── test_models.py
├── test_losses.py
├── test_ensemble.py
└── test_pipeline.py

notebooks/eeg99/
├── 00_setup_verify.py
├── 01_data_explore.py
├── 02_baseline_repro.py     # v4 を eeg99 で再現
├── 03_train_singles.py
├── 04_train_bagging.py
├── 05_train_stacking.py
└── 06_final_inference.py

docs/eeg99/
├── algorithm_subject_adaptive_film.md
├── algorithm_hierarchical_conformer.md
├── algorithm_causal_bendr.md
├── algorithm_event_order.md
└── algorithm_bayesian_calibration.md
```

**既存の `src/eegdemo/` は残す**: ベースライン参照、 v1-v4 の結果を捨てない。

---

## 6. リソース計画

### GPU 時間予算

| Phase | GPU 時間 (T4 換算) | 累積 |
|---|:---:|:---:|
| Phase 0 (review) | 0 | 0 |
| Phase 1 (data) | 5 | 5 |
| Phase 2 (singles) | 30 | 35 |
| Phase 3 (SSL) | 40 | 75 |
| Phase 4 (bagging) | 80 | 155 |
| Phase 5 (stacking) | 30 | 185 |
| Phase 6 (calibration) | 15 | 200 |
| Phase 7 (stretch) | 30 | 230 |
| **合計** | | **~230h** |

### GPU 調達計画

| プラン | 速度 | 月額 | 必要月数 | 総コスト |
|---|:---:|:---:|:---:|:---:|
| Kaggle 無料 (30h/週) | T4 | $0 | 2 ヶ月 | $0 |
| **Kaggle paid (60h/週)** | T4 | $10 | 1 ヶ月 | **$10** |
| Colab Pro | T4 + A100 partial | $10 | 1 ヶ月 | $10 |
| Colab Pro+ | A100 unlimited | $50 | 0.5-1 ヶ月 | $25-50 |
| RunPod (on-demand) | A100/RTX 4090 | per-hour | — | ~$50 (230h × $0.22) |

**推奨**: Kaggle paid + Colab Pro 組合せ（合計 $20-30）

---

## 7. 成功基準（Phase ごと）

| Phase | 目標 AUC | 失敗時の対応 |
|:---:|:---:|---|
| Phase 2 (singles) | 0.80-0.85 | 単一モデルで届かないなら Phase 3 (SSL) で補強 |
| Phase 3 (SSL) | 0.83-0.87 | SSL 不発なら Phase 4 ensemble に注力 |
| Phase 4 (bagging) | 0.90-0.93 | bagging で 0.88 未満なら base model 多様化不足 |
| Phase 5 (stacking) | 0.93-0.96 | stacking が機能しないなら geometric mean に戻す |
| Phase 6 (calibration) | 0.95-0.97 | calibration が効かないなら ensemble の問題 |
| Phase 7 (stretch) | **0.97-0.99** | 0.99 届かなくても 0.95+ なら成功 |

### 中間 mile-stone (各 Phase の評価基準が明確)

各 Phase 終了時に:
1. `reports/eeg99/phase_N_results.md` を作成
2. 数値ベンチマーク
3. 仮説検証ステータス
4. 次 Phase への引継ぎ

---

## 8. リスク管理

| リスク | 影響 | 確率 | 緩和策 |
|---|:---:|:---:|---|
| 0.99 に届かない | HIGH | HIGH | 0.95-0.97 でも成功と定義 |
| GPU 予算超過 | MED | MED | Phase 7 (stretch) を縮減、 Phase 4 を絞る |
| SSL 不発 (Phase 3) | MED | MED | Skip し supervised 集中 |
| Conformer 救済失敗 | LOW | MED | EEGNet+ と TCN で補う |
| Bagging で diversity 不足 | HIGH | LOW | architecture を 5 → 8 に増やす |
| 時間超過 (17 週 → 25 週) | MED | MED | Phase 7 削減、 各 Phase の "MVP" 定義 |

---

## 9. ポートフォリオ的物語

### GitHub README 構造（将来案）

```markdown
# Grasp-and-Lift EEG: From 0.57 to 0.97 in 8 Iterations

This repository documents a 4-month journey to compete with the
2015 Kaggle Grasp-and-Lift winners.

## Results

| Iteration | Date | Approach | AUC |
|:--:|:--:|:--|:--:|
| v1 | 2026-04 | EEGNet baseline | 0.668 |
| v2 | 2026-04 | RiemannCovPool experiment | 0.703 |
| v3 | 2026-05 | (skipped) | — |
| v4 | 2026-05 | Data-centric (window+aug+loss) | 0.734 |
| v5 | 2026-05 | Subject embedding (FiLM) | 0.80 |
| v6 | 2026-06 | Multi-scale + Hybrid | 0.88 |
| v7 | 2026-07 | SSL pre-training | 0.91 |
| v8 | 2026-08 | Bagging + Stacking | 0.96 |
| v9 | 2026-09 | Calibration + TTA | **0.97-0.99** |
| LR baseline | — | Bandpower + Logistic | 0.738 |
| Kaggle 1st (2015) | — | Cat & Dog | 0.981 |

## 5 Original Algorithms

1. **SubjectAdaptiveFiLM** — ...
2. **HierarchicalCausalConformer** — ...
3. **CausalBENDR** — ...
4. **EventOrderConsistency Loss** — ...
5. **HierarchicalBayesianCalibration** — ...

## Critical Self-Reflection

This project documents not just successes but failures:
- v2/v3 Conformer collapse and the root cause analysis
- Hypothesis A1 (focal loss collapse) was refuted by experiment
- v4 showed strong negative correlation with LR baseline (r=-0.73)
- Each "failure" generated learning recorded in docs/technical_notes/
```

### 5 つのポートフォリオハイライト

1. **多角的根本原因分析** (v2 RCA 文書)
2. **仮説駆動の改善** (A1 反証含む)
3. **被験者ヘテロ性の構造的解決** (FiLM)
4. **競技グレードアンサンブル** (100 model bagging + stacking)
5. **率直な失敗記録** (Conformer 退化、 仮説反証)

これらは「単に精度が高い」より雇用市場での評価が高い。

---

## 10. 次のステップ（即時）

### Phase 0 完了 (2026-05-17) ✅

- [x] マスタープラン作成（本文書 v1.0）
- [x] `src/eeg99/` パッケージ骨格完成（57 tests all passing）
- [x] `docs/literature_review_master.md` 作成（15 本 + Synthesis + Open Questions）
- [x] Cat & Dog solution 解読 (3-level pipeline, 50+ models, Riemannian features)
- [x] memory に新方針を保存

### Phase 1 開始 (次セッション)

`docs/MASTER_PLAN_99.md` を読んで Phase 1 (因果前処理 + データパイプライン v2) から開始。

最初のアクション (Phase 1, W2):
1. `src/eeg99/data/loader.py` — 実装 (Phase 0 は stub)
2. `src/eeg99/data/preprocess.py` — causal HP/LP filter (0.5-45 Hz)
3. `src/eeg99/data/filterbank.py` — 5 帯域並列 filterbank
4. `src/eeg99/data/dataset.py` — multi-window dataset (PyTorch Dataset)
5. キャッシュ機構 (parquet)

---

## 11. 採用判断（確定事項、 2026-05-14）

ユーザー決定: **全項目 (a)**

| Q | 内容 | 決定 |
|:---:|---|---|
| Q1 | プロジェクト名 | **`src/eeg99/`** |
| Q2 | 既存 codebase | **`src/eegdemo/` 残す**（baseline 参照） |
| Q3 | GPU 予算 | **Kaggle paid のみ ($10/月、 60h/週)** |
| Q4 | タイムライン | **柔軟（学習優先、 終わるまで）** |
| Q5 | 成功定義 | **0.99 必達**（届かないなら延長） |

### 確定事項の含意

**Q3 (Kaggle paid のみ)** の制約:
- 60h/週 = 4 ヶ月で総 ~1000h 利用可能
- プラン総予算 230h → 余裕は十分だが、 Phase 4 (bagging 80h) は **複数週に分割** 必要
- Phase 4 を 3 週に分割（27h/週）、 並行で軽い実験を回す設計

**Q5 (0.99 必達)** の含意:
- 各 Phase で目標未達でも **次に進む**（中間 mile-stone は柔軟）
- 最終的に 0.99 に届くまで **Phase 7 (stretch) を延長**
- 17 週超えても継続、 最大 25-30 週を想定
- ただし「届かない努力」も成果物として正直に記録

**Q4 (柔軟)** との整合:
- 学習優先（thoroughness）を維持
- 各 Phase でじっくり ablation + 文献レビュー
- 結果として 17 週 → 20-25 週も許容

---

## 12. このマスタープランの位置づけ

本文書は **"north star"** として、 4 ヶ月にわたるプロジェクトの設計図。
各 Phase 完了後にこの文書を refine する（version 1.0, 1.1, ...）。

**ファイルバージョン**: 1.1 (Phase 0 完了時 refine)
**v1.0**: 2026-05-14 (初版)
**v1.1**: 2026-05-17 (Phase 0 完了; open questions 追加; Phase 1 kick-off 更新)

### v1.1 追記事項 (Phase 0 知見)

**Cat & Dog 非因果差分について**:
- 彼らの 0.981 は非因果 window を使用 (1秒前 epoch + XDawn: 未来情報あり)
- 我々の因果版の実質的 upper bound は 0.96-0.98 と推定
- 「因果制約 + 0.97 達成」はむしろポートフォリオとして stronger な主張になる

**Open Questions (Phase 1+ で解決)**:
- OQ-1: CausalBENDR の masking 戦略 (Phase 3 ablation)
- OQ-2: causal FBCSP 性能差 (Phase 2, W3 比較)
- OQ-3: HierarchicalBayesianCalibration prior 設計 (Phase 6)
- OQ-4: SubjectAdaptiveFiLM embed_dim ablation {8, 16, 32} (Phase 2, W4)
- OQ-5: 因果/非因果 split での AUC 差の定量化 (Phase 2 baseline 実験)

**依存パッケージ追加確定 (Phase 1 で install)**:
- `pyriemann` — Riemannian geometry
- `lightgbm` — stacking meta-learner
- `mne` — EEG signal processing utilities
- `torch` (既存) — DL backbone
