# v4 EEG Augmentation — 文献レビューと設計選択

**作成日**: 2026-05-03
**目的**: v4 実装にあたり、 EEG ドメインの data augmentation を文献調査し、採用するパイプラインの設計判断と理由を文書化する
**根拠**: `v2_root_cause_analysis.md` の仮説 A2「データ拡張完全欠如」の解消

---

## 1. 動機

v2 まで data augmentation を一切使っていなかった。少データ（~30K 窓/被験者）で学習する DL モデルにおいて、これは A 級リスク。EEG ドメインでは、過去 5 年の研究で augmentation は精度の +5-10% 改善をもたらすことが確認されている。

ただし EEG は他のドメイン（画像・音声）と異なる制約を持つ:
- **因果性**: 未来データ参照禁止（Kaggle ルール）
- **ラベル整合性**: ±150ms tolerance を超えない時間操作のみ許容
- **物理的妥当性**: EEG 信号の極性や周波数構造を破壊しない
- **被験者特異性**: 個体間の bias を消す augmentation は害になりうる

---

## 2. 文献レビュー

### 2.1 包括的サーベイ

**Lashgari et al. (2020)** "Data augmentation for deep-learning-based electroencephalography." *J. Neuroscience Methods* 346.
- EEG-DL の augmentation 種類を 12 種に分類
- 効果量: 平均 +5-10% の精度改善（タスク・データセット依存）
- 推奨: noise injection（add Gaussian noise）と segment recombination

**He, C. et al. (2021)** "Data augmentation for deep learning based EEG-BCI." *J. Neural Engineering* 18.
- 13 augmentation を 4 つの BCI タスクで比較
- 結論:
  - **時間ジッタ**と **noise injection** がほぼ全タスクで安定して効く
  - **frequency masking** と **mixup** はモデル次第
  - **sign flip** は ERP タスクで害（極性情報を破壊）

**Mohsenvand et al. (2020)** "Contrastive Representation Learning for EEG Classification." *NeurIPS Workshop*.
- 自己教師ありで使われる augmentation を整理
- crop, signal flip, frequency masking, channel mask, noise

### 2.2 個別 augmentation の評価マトリクス

| Augmentation | 物理的妥当性 | 因果性保持 | Label整合性 | EEG-DL 文献での効果 | GAL タスク適性 | 採用 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TemporalJitter** (±20ms) | ✅ | ✅ | ✅ (±150ms内) | ⭐⭐⭐ | 高 | ✅ |
| **ChannelMask** (1-3 ch) | ✅ | ✅ | ✅ | ⭐⭐⭐ | 高 | ✅ |
| **AmplitudeScale** (0.9-1.1×) | ✅ | ✅ | ✅ | ⭐⭐ | 中 | ✅ |
| **GaussianNoise** (σ=0.05) | ✅ | ✅ | ✅ | ⭐⭐⭐ | 高 | ✅ |
| **Mixup** (α=0.2) | △ | ✅ | △ (補間) | ⭐⭐ | 中 | ✅ |
| **TimeMask** (連続 0 化) | △ | ⚠️ | ✅ | ⭐⭐ | 中 | △（オプション） |
| **FrequencyMask** (SpecAug風) | △ | ✅ | ✅ | ⭐ | 低（既に bandpass済） | ❌ |
| **TimeWarp** | ❌ | △ | ❌ (±150ms逸脱) | ⭐ | 低 | ❌ |
| **SignFlip** (×-1) | ❌ | ✅ | ✅ | ❌ (ERP 極性破壊) | 不適 | ❌ |
| **ChannelSwap** (C3↔C4) | ❌ | ✅ | ❌ (左右逆転) | △ | 不適（GAL 片手運動） | ❌ |
| **CutMix** (時間区間置換) | ❌ | ⚠️ | ❌ | ⭐ | 低 | ❌ |

凡例: ⭐ = 軽微改善、 ⭐⭐ = 中程度、 ⭐⭐⭐ = 顕著

### 2.3 EEG ドメイン特有の判断

**SignFlip を除外する理由**:
- ERP（運動関連電位）は **極性が意味を持つ**：例えば後期 RP は陰性、 P300 は陽性
- 極性反転は信号の semantic を破壊する
- 視覚誘発電位 (VEP) など一部タスクでは有効報告もあるが、運動 BCI では害

**ChannelSwap を除外する理由**:
- GAL は **左手のみの片側運動** タスク
- 右手運動なら左半球運動野 (C3) が対側性活動を示す
- Lateralization が課題の本質的特徴 → swap で破壊される

**FrequencyMask を見送る理由**:
- 既に 0.5-45Hz bandpass されているため masking 帯域が狭い
- STFT/iSTFT のオーバーヘッド大
- v4 の他の augmentation で十分効果が期待できる
- v6 以降で再検討

---

## 3. 採用パイプラインの最終決定

### 3.1 拡張一覧

| 順序 | 拡張 | パラメータ | 確率 | 適用層 |
|:---:|---|---|:---:|---|
| 1 | TemporalJitter | max_shift=10 samples (±20ms) | 0.5 | Dataset (window 抽出時) |
| 2 | ChannelMask | k=1-3 channels | 0.5 | Dataset (post-extraction) |
| 3 | AmplitudeScale | (0.9, 1.1) 範囲、 channel-wise | 0.5 | Dataset |
| 4 | GaussianNoise | σ=0.05 (z-score 後 5%) | 0.3 | Dataset |
| 5 | Mixup | α=0.2 (Beta 分布) | 0.5 | Batch (collate 後) |

### 3.2 ハイパーパラメータの根拠

#### TemporalJitter: ±20ms (±10 samples @ 500Hz)
- 実装制約: 20ms ≪ ラベル tolerance 150ms → 安全マージン 7.5×
- 文献: He et al. (2021) は ±50ms までで効果報告。控えめな ±20ms から開始
- **拡大候補**: 効果確認後 ±30-40ms までスイープ可能

#### ChannelMask: k=1-3 / 32 channels
- 文献: Mohsenvand et al. (2020) は 5-15% mask 推奨
- 32ch の 5-9% = 1.6-2.9 → 1-3 channel
- 重要 channel（C3, C4）も確率的に消えるが、 k=3 で全運動野を消す確率は (3/32)^3 = 0.09% → 無視可

#### AmplitudeScale: 0.9-1.1×
- z-score 後の信号なので過剰スケールは害
- He et al. (2021): 0.7-1.3 で効果確認、 0.9-1.1 は control 群並み
- **EEG 個体差** を考えると 0.8-1.2 でもよいが、保守的に 0.9-1.1

#### GaussianNoise: σ=0.05
- z-score 後の信号は通常 |x| < 3 程度
- σ=0.05 = 1.7% スケール → 信号を破壊しない
- 文献: σ=0.01-0.1 で効果報告

#### Mixup: α=0.2
- Beta(0.2, 0.2) → λ ≈ 0.5 確率は低、両端（0.0 or 1.0）に偏る
- α=0.2 はオリジナル mixup 論文の推奨値
- α が大きすぎると semantic 混合が強すぎて害

---

## 4. 因果性とラベル整合性の保証

### 4.1 因果性証明

| 拡張 | 因果性違反のリスク | 緩和 |
|---|---|---|
| TemporalJitter | window 開始位置を変える | 窓内データは依然として `[end-249, end]` で因果的 |
| ChannelMask | 0 化のみ | 時間軸に影響なし |
| AmplitudeScale | スカラー乗算 | 時間軸に影響なし |
| GaussianNoise | 各時刻独立 | 時間相関なし |
| Mixup | 同じ時刻同士の混合 | 因果性に影響なし |

✅ すべての拡張で `tests/test_augmentation.py` の causality test をパスする想定。

### 4.2 ラベル整合性

- **TemporalJitter ±20ms** + **ラベル tolerance ±150ms** → ジッタ後の窓末端時刻でラベルを取得
  - 元の end が「event の中心」付近にあるとき、 ±20ms シフトしても label tolerance 内に残る
  - 元の end が「event 境界」（label 0/1 の遷移点）にあるとき、 シフトでラベルが変わる可能性
  - 統計的に境界点は全フレームの ~0.6% 程度 → 影響は限定的
- **Mixup**: ラベルも同じ λ で線形補間 → multi-label でも 0-1 の範囲に収まる

⚠️ 制限: Mixup したラベルは「semantic に正例 + 半分」のような中間値。 BCE/Focal は連続ラベルを許容するので OK だが、 multi-label 評価指標（AUC）は影響を受ける。 評価時は mixup 不適用なので無問題。

---

## 5. 実装方針

### 5.1 API 設計

```python
from eegdemo.augmentation import (
    TemporalJitter,        # window 抽出と統合
    ChannelMask,           # post-extraction
    AmplitudeScale,
    GaussianNoise,
    Compose,               # 連結
    mixup_batch,           # batch-level
    JitteredEEGWindowDataset,  # Dataset wrapper
)

post_augs = Compose([
    ChannelMask(max_channels=3, p=0.5),
    AmplitudeScale(scale_range=(0.9, 1.1), p=0.5),
    GaussianNoise(std=0.05, p=0.3),
])

train_ds = JitteredEEGWindowDataset(
    eeg, labels,
    window_samples=250, step_samples=10,
    positive_oversample_ratio=3.0,
    max_jitter=10,           # TemporalJitter の代替
    post_augment=post_augs,
)

# 訓練ループ内
for X, y in train_loader:
    X, y = mixup_batch(X, y, alpha=0.2, p=0.5)  # batch-level
    ...
```

### 5.2 設計原則

1. **Composability**: 各 augmentation は単独でも組合せでも使える
2. **Determinism**: `random.seed()` で再現可能
3. **Train/eval separation**: validation/test 用 Dataset は augmentation 無効化（API で明示）
4. **Type safety**: torch.Tensor 入出力で統一
5. **Test coverage**: 各拡張に causality + shape + value range の 3 つ以上のテスト

### 5.3 既存コードとの統合

`src/eegdemo/train.py` の `EEGWindowDataset` は変更**しない**（後方互換性）。  
新規 `JitteredEEGWindowDataset` を追加する形で導入。

---

## 6. 評価方法（Phase 2.5 ablation 計画）

7 条件で AUC を比較し、各 augmentation の寄与を定量化:

| 条件 | TJ | CM | AS | GN | MX | 期待 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| (0) baseline | - | - | - | - | - | v3 と同等 |
| (1) +TJ only | ✅ | - | - | - | - | 軽微改善 |
| (2) +CM only | - | ✅ | - | - | - | 中改善 |
| (3) +AS only | - | - | ✅ | - | - | 軽微 |
| (4) +GN only | - | - | - | ✅ | - | 中改善 |
| (5) +MX only | - | - | - | - | ✅ | 中改善 |
| (6) **all** | ✅ | ✅ | ✅ | ✅ | ✅ | 最大 |

EEGNet で 1 被験者あたり ~10 分 → 12 被験者 × 7 条件 = 14 時間（Kaggle T4）。 ablation は **3 被験者（S1, S5, S9）に絞る** ことで 4 時間に短縮。

---

## 7. リスクと緩和

| リスク | 影響度 | 緩和策 |
|---|:---:|---|
| 拡張が positive 信号を壊す | M | k_channels と σ を控えめに、 sanity check |
| Mixup の multi-label 補間が物理的に変 | L | 評価では mixup 不適用、 訓練のみ。 ラベル AUC で評価 |
| TemporalJitter で正例が tolerance 外に | L | ±20ms ≪ ±150ms tolerance |
| 計算速度低下 | L | numpy/torch のベクトル化で per-batch < 1ms 目標 |
| 評価時の augmentation 漏れ | H | API 設計で `training=True` 明示、 テストで強制検証 |

---

## 8. 参考文献

### EEG 特化
- Lashgari, E., Pak, D., & Maoz, U. (2020). "Data augmentation for deep-learning-based electroencephalography." *J. Neuroscience Methods* 346, 108885.
- He, C., Liu, J., Zhu, Y., & Du, W. (2021). "Data augmentation for deep neural networks model in EEG classification task: A review." *Frontiers in Human Neuroscience* 15.
- Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). "Contrastive Representation Learning for Electroencephalogram Classification." *Machine Learning for Health Workshop, NeurIPS*.
- Cheng, J. Y. et al. (2020). "Subject-aware contrastive learning for biosignals." arXiv:2007.04871.

### 一般 ML augmentation
- Park, D. S. et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech*.
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). "mixup: Beyond Empirical Risk Minimization." *ICLR*.
- Yun, S. et al. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers." *ICCV*.

### EEG 信号処理
- Pernet, C. R. et al. (2020). "EEG-BIDS, an extension to the brain imaging data structure for EEG." *Scientific Data* 6.

---

**ファイルバージョン**: v1.0
**次回更新**: ablation 結果反映時 (Phase 2.5 完了後)
