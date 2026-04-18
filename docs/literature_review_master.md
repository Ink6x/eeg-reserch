# EEG-99 Literature Review Master

**作成日**: 2026-05-17  
**バージョン**: 1.0  
**担当**: Phase 0 文献集中レビュー  
**目的**: 5 つの独自アルゴリズム設計のための文献基盤を整理する

---

## §0 サマリーテーブル

| # | 著者 (年) | 題名 (短縮) | 分類 | Adoption | 対応 Component |
|:--:|---|---|:---:|:---:|:---:|
| 1 | Barachant & Cycon (2015) | Cat & Dog — Kaggle GAL 1位 | Competition | ADAPT | 8 (Ensemble) |
| 2 | Pavlyshenko (2018) | Using Stacking Approaches | Competition | ADOPT | 9 (Stacking) |
| 3 | Luciw et al. (2014) | WAY-EEG-GAL dataset | Dataset | REFERENCE | — |
| 4 | Kostas et al. (2021) | BENDR: EEG SSL transformer | SSL | ADAPT | 6 (SSL) |
| 5 | Yang et al. (2023) | BIOT: Cross-data Biosignal | SSL | REFERENCE | 6 (SSL) |
| 6 | Banville et al. (2021) | Self-supervised EEG (RP/CPC) | SSL | REFERENCE | 6 (SSL) |
| 7 | Lawhern et al. (2018) | EEGNet | Architecture | ADOPT-AS-IS | 1 (MultiScale) |
| 8 | Song et al. (2022) | EEG Conformer | Architecture | ADAPT | 5 (HierConformer) |
| 9 | HCFT (2025) | Hierarchical Convolutional Fusion Transformer | Architecture | REFERENCE | 5 (HierConformer) |
| 10 | Ang et al. (2008) | FBCSP | Classical | ADOPT-AS-IS | 2 (FBCSP) |
| 11 | Barachant et al. (2013) | Riemannian-based kernel BCI | Classical | ADAPT | 3 (Riemannian) |
| 12 | Perez et al. (2018) | FiLM | Conditioning | ADOPT-AS-IS | 4 (SubjectFiLM) |
| 13 | Huang & Belongie (2017) | AdaIN | Conditioning | REFERENCE | 4 (SubjectFiLM) |
| 14 | Niculescu-Mizil & Caruana (2005) | Predicting Good Probabilities | Calibration | ADOPT-AS-IS | 10 (Calibration) |
| 15 | Guo et al. (2017) | On Calibration of Modern NNs | Calibration | ADOPT-AS-IS | 10 (Calibration) |

**Adoption 定義**:
- **ADOPT-AS-IS**: アルゴリズムをほぼそのまま実装  
- **ADAPT**: 核心アイデアを採用、EEG-99 向けに改変  
- **REFERENCE-ONLY**: 背景理解・比較対象として参照  
- **REJECTED**: 採用しない (理由を明記)

---

## §1 Barachant & Cycon (2015) — Kaggle GAL 1位

**Source**: Kaggle blog writeup; HAL 論文 "Pushing the limits of BCI accuracy" (hal-01349562); GitHub: `alexandrebarachant/Grasp-and-lift-EEG-challenge`

**Why we read it**: 我々が目標とするスコアを達成した唯一の公開手法。設計判断の全基準。

**Key idea**: 3-level ensemble pipeline で AUC 0.981 達成。単一モデルではなく、50+ モデルの多様な ensemble が勝因。

**Method essence**:
```
Level 1 — 多様な単一モデル群 (Riemannian + filter bank + XDawn ERP):
  - 32ch EEG → Butterworth lowpass FB (0.5, 1, 2, 3, 4, 5, 7, 9, 15, 30 Hz)
  - XDawn spatial filter → covariance → Riemannian tangent space → LDA/SVM
  - FBL_delay features: 5 past samples × 200-sample intervals (2 sec span)

Level 2 — 各 L1 モデルの OOF 予測を concat → LSTM/XGBoost で統合
Level 3 — L2 出力を重み付き平均 (AUC 最大化で最適化)
```

**Adoption decision**: **ADAPT**  
核心戦略（多段 ensemble + Riemannian + filter bank）を採用。ただし以下を置換:
- Non-causal window → causal window (重要: 彼らは1秒前 epoch、未来参照あり)
- XDawn (非因果 ERP average) → causal running average で代替
- LSTM L2 → LightGBM stacking (より interpretable)

**Implementation note for eeg99**:
- `ensemble/stacking.py` の 3-level 構造はこの設計を参照
- `features/riemannian.py` の covariance → tangent space が L1 の核心
- L2 OOF 生成は `ensemble/bagging.py` で管理

**Caveats / Limitations**:
- Non-causal 設計のため、Kaggle submission はサンプル単位でも実質的に未来情報にアクセス可能
- 我々の厳密因果版は性能低下を許容すべき (推定 0.01-0.03 AUC 差)
- 彼らの LSTM L2 は time-series 的な整合性を利用しており、LightGBM 代替では取れない情報がある可能性

---

## §2 Pavlyshenko (2018) — Using Stacking Approaches

**Source**: IEEE DSMP 2018, pp. 255-258. DOI: 10.1109/DSMP.2018.8478522. Semantic Scholar: 2dee25bb

**Why we read it**: Component 9 (Stacking meta-learner) の理論的根拠。

**Key idea**: Base learners の OOF (out-of-fold) 予測を第2段階の meta-learner の入力とすることで、各 base model の bias を補正しつつ complementary な情報を統合できる。

**Method essence**:
```python
# 擬似コード (stacking の骨格)
for fold in k_folds:
    train_base_models(fold.train)
    oof_predictions[fold.val] = predict(fold.val)  # OOF
meta_X = oof_predictions  # shape: (n_samples, n_base_models * n_events)
meta_model = LightGBM().fit(meta_X, y)
```

**Adoption decision**: **ADOPT-AS-IS**  
標準的な stacking。LightGBM を meta-learner に選ぶ (interpretability + fast training)。

**Implementation note for eeg99**:
- `ensemble/stacking.py`: 5-fold subject-stratified CV で OOF 生成
- per-event meta-model (6 event × 1 LightGBM = 6 models)
- OOF 予測は model registry に保存 (`utils/registry.py`)

**Caveats / Limitations**:
- Subject-stratified split が必須 (被験者混同を防ぐ)
- 過学習リスク: base models が 50+ の場合 meta-learner の feature 数が大きい → L1 正則化必須

---

## §3 Luciw et al. (2014) — WAY-EEG-GAL Dataset

**Source**: Scientific Data 1:140047, doi: 10.1038/sdata.2014.47. PubMed: 25977798.

**Why we read it**: 我々が使用するデータセットの公式記述論文。仕様の根拠。

**Key idea**: 12 被験者 × 10 series × 32ch EEG @ 500 Hz, 6 イベント (HandStart, FirstDigitTouch, BothStartLoadPhase, LiftOff, Replace, BothReleased)。

**Method essence**:
```
被験者: 12 人 (健常者)
Series: 10 series/subject (8 train + 2 test = Kaggle 区分)
EEG: 32 ch, 500 Hz, BrainAmp amplifier
Events: 6 (順序固定: HandStart < FirstDigitTouch < ... < BothReleased)
物体重量: 165/330/660 g (unpredictably varied)
表面摩擦: sandpaper/suede/silk (unpredictably varied)
```

**Adoption decision**: **REFERENCE-ONLY**  
データ仕様の参照のみ。`src/eeg99/data/loader.py` の SubjectSeries dataclass はこの仕様から設計済み。

**Implementation note for eeg99**:
- `constants.py` に 6 イベント名・サンプリング周波数・被験者数を定義 (Phase 1)
- 32ch → channel map は `channels.jpg` 参照

**Caveats / Limitations**:
- Test series (9-10) のラベルは公開されていない → SSL pre-training ではラベルなしで使用可能

---

## §4 Kostas et al. (2021) — BENDR

**Source**: Frontiers in Human Neuroscience, 2021. DOI: 10.3389/fnhum.2021.653659. arXiv: 2101.12037.

**Why we read it**: Component 6 (CausalBENDR) の直接の原型。

**Key idea**: EEG 専用の contrastive SSL pre-training。波形の一部を mask し、contextual representation と target segment の表現を contrastive loss で近づける。wav2vec 2.0 の EEG 版。

**Method essence**:
```python
# BENDR の核心 (因果版への改変前)
z = temporal_conv_encoder(eeg)          # local features
z_masked = apply_mask(z, p=0.5)         # 50% time masking
c = transformer(z_masked)               # context
loss = InfoNCE(c, z_target)             # contrastive: context vs target
```

**Adoption decision**: **ADAPT** → **CausalBENDR** (独自アルゴリズム #3)  
改変点:
1. `transformer` → causal transformer (future attention を mask)
2. masking: 未来 token のみを mask (現在以前はそのまま)
3. target: masked 位置の true embedding (因果的に推論可能な範囲)
4. auxiliary loss: MAE 風復元 loss を追加

```python
# CausalBENDR 設計 (eeg99 での実装予定)
class CausalMaskedEEGEncoder(nn.Module):
    def __init__(self):
        self.conv_encoder = CausalConvEncoder(...)
        self.causal_transformer = CausalTransformer(causal_mask=True)
        self.projection = nn.Linear(d_model, d_proj)

    def forward(self, eeg, mask_future_only=True):
        z = self.conv_encoder(eeg)              # (T, C, d)
        z_masked = causal_mask_future(z)        # mask p=0.5 of future
        c = self.causal_transformer(z_masked)   # causal context
        return c, z  # (context, target)
```

**Implementation note for eeg99**:
- `models/ssl_bendr.py` (Phase 3 で実装)
- pre-training data: 12 subj × 10 series (test series も label なしで使用)
- linear probing で SSL 表現の質を評価してから fine-tune

**Caveats / Limitations**:
- 元の BENDR は非因果 transformer → 因果版は性能低下の可能性あり
- contrastive loss は negative sample の選択が重要: 同一被験者内では避ける
- 計算コスト高: Phase 3 で GPU 40h 消費予定

---

## §5 Yang et al. (2023) — BIOT

**Source**: NeurIPS 2023. arXiv: 2305.10351. GitHub: `ycq091044/BIOT`.

**Why we read it**: BENDR より新しい EEG foundation model。mismatched channels への対応が特徴。

**Key idea**: 各 channel を独立に tokenize し、channel embedding + relative position embedding を加えて Transformer で統合。複数データセット横断で pre-train 可能。

**Method essence**:
```python
# BIOT の channel-wise tokenization
for ch in channels:
    tokens[ch] = linear_proj(segment(eeg[ch]))  # 固定長セグメントに分割
x = concat([tokens[ch] + ch_embed[ch] + pos_embed for ch in channels])
y = transformer(x)
```

**Adoption decision**: **REFERENCE-ONLY**  
BENDR の方が実装が simpler で我々のユースケース (固定 32ch) に向いている。BIOT の channel-wise tokenization は参考にするが直接採用はしない。

**Caveats / Limitations**:
- mismatched channels 対応の複雑さは我々には不要 (32ch固定)
- GitHub 実装は参考コードとして Phase 3 で参照

---

## §6 Banville et al. (2021) — Self-supervised EEG (RP/CPC)

**Source**: Journal of Neural Engineering, Vol. 18, No. 4, 2021. DOI: 10.1088/1741-2552/abca18. PubMed: 33181507.

**Why we read it**: SSL for EEG の基礎論文。複数の pretext task を比較。

**Key idea**: 3つの pretext task を比較: Relative Positioning (RP), Temporal Shuffling (TS), Contrastive Predictive Coding (CPC)。低ラベル regime で教師あり DNN を上回る。

**Method essence**:
```python
# Relative Positioning (RP): 2 window が "近い" か "遠い" かを判定
anchor = eeg[t1:t1+W]
positive = eeg[t2:t2+W]  # |t1-t2| < tau_pos
negative = eeg[t3:t3+W]  # |t1-t3| > tau_neg
loss = BinaryCE(sim(anchor, positive) > sim(anchor, negative))
```

**Adoption decision**: **REFERENCE-ONLY**  
RP/CPC よりも BENDR の masked modeling の方が我々のアーキテクチャに一貫している。補助 loss として RP を追加実装する可能性はある (Phase 3 で判断)。

**Caveats / Limitations**:
- 睡眠・病理データで評価; motor imagery への転用は別途確認が必要

---

## §7 Lawhern et al. (2018) — EEGNet

**Source**: Journal of Neural Engineering, Vol. 15, No. 5, p. 056013, 2018. arXiv: 1611.08024. DOI: 10.1088/1741-2552/aace8c.

**Why we read it**: 我々の v1-v4 baseline。Component 1 (MultiScaleCausalNet) の出発点。

**Key idea**: Depthwise separable convolution で EEG 専用の compact モデルを設計。時間 conv → depthwise spatial conv → separable conv の 3 block。

**Method essence**:
```python
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, sfreq, F1=8, D=2, F2=16):
        # Block 1: temporal + depthwise spatial
        self.temporal_conv = Conv2d(1, F1, (1, sfreq//2), padding='same')
        self.depthwise = Conv2d(F1, F1*D, (n_channels, 1), groups=F1)
        self.bn1 = BatchNorm2d(F1*D)
        # Block 2: separable conv
        self.separable = Conv2d(F1*D, F2, (1, 16), padding='same', groups=F1*D)
        self.pointwise = Conv2d(F2, F2, 1)
```

**Adoption decision**: **ADOPT-AS-IS** → **MultiScaleEEGNet+** へ拡張  
v4 実装 (`src/eegdemo/models/eegnet.py`) を eeg99 向けに拡張:
- 複数 temporal window (125, 250, 500, 1000, 2000 samples) の parallel branches
- SubjectAdaptiveFiLM (Component 4) を各 BatchNorm に統合
- causal padding を全 conv に適用

**Implementation note for eeg99**:
- `models/eegnet_plus.py` (Phase 2, W4)

**Caveats / Limitations**:
- v4 で AUC 0.733 — LR baseline と同程度。単独では不十分
- per-subject 分析で被験者ヘテロ性が大きい → FiLM 統合が必須

---

## §8 Song et al. (2022) — EEG Conformer

**Source**: IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710-719, 2022. GitHub: `eeyhsong/EEG-Conformer`.

**Why we read it**: Component 5 (HierarchicalCausalConformer) の原型。v2-v4 で崩壊した Conformer の改良版の参考。

**Key idea**: CNN (局所特徴) + Transformer (長距離依存) を serial に結合。PatchEmbedding で EEG を token 化 → Self-attention で global context。

**Method essence**:
```python
class EEGConformer(nn.Module):
    # 1. Patch embedding via shallow FBCSP-inspired conv stem
    self.patch_embed = ShallowConvStem(n_ch, d_model)
    # 2. Transformer encoder (full attention)
    self.transformer = TransformerEncoder(n_layers=6, n_heads=8, d_model=d_model)
    # 3. Classification head
    self.head = Linear(d_model, n_classes)
```

**Adoption decision**: **ADAPT** → **HierarchicalCausalConformer** (独自アルゴリズム #2)  
v2-v4 での崩壊原因分析 (`docs/technical_notes/v2_root_cause_analysis.md`) を反映:
- 崩壊原因: full self-attention での数値不安定性 + gradient vanishing
- 改変: 3階層 attention (short 125ms / medium 500ms / long 2s) を並列に
- causal mask を全 attention に強制 (未来 token を mask)
- 各階層の出力を concat → cross-scale attention で統合

```python
class HierarchicalCausalConformer(nn.Module):
    """3-scale causal attention: short / medium / long context."""
    def __init__(self):
        self.short_attn  = CausalTransformerBlock(window=125)
        self.medium_attn = CausalTransformerBlock(window=500)
        self.long_attn   = CausalTransformerBlock(window=2000)
        self.cross_scale = CrossScaleAttention()  # fuse 3 scales
        self.film        = SubjectAdaptiveFiLM()  # Component 4
```

**Implementation note for eeg99**:
- `models/conformer_hier.py` (Phase 2, W4)
- 単独目標: AUC > 0.80 (v2-v4 での 0.55-0.73 から大幅改善)

**Caveats / Limitations**:
- 計算コスト: full attention は O(T²)。window attention で対応
- 崩壊リスクは階層化で低減できる見込みだが未実証

---

## §9 HCFT (2025) — Hierarchical Convolutional Fusion Transformer

**Source**: arXiv: 2601.12279 (January 2025). "HCFT: Hierarchical Convolutional Fusion Transformer for EEG Decoding"

**Why we read it**: HierarchicalCausalConformer 設計の参考アーキテクチャ。最新手法。

**Key idea**: 複数解像度の CNN branch からの特徴を Transformer で cross-scale fusion。local (CNN) + global (Transformer) の階層的統合。

**Method essence**:
```python
# HCFT の核心
branches = [CNN(kernel=k) for k in [32, 64, 128, 256]]  # multi-scale
features = [b(eeg) for b in branches]
fused = HierarchicalFusionTransformer(features)
```

**Adoption decision**: **REFERENCE-ONLY**  
設計の参考として使用。我々の HierarchicalCausalConformer は独自の因果制約と Subject FiLM を追加する。

**Caveats / Limitations**:
- non-causal 設計 → 因果版への変換が必要
- Grasp-and-Lift での評価なし (motor imagery 系データセット)

---

## §10 Ang et al. (2008) — FBCSP

**Source**: IEEE IJCNN 2008, pp. 2390-2397. DOI: 10.1109/IJCNN.2008.4634130.

**Why we read it**: Component 2 (FBCSP) の原論文。古典 BCI 手法で受賞者も使用。

**Key idea**: 5-6 帯域に分解 → 各帯域で CSP (Common Spatial Pattern) spatial filter → log-variance を特徴に → 分類。

**Method essence**:
```python
def fbcsp(eeg, bands=[(4,8),(8,12),(12,16),(16,20),(20,24),(24,28)]):
    features = []
    for low, high in bands:
        filtered = butterworth_bandpass(eeg, low, high)
        W = fit_csp(filtered, labels)          # CSP spatial filters
        projected = W.T @ filtered              # (n_comp, T)
        features.append(np.log(np.var(projected, axis=1)))
    return np.concatenate(features)            # (n_bands * n_comp,)
```

**Adoption decision**: **ADOPT-AS-IS**  
`features/fbcsp.py` に実装。因果版では causal Butterworth filter を使用。CSP は epoch 全体を使用するため、sliding window を causal に適用。

**Implementation note for eeg99**:
- `features/fbcsp.py` (Phase 2, W3)
- hybrid model (`models/hybrid.py`) の feature stream の一つ
- 目標: 単独 SVM で AUC > 0.74 (LR baseline を上回る)

**Caveats / Limitations**:
- CSP は二値分類向け。6 イベント → one-vs-rest または per-event FBCSP
- 因果適用: window-based → 計算コスト O(T × n_windows)

---

## §11 Barachant et al. (2013) — Riemannian-based kernel BCI

**Source**: Neurocomputing, vol. 112, pp. 172-178, 2013. HAL: hal-00820475. DOI: 10.1016/j.neucom.2012.12.039.

**Why we read it**: Component 3 (CausalRiemannianBlock) の理論的基盤。Cat & Dog の L1 核心。

**Key idea**: EEG の空間共分散行列を SPD (Symmetric Positive Definite) 多様体上の点として扱い、Riemannian 距離ベースのカーネルを SVM と組み合わせる。

**Method essence**:
```python
def riemannian_feature(eeg_epoch):
    C = cov(eeg_epoch)                    # (n_ch, n_ch) SPD matrix
    # Riemannian mean の tangent space に投影
    T = tangent_space(C, reference=C_mean)  # (n_ch*(n_ch+1)/2,) vector
    return T  # Euclidean 空間の特徴ベクトルとして使用

# pyriemann で実装可能
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
```

**Adoption decision**: **ADAPT** → **CausalRiemannianBlock** (Component 3)  
改変点:
- causal running covariance (全 epoch ではなく sliding window)
- `pyriemann` を依存に追加
- DL model の layer として微分可能な実装 (geomstats または独自実装)

**Implementation note for eeg99**:
- `features/riemannian.py` (Phase 2, W3)
- hybrid model でも使用

**Caveats / Limitations**:
- 計算コスト: 共分散行列の Riemannian 演算は O(n_ch³)
- 微分可能化は複雑: Phase 2 では sklearn ベース、Phase 4 以降で DL 統合

---

## §12 Perez et al. (2018) — FiLM

**Source**: AAAI 2018. arXiv: 1709.07871. GitHub: `ethanjperez/film`.

**Why we read it**: Component 4 (SubjectAdaptiveFiLM) の直接の技術的基盤。

**Key idea**: 条件情報 (ここでは subject_id) で neural network の各 feature map を affine 変換: `y = γ(c) * x + β(c)`。VQA で CLEVR ベンチマークの誤り率を半減。

**Method essence**:
```python
class FiLMLayer(nn.Module):
    def __init__(self, d_feat, d_cond):
        self.gamma_net = Linear(d_cond, d_feat)
        self.beta_net  = Linear(d_cond, d_feat)

    def forward(self, x, cond):
        γ = self.gamma_net(cond)   # (batch, d_feat)
        β = self.beta_net(cond)    # (batch, d_feat)
        return γ.unsqueeze(-1) * x + β.unsqueeze(-1)  # feature-wise affine
```

**Adoption decision**: **ADOPT-AS-IS** → **SubjectAdaptiveFiLM** (独自アルゴリズム #1)  
EEG への適用が我々の独自貢献:
- `cond` = subject embedding (n_subjects=12, embed_dim=16)
- 全モデルの全 BatchNorm を FiLM で置換
- FiLM γ, β が各被験者のヘテロ性を吸収

```python
class SubjectAdaptiveFiLM(nn.Module):
    def __init__(self, n_subjects=12, embed_dim=16, n_features):
        self.subject_emb = Embedding(n_subjects, embed_dim)
        self.film = FiLMLayer(n_features, embed_dim)

    def forward(self, x, subject_id):
        c = self.subject_emb(subject_id)
        return self.film(x, c)
```

**Implementation note for eeg99**:
- `models/adapter.py` (Phase 2, W4)
- 全モデル (EEGNet+, Conformer, TCN, Hybrid) に統合
- 期待効果: AUC +0.04-0.06 (per-subject analysis で検証)

**Caveats / Limitations**:
- EEG への FiLM 適用は先行研究なし → 我々の ablation が検証になる
- subject embedding は訓練データのある 12 被験者のみ。unseen subject には対応不可 (今回は問題なし)

---

## §13 Huang & Belongie (2017) — AdaIN

**Source**: ICCV 2017 (Oral). arXiv: 1703.06868. GitHub: `xunhuang1995/AdaIN-style`.

**Why we read it**: FiLM の代替候補。style transfer から来る feature normalization。

**Key idea**: instance normalization の mean/variance をスタイル統計で置換: `AdaIN(x, y) = σ(y) * ((x - μ(x)) / σ(x)) + μ(y)`.

**Adoption decision**: **REFERENCE-ONLY**  
FiLM の方が subject embedding との統合が straightforward。AdaIN は IN の mean/variance を使うため、batch 内の分布に依存し、EEG の非定常性との相性が不明。Phase 2 ablation で必要であれば比較実験を追加。

**Caveats / Limitations**:
- AdaIN は instance normalization ベース; EEG で IN が有効かは不明

---

## §14 Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities

**Source**: ICML 2005, pp. 625-632. DOI: 10.1145/1102351.1102430. PDF: cs.cornell.edu.

**Why we read it**: Component 10 (HierarchicalBayesianCalibration) の Platt scaling 基盤。

**Key idea**: Boosted tree, SVM 等は確率をよく calibrate しない (sigmoid-shaped distortion)。Platt scaling (sigmoid fitting) または Isotonic Regression で補正できる。

**Method essence**:
```python
# Platt scaling: sigmoid で変換
from sklearn.calibration import CalibratedClassifierCV

# 独立した calibration データ (val set) で fit
cal_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
cal_model.fit(X_val, y_val)
proba = cal_model.predict_proba(X_test)[:, 1]  # calibrated probabilities
```

**Adoption decision**: **ADOPT-AS-IS** → per-subject Platt scaling  
Component 10 (HierarchicalBayesianCalibration) の基盤:
- 各被験者ごとに独立した Platt scaling
- 階層 Bayes prior: 被験者間で sharing (独自拡張)

**Implementation note for eeg99**:
- `ensemble/calibration.py` (Phase 6)
- calibration data: val series (subject-specific)

**Caveats / Limitations**:
- calibration は正しい train/val split が必要; val を calibration に使うと訓練と混同するリスク
- Platt scaling は 2-parameter sigmoid; 複雑な歪みには Isotonic Regression が有効

---

## §15 Guo et al. (2017) — On Calibration of Modern Neural Networks

**Source**: ICML 2017. arXiv: 1706.04599. Proceedings: proceedings.mlr.press/v70/guo17a.

**Why we read it**: 現代 DNN の calibration 問題と temperature scaling の有効性。

**Key idea**: 現代の DNN (ResNet 等) は calibration が悪化している (confidence が高すぎる)。Temperature scaling (1 parameter T で softmax 温度を調整) が最もシンプルかつ効果的。

**Method essence**:
```python
def temperature_scale(logits, T):
    """T > 1 で confident を抑制, T < 1 で confident を増大."""
    return logits / T

# 最適 T: NLL 最小化
T_opt = minimize(nll(temperature_scale(logits_val, T), y_val))
```

**Adoption decision**: **ADOPT-AS-IS**  
ensemble の final output に temperature scaling を適用。Platt scaling (§14) と組み合わせ:
1. per-subject Platt scaling で個人差補正
2. temperature scaling で全体的な信頼度調整

**Implementation note for eeg99**:
- `ensemble/calibration.py` に組み込み (Phase 6)

**Caveats / Limitations**:
- Temperature scaling は単一スカラー → 全クラス・全サンプルに同じ T を適用
- per-event, per-subject で T を別々に最適化する方が良い可能性 (我々の独自拡張)

---

## §16 Synthesis: 5つの独自アルゴリズムの系譜

以下に、5つの独自アルゴリズムが各文献からどう派生したかを整理する。

```
┌───────────────────────────────────────────────────────────────────┐
│ 独自アルゴリズム #1: SubjectAdaptiveFiLM                          │
│   ← FiLM (§12, Perez 2018): 核心メカニズム                        │
│   ← AdaIN (§13, Huang 2017): alternative として参照               │
│   ← Cat & Dog (§1): 被験者ヘテロ性の重要性を示した                │
│   独自貢献: EEG + subject embedding への適用 (先行研究なし)        │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 独自アルゴリズム #2: HierarchicalCausalConformer                  │
│   ← EEG Conformer (§8, Song 2022): 基本アーキテクチャ            │
│   ← HCFT (§9, 2025): 階層的 fusion の設計参考                    │
│   ← EEGNet (§7, Lawhern 2018): depthwise spatial 部分を継承      │
│   独自貢献: 3階層因果 attention + SubjectFiLM 統合 (崩壊を防ぐ)  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 独自アルゴリズム #3: CausalBENDR                                  │
│   ← BENDR (§4, Kostas 2021): contrastive masked SSL              │
│   ← BIOT (§5, Yang 2023): foundation model 設計思想              │
│   ← Banville (§6, 2021): RP/CPC の SSL 多様性                    │
│   独自貢献: 因果 masked modeling (BENDR の causal 派生)           │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 独自アルゴリズム #4: EventOrderConsistency Loss                   │
│   ← Luciw (§3): 6イベントの固定順序 (データ仕様)                  │
│   ← Cat & Dog (§1): イベント間の相関を ensemble で利用            │
│   独自貢献: 順序制約を auxiliary loss として training に統合       │
│   (先行研究なし — 完全独自)                                       │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 独自アルゴリズム #5: HierarchicalBayesianCalibration              │
│   ← Niculescu-Mizil & Caruana (§14): Platt scaling 基盤          │
│   ← Guo et al. (§15): temperature scaling + DNN calibration      │
│   独自貢献: per-subject 階層 Bayes prior 付き calibration         │
│   (既存: per-subject Platt; 独自: hierarchical Bayes sharing)    │
└───────────────────────────────────────────────────────────────────┘
```

---

## §17 Open Questions for Phase 1+

Phase 0 での文献調査から生じた未解決の設計判断:

### Q1: CausalBENDR の masking 戦略
- BENDR は双方向 transformer で未来 token も参照
- 因果版では「未来のみ mask」vs「ランダム mask (過去のみ)」のどちらが適切か
- **Phase 3 で小規模 ablation で判断**

### Q2: FBCSP の因果適用でどのくらい性能が落ちるか
- 古典 CSP は epoch 全体の共分散を使う (非因果)
- causal running covariance は理論的には劣る
- **Phase 2, W3 で causal vs non-causal FBCSP を比較**

### Q3: HierarchicalBayesianCalibration の Bayes prior 設計
- 被験者間での prior sharing をどう実装するか
- 単純な Gaussian prior vs. 完全な階層モデル
- **Phase 6 で empirical Bayes として実装 (まず simple, 複雑化は後で)**

### Q4: SubjectAdaptiveFiLM の embed_dim 選択
- embed_dim=16 (master plan の仮設定) は根拠が薄い
- 12 被験者 × embed_dim matrix を過学習するリスク
- **Phase 2, W4 で embed_dim ∈ {8, 16, 32} を ablation**

### Q5: Cat & Dog の非因果差分は AUC 換算でどのくらいか
- 彼らの AUC 0.981 は非因果 (validation はラベル既知、先読み可)
- 我々の因果版での期待 upper bound は 0.96-0.98 程度か
- **Phase 2 baseline 再現実験で定量化 (causal vs non-causal split 比較)**

---

**文書バージョン**: 1.0  
**次回更新**: Phase 1 完了時 (data pipeline 確立後)  
**関連ファイル**:
- `docs/MASTER_PLAN_99.md` — プロジェクト全体設計
- `src/eeg99/` — 実装
- `docs/eeg99/` — 各独自アルゴリズムの詳細設計ノート (Phase 2 以降に作成)
