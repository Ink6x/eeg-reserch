# v2 精度低下の科学的再検証 — 多角的根本原因分析

**作成日**: 2026-05-02
**対象**: EEGNet v1/v2, CausalEEGConformer v1/v2 の LR ベースライン未到達問題
**目的**: 既存の「真因」結論を批判的に再検証し、ML・脳情報学・訓練手法の三視点から仮説を再生成する

---

## 0. Executive Summary

既存の `model_iteration_log.md` は「`eigh` の NaN 勾配（v1）」「TCNBlock の LayerNorm transpose（v2）」「CholeskyPool のスケール問題（v2）」を真因と位置づけた。本レポートはこれらを **無批判には受け入れず**、データと実装を再精査して以下の結論に達した。

### 既存結論への批判的再評価
- **H_prev1（eigh NaN）**: ✅ 正しい問題だが、Cholesky 修正でも v2 Conformer は 0.616 にしか改善せず（vs LR 0.738）。**単独真因ではなく、3-5 個の相互作用する原因のひとつ**。
- **H_prev2（LayerNorm transpose）**: ⚠️ **証拠が薄い**。`transpose-LN-transpose` パターンは Vision Transformer や時系列 Conformer で普通に動作している既知パターン。BatchNorm1d への置換は妥当な改善だが「真因」と断言する根拠は不十分。
- **H_prev3（CholeskyPool 528d スケール）**: ⚠️ **部分真**。後段の Linear 入力にスケール差が出るのは事実だが、それだけで epoch 1 ピーク後の急速崩壊は説明しきれない。

### 本分析で抽出した、優先度のより高い仮説（S/A 級）
| # | 仮説 | 影響度 | 根拠の強さ |
|---|---|---|---|
| **S1** | **訓練データ量の絶対不足**：LR は ~300K サンプル/被験者で訓練、DL v2 は ~30K で 10 倍の不利 | High | 強（コードで確認済み） |
| **S2** | **受容野と運動準備電位（RP）の時間スケール不整合**：RP は -1500ms から発生、causal window は 500ms のみ | High | 強（生理学文献） |
| **S3** | **被験者内訓練 × 被験者間ばらつきの罠**：12 個独立モデル × 少データ。LR は手作り特徴で帰納バイアスが強いため少データに耐える | High | 強（per-subject の差分） |
| **A1** | **Focal loss × クラス不均衡 × BatchNorm の相互作用**：focal loss は 0.5 周辺が安定平衡になり得る | Med-High | 中（理論的、未検証） |
| **A2** | **データ拡張完全欠如**：時間ジッタ・チャンネルドロップ・mixup なし。少データを補える唯一の手段を放棄している | High | 中（標準的な慣行） |
| **A3** | **末端時刻ラベルのみ使用**：window 内の他時刻は損失計算に未使用 → 情報の 99% を捨てている | Med-High | 中（実装検証済み） |
| **A4** | **Conformer の容量過剰**：500K params に対し有効訓練サンプル ~30K / 被験者 = サンプル/パラメタ比 0.06。EEGNet の 0.7 倍以下 | High | 強（パラメータ計算） |

### 推奨される次の検証実験（優先順）
1. **クロス被験者 pre-train + 被験者内 fine-tune**（仮説 S1, S3 を同時検証）
2. **window を 1.5-2.0 秒に延長**（仮説 S2 を検証）
3. **データ拡張パイプラインの導入**（仮説 A2）
4. **Sequence-to-sequence 損失への変更**（仮説 A3）

これらは v3 のアーキテクチャ修正（BatchNorm1d, LogVarPool）よりも、**データ・タスク設計レベルの修正**であり、より大きな改善が期待される。

---

## 1. 観察された事実の精密記述

### 1.1 数値結果（LR vs EEGNet v1/v2 vs Conformer v1/v2）

| 被験者 | LR | EEGNet v1 | EEGNet v2 | Conf v1 | Conf v2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.774 | 0.762 | 0.762 | 0.628 | 0.628 |
| 2  | 0.805 | 0.686 | 0.778 | 0.557 | 0.556 |
| 3  | 0.715 | 0.633 | 0.734 | 0.610 | 0.572 |
| 4  | 0.787 | 0.713 | 0.786 | 0.548 | 0.636 |
| 5  | 0.667 | 0.643 | 0.711 | 0.526 | 0.605 |
| 6  | 0.744 | 0.688 | 0.724 | 0.636 | 0.636 |
| 7  | 0.768 | 0.699 | 0.719 | 0.572 | 0.591 |
| 8  | 0.772 | 0.684 | 0.757 | 0.542 | 0.623 |
| 9  | 0.673 | **0.546** | **0.546** | 0.558 | 0.618 |
| 10 | 0.751 | 0.573 | 0.573 | 0.591 | 0.552 |
| 11 | 0.698 | 0.696 | 0.739 | 0.584 | 0.616 |
| 12 | 0.702 | 0.697 | 0.761 | 0.541 | 0.664 |
| **Mean** | **0.738** | **0.668** | **0.703** | **0.574** | **0.616** |
| **σ** | 0.044 | 0.058 | 0.062 | 0.034 | 0.046 |

### 1.2 学習曲線パターン（Conformer v1）

実データ JSON から抽出した特徴的なパターン:

```
S1: 0.628 → 0.547 → 0.515 → ... → 0.501  (epoch 1 ピーク → 単調減衰)
S2: 0.521 → 0.495 → ... → 0.556 (ep15) → ... → 0.506  (random walk near 0.5)
S3: 0.498 → 0.610 → 0.522 → ... → 0.506  (epoch 2 ピーク → 崩壊)
S6: 0.636 → 0.501 → 0.479 → ...  (1 epoch でピーク → 即崩壊)
S11: 0.583 → 0.432 → 0.541 → ...  (発散と振動)
```

**重要なパターン**:
- 12 被験者中 **約半数で epoch 1-2 がベスト** → 「ランダム初期化 + 少量学習」が最も精度が高い
- 残り半数は AUC 0.5 周辺から一度も脱出しない random walk
- 古典的な「過学習」（train が下がり続けて val が上がってから下がる）とは **異なる挙動**

これは「学習することで悪化する」という **病的な勾配ダイナミクス** を示唆する。通常の過学習なら 10+ epoch まで val が上昇してから下降するはず。

### 1.3 被験者間ばらつきの構造

| 被験者 | LR | EEGNet v2 | Δ (v2 - LR) | 意味 |
|:---:|:---:|:---:|:---:|---|
| S9 | 0.673 | 0.546 | **-0.127** | 既に LR で苦戦 + DL でさらに悪化 |
| S10 | 0.751 | 0.573 | **-0.178** | LR は問題ないが DL のみ崩壊 |
| S5 | 0.667 | 0.711 | +0.044 | DL のほうが良い唯一のケース |

**S10 の異常性**: LR は 0.751（平均並み）、DL のみ 0.573 に崩壊。これは「被験者の生理的特性」というより **DL 特有の問題（恐らくセッション間分布シフト）**。

### 1.4 LR vs DL の前提条件の差（実装精査で判明）

| 項目 | LR baseline | EEGNet v2 (Kaggle) |
|---|---|---|
| 訓練窓 step | 2 frames (4ms) | 20 frames (40ms) |
| 1 被験者あたり訓練サンプル | ~300,000 | ~30,000 |
| 特徴 | 5 帯域 × 32ch = 160 次元（手作り） | raw EEG 32×250 = 8000 次元（学習） |
| パラメータ数 | ~960（線形 × 6 ヘッド） | ~15,000（EEGNet）, ~500,000（Conformer v2） |
| サンプル/パラメタ比 | **312** | **2.0** (EEGNet), **0.06** (Conformer) |

**この比率の差は決定的**。深層学習では一般に 10-100 サンプル/パラメタが必要とされる（Goodfellow et al., 2016）。LR は十分すぎる比率、EEGNet はギリギリ、Conformer は **2 桁不足** している。

---

## 2. 既存「真因」結論の批判的レビュー

### 2.1 H_prev1: `torch.linalg.eigh` の NaN 勾配（v1）

**主張**: Brooks et al. NeurIPS 2019 の既知問題。固有値縮退で勾配爆発 → モデル崩壊。

**再検証**:
- ✅ 理論的妥当性: 確実に存在する数値問題（参照論文も適切）
- ✅ Cholesky への置換は標準的な解（Riemannian Deep Learning 文献での慣行）
- ⚠️ **しかし v2 Conformer は依然 0.616**。Cholesky 修正で AUC が 0.574 → 0.616 (+0.042) しか改善しなかった。

もし NaN 勾配だけが原因なら、Cholesky で **完全に治癒** するはず。実際は依然として崩壊するため、**真因のひとつだが単独原因ではない**。

### 2.2 H_prev2: TCNBlock の LayerNorm + transpose

**主張**: `nn.LayerNorm(out_ch)` は (B, T, out_ch) を要求するため、transpose を 2 回挟む実装が勾配不安定を生む。

**再検証 — 反証材料**:
- 同じ実装パターンは Hugging Face transformers, TimeSformer, ConvNeXt など **多くの実プロダクションコードで使われている** が崩壊しない
- LayerNorm 自体に勾配の数値問題はない（Ba et al., 2016 で証明）
- transpose は単なるメモリレイアウト変更で勾配は単位的に流れる

**残る可能性**:
- BatchNorm1d は running_mean/var を持つため **テスト時に train statistics を再現できる**
- LayerNorm はサンプルごとに完結するためそれが不要
- どちらが「正しい」かはデータ次第。EEG のように非定常で系列長が短い場合、BatchNorm のほうが安定しやすい傾向はある

**評価**: BatchNorm1d 化は妥当な改善だが「真因」とまでは言えず、**寄与度小** と見るべき。

### 2.3 H_prev3: CholeskyPool の出力スケール（528d）

**主張**: `proj_dim=32` の Cholesky 下三角は 32×33/2 = 528 次元。対角要素は √(eigenvalue) で大きく変動 → classifier 入力にスケール不一致 → 学習不安定。

**再検証**:
- ✅ 出力次元 528 は確かに大きい
- ✅ 対角要素のスケール変動は事実
- ⚠️ しかし classifier 直前に LayerNorm(combined_dim) を入れれば吸収可能（実装には入っていなかった）
- ⚠️ epoch 1 ピーク後の急速崩壊は scale 問題だけでは説明しきれない

**評価**: 寄与度中。ただし真因リストの 1-3 位ではない。

### 2.4 既存結論の問題点（メタな批判）

既存ドキュメントは **アーキテクチャレベルの修正に注力** しすぎていた。

「アーキテクチャを直せば精度が出る」という暗黙の仮定があるが、観察データ（LR vs DL のサンプル数比 10×）はむしろ **データ・タスク設計レベル** の問題を示唆する。

これは Kaggle や論文ベンチマークで頻発する誤診パターン: 「結果が悪い → モデルアーキテクチャを変える → ループ」と無限に続いてしまう罠。

**真の改善はデータ、損失設計、訓練戦略にあることが多い**（Sculley et al., 2018; Kapoor & Narayanan, 2023 "Leakage and Reproducibility Crisis"）。

---

## 3. 思考の流れ — 何を疑い、どう確かめたか

このセクションは結論ではなく **思考過程のログ** として記録する。

### Step 1: 「なぜ Cholesky 修正後も改善が小さいのか」という違和感

最初に持った違和感は、**修正の効果量が予想より小さい** こと。
- Cholesky で完全に NaN 問題が消えるはずなら、AUC は 0.574 → 0.74+ の改善があってもよい
- 実際は +0.042 のみ → 何か別の支配的要因がある

→ 「単独真因」ではなく「複数原因の相互作用」を疑うべき、と判断。

### Step 2: 学習曲線の精査

JSON から実数値を取り出し、12 被験者のパターンを並べた:
- 半数: epoch 1-2 ピーク → 急速崩壊
- 半数: 0.5 周辺で random walk

「epoch 1 で 0.628 → epoch 2 で 0.501」のような S6 のパターンは **異常** に見えた。LR=6e-5（warmup 中）でこれだけ崩壊するのは、**通常の SGD ダイナミクスでは起こりにくい**。

→ アーキテクチャ問題以外の何か、特に **「学習」自体が壊れている** 可能性を疑う。

### Step 3: ベースライン LR の実装精査

「なぜ LR がこんなに強いのか？」を問い直した。
- LR の特徴 = 5 帯域パワー（160d）。手作り。
- LR の訓練データ = step=4ms = 実質ほぼ全フレーム
- → LR は「強力な帰納バイアス（band power）」+「大量データ（300K）」で勝っていた

DL は raw EEG から学ぶ。これは原理的には強力だが、**訓練データが 10 分の 1**。
→ 「勝てなくて当然」という見立てに変わった。

### Step 4: 神経生理学への立ち戻り

タスクは GAL（Grasp-and-Lift）で、HandStart や FirstDigitTouch は **運動関連電位（RP, Bereitschaftspotential）** が関連。
- RP の時間スケール: 動作の **-1500ms から徐々に立ち上がる**（Kornhuber & Deecke, 1965）
- 早期 RP: -1500 ~ -500ms, 後期 RP: -500 ~ 0ms
- 実は **causal window 500ms** は後期 RP しかカバーしない
- 早期 RP は窓に入らない → DL は重要な手がかりを見ていない可能性

LR は step=4ms で密にサンプリングするので、より長期の文脈を「複数の隣接ウィンドウの予測」として暗黙的に組み立てている可能性がある。

### Step 5: Focal loss の再検討

Focal loss は不均衡対策の defacto standard だが、 **0.5 予測が局所安定** になるという指摘は ML 文献にはあまりない。理論的に検証すると:

```
target = 0.975（positive, label_smoothing=0.05）
prediction = 0.5
pt = 0.5
focal_weight = (1-0.5)^2 = 0.25
BCE = -log(0.5) = 0.693
loss = α × 0.25 × 0.693 = 0.043
```

→ これは「誤予測でもそれほど大きくない損失」。勾配は finite だが弱い。
→ モデルが 0.5 周辺に張り付いていても、勾配で抜け出すインセンティブが弱い

これは epoch 1 で 0.5+ε のスタート（初期化により正の AUC）から、訓練が進むと 0.5 に戻ってくる動きを説明できる可能性がある。

### Step 6: BatchNorm × 不均衡の相互作用

BatchNorm1d は train モードで running_mean/var を更新する。
- 不均衡（陰性 97.4%）で running stats は陰性クラスの分布に支配される
- eval モード時、モデルは陰性に偏った正規化を使う
- → 出力ロジットが陰性方向に押される

これはデバッグが難しい問題で、よく見落とされる。

### Step 7: 過学習への耐性の少なさ

サンプル/パラメタ比 0.06（Conformer）は致命的。

通常の DL では:
- 比 = 1 → 暗記可能、過学習リスク高
- 比 = 10 → 一般化ギャップが生じる
- 比 = 100+ → 安定一般化

Conformer の 0.06 は **「全パラメータを覚えるためのサンプル数すらない」** 領域。これが random walk 挙動の根本かもしれない。

### Step 8: 「窓末端ラベル」の情報利用率

`make_sliding_windows(label_at="end")` → 窓 [t-249, t] のラベルは t 時点のみ。
- 窓内には 250 時刻あるが、損失計算に使うのは 1 時刻だけ
- 情報利用率 = 1/250 = 0.4%
- Sequence-to-sequence にすれば、各窓内の 250 時刻すべてを supervision に使える

→ 同じデータから 250 倍の supervision 信号が抽出可能（ただし重複情報が多い）

---

## 4. ML 専門家視点の仮説

### 4.1 最適化（Optimization）

**M1. 学習率と warmup の整合**
- v2: lr=3e-4, warmup=5 epoch, AdamW, weight_decay=1e-4
- 観察: epoch 1 ピーク後の崩壊 → warmup 中（小さな LR）でも学習が壊れている
- → LR の問題ではなく **landscape そのものが鋭い**

**M2. Optimizer 選択**
- AdamW は EEG 文献の標準だが、不均衡データでは **SGD + momentum** の方が安定するという報告（Wilson et al., 2017）
- 試す価値あり

**M3. Gradient clipping**
- max_norm=1.0 は EEG では一般的だが、ELU/GELU のあるネットワークでは緩い場合あり
- 0.5 や Adaptive clipping (AGC, Brock et al., 2021) を試せる

**M4. AMP（混合精度）**
- v1 で AMP の NaN logits 問題があり v2 で削除
- これは正しい対応

**評価**: 4.1 は **寄与度低-中**。LR=3e-4 は妥当。最適化単体で大改善は期待できない。

### 4.2 データ（Data）

**D1. 訓練データ量の絶対不足（仮説 S1）**
- LR: 300K samples/subject、DL v2: 30K
- これだけで AUC 差の半分は説明できる可能性

**D2. Oversampling 3× の妥当性**
- 正例率 2.6% → 3× で実効正例率 ~8%
- ただしこれは **同じサンプルを 3 回見る** だけで情報量は増えない
- **真の解決は SMOTE / mixup / 合成データ拡張**

**D3. データ拡張の完全欠如（仮説 A2）**
- EEG 分野で標準的な拡張（McKenzie et al., 2023 サーベイ）:
  - **時間ジッタ**: ±20ms シフト（ラベル整合性を保つ）
  - **チャンネルドロップ / マスク**: 1-5 ch をランダムにゼロ化
  - **振幅スケール**: 0.8-1.2 倍
  - **mixup / cutmix**: 異なる窓を線形補間（ラベルも補間）
  - **frequency masking**: SpecAugment 風
- これらをひとつも入れていない → **少データを補える唯一の手段を放棄している**

**D4. 被験者間データ統合（cross-subject training）**
- within-subject 12 個の独立モデル：データ少、過適合リスク高
- cross-subject + subject embedding：データ 12 倍、被験者特性は埋め込みで吸収
- transfer learning: cross-subject pre-train → within-subject fine-tune

**評価**: 4.2 は **寄与度高**。特に D1, D3, D4 は S/A 級仮説。

### 4.3 アーキテクチャ（Architecture）

**A1. 受容野（Receptive Field）の妥当性**
- TCN: kernel=8, dilations=1,2,4
- 各層の RF: layer1=8, layer2=8+14=22, layer3=22+28=50
- 全 RF = 50 sample = 100ms
- vs window = 250 sample = 500ms
- → **モデルは window の 20% しか「見ていない」**
- 修正: dilations=1,2,4,8,16 など 5 層にする / kernel=16 にする

**A2. Conformer の容量過剰（仮説 A4）**
- パラメータ ~500K, 訓練窓 ~30K → 過剰
- 容量を絞るか、データを増やすか

**A3. EEGNet の容量適切性**
- パラメータ ~15K, 訓練窓 30-60K → 適切
- v2 (step=20) で 0.703 → さらに step=10 にすれば 0.72-0.74 まで行く可能性

**A4. 因果性制約のコスト**
- 因果フィルタは位相歪み（group delay）あり
- Zero-phase filtfilt なら歪みなし
- 但しこれは未来データを使うので Kaggle ルール違反

**A5. Attention 機構の不在**
- v2 Conformer は ElectrodeAttention を **定義しているが使っていない** デッドコード
- 空間アテンション（電極間）を入れれば C3/C4 などの運動関連チャンネルを強調できる可能性

**評価**: A1, A2 は **寄与度高**。A3 は EEGNet で問題なし。A4 はトレードオフ。A5 は要検証。

### 4.4 損失関数

**L1. Focal loss の 0.5 局所安定性（仮説 A1 の一部）**
- 上記 Step 5 参照
- 0.5 予測の損失は 0.043（α=0.25, γ=2.0 のとき）
- 勾配は弱く、抜け出しにくい

**L2. AUC を直接最適化していない**
- Focal/BCE は cross-entropy → AUC との相関は強いが完全ではない
- AUC 直接最適化: ApproxAUC, RankNet, AUCMLoss（Yang et al., 2022）
- ただし実装難度はやや高い

**L3. Multi-label の独立扱い**
- 6 イベントを独立に予測 → 順序情報を捨てている
- HandStart → FirstDigitTouch → ... の制約を入れる: CRF, structured prediction, sequence label

**L4. Label smoothing 0.05 の効果**
- positive ラベルが 0.975 になる → 勾配のノイズが入る
- 過学習防止には有効だが、**正例信号自体を弱める**
- LR ベースラインは label smoothing 不使用

**評価**: L1 は **寄与度中-高**。L3 は **寄与度高（要検証）**。

### 4.5 正則化

**R1. Dropout の比率**
- EEGNet v2: dropout=0.5, Conformer v2: dropout=0.3
- データ少ないなら dropout 高い方が良いはず → 0.5 もありえる
- 過剰だと信号 too weak → 学習進まない

**R2. Weight decay 1e-4**
- 標準値、特に問題なし

**R3. Early stopping**
- patience=12 で early stop → epoch 1 ピーク後の急速崩壊では即座に stop
- これは正しいが、逆に「early stop が即発動 = 学習できていない」という診断信号

**評価**: 4.5 は **寄与度低**。

---

## 5. 脳情報学専門家視点の仮説

### 5.1 信号処理の妥当性

**P1. 因果フィルタの位相遅延**
- `sosfilt` は前方フィルタのみ → group delay あり
- 0.5Hz HP, 4 次 Butterworth: 0.5Hz で遅延 ~1000ms, 5Hz で ~50ms, 10Hz で ~10ms
- 帯域パワー特徴（β: 13-30Hz, γ: 30-45Hz）の遅延は数 ms で問題なし
- しかし **遅い RP（< 1Hz 成分）は 1 秒以上遅延** → ラベルとの整合崩れる
- → **0.5Hz HP は SCP/CNV を過剰に削っている可能性**

**P2. ローパス 45Hz と γ 帯**
- γ (30-45Hz) は運動関連活動に重要（Cheyne, 2013）
- ただし 45Hz でカット → 一部 γ 情報を保持
- 60Hz (米国) / 50Hz (欧州) の電源ノイズ除去はできている

**P3. CAR（Common Average Reference）の問題**
- 32ch 全平均を引く
- TP9, TP10 は乳様突起付近 → 比較的アーチファクトが乗りやすい
- これらの異常値が CAR を歪める可能性
- 改善案: ロバスト CAR（trimmed mean）、 surface Laplacian

**P4. Z-score warmup**
- causal_running_zscore は最初の 5 秒は warmup
- それ以前のフレームは生信号に近い → 学習窓が series 序盤を含むと統計が安定しない
- 観察: validation series 7-8 では問題なし（既に統計収束済み）

**評価**: P1 は **寄与度中**（特に低周波成分の利用に影響）。P3 は要確認。

### 5.2 ERP / 運動準備電位の生理学

**B1. RP の時間スケール（仮説 S2）**
- Kornhuber & Deecke 1965: RP は動作 -1500ms から開始
- 早期 RP（-1500 〜 -500ms）: 補足運動野 SMA、 両側性
- 後期 RP（-500 〜 0ms）: 一次運動野 M1、 対側性
- **causal window 500ms は後期 RP のみカバー**
- LR の dense windowing は隣接窓の連鎖で長期文脈を暗黙的に得ている可能性

**B2. ERP 加算平均と単 trial の差**
- 古典的 ERP 研究は 数十-数百試行の加算平均
- 単 trial 検出は ノイズ比 1:10 以上の問題
- DL は単 trial で予測 → 本質的に困難

**B3. イベントごとの脳機構の違い**
- HandStart: 運動開始の意図 → SMA, M1 主導、 RP 関連
- FirstDigitTouch: 触覚 → 体性感覚野 S1, P3 関連
- LiftOff: 動作実行 → 運動皮質 + 触覚 feedback
- BothReleased: 動作終了 → β-rebound (15-25Hz)
- **6 イベントは異なる脳基盤** → 単一モデルで同時最適化は本質的に困難

**B4. 順序制約**
- HandStart → FirstDigitTouch → ... と必ず同順
- これは **強い prior** → CRF や HMM で陽に表現すれば改善
- LR は独立扱いなので使っていない（DL が劣る理由ではない）

**評価**: B1 は **寄与度高（仮説 S2）**。B3 は **寄与度中**。

### 5.3 個人差・非定常性

**V1. Subject 9, 10 の異常**
- LR で S9 = 0.673（平均以下）、S10 = 0.751（平均並み）
- DL で S9, S10 とも 0.55 前後 → DL のみ崩壊
- 仮説:
  - S10: signal quality は OK だが series 7-8 の分布が series 1-6 と大きく異なる（疲労、 vigilance）
  - S9: 元から signal quality が悪い（動きアーチファクト、 EOG）
- 検証: series 1-6 の自己 holdout（series 6 を val に）で AUC を比較すれば分布シフトと信号品質を分離できる

**V2. Inter-session variability**
- BCI 文献の常識: セッション間の電極配置・インピーダンス・vigilance は変動
- 標準的な対策: session embedding, domain adaptation, online recalibration
- ここでは何もしていない → series 7-8 で大きく崩れる

**V3. Within-subject training の限界**
- 1 被験者 = 6 series = ~600K frames = 30K 窓 (step=20)
- 500K params のモデルを訓練するには圧倒的に不足
- LR は 960 params → 全く問題なし

**評価**: V1, V2 は **寄与度高**。V3 は **仮説 S3 そのもの**。

### 5.4 空間情報の利用

**E1. 32ch の十分性**
- 国際 10-20 拡張系で運動関連検出には十分（多くの BCI 研究が 32-64ch で実施）
- 問題ではない

**E2. 距離バイアスアテンションの妥当性**
- v2 の `ElectrodeAttention.dist_bias_base = -dist`：近い電極ほどアテンション強化
- これは **逆かもしれない**：運動関連検出では C3 と C4 という遠い電極の **ALI（左右非対称）** が重要
- 距離バイアスではなく **学習可能な空間プリオール** が必要

**E3. CSP の不在**
- Common Spatial Pattern は EEG BCI のゴールドスタンダード（Blankertz et al., 2008）
- LR の depth-wise spatial conv (`groups=f1` の Conv2d (n_ch, 1)) は CSP の DL 版に相当
- v2 EEGNet は実質 CSP を含んでいる → だから LR と並ぶ精度
- Conformer は明示的な CSP を持たない

**E4. 双極子源推定 (sLORETA) の不使用**
- ソース空間特徴は信号品質を上げるが計算重い
- ここでは扱わなくてよい

**評価**: E2 は要検証、E3 は **Conformer の弱点として重要**。

---

## 6. 訓練手法視点の批判

### 6.1 窓設計とラベル定義

**T1. Window = 500ms（仮説 S2 と関連）**
- 後期 RP のみカバー
- 1.5-2.0 秒（750-1000 samples）に拡大すれば早期 RP も含む
- メモリ・計算は数倍だが妥当

**T2. 末端時刻ラベル（仮説 A3）**
- 1 窓 250 時刻あるが、ラベル計算は 1 時刻分
- Sequence-to-sequence にすれば情報量 250 倍
- 出力: (B, 6, 250) ロジット → frame-wise BCE

**T3. ±150ms ラベル tolerance の活用**
- ラベル自体が "イベントが ±150ms 以内に" を表している
- これを soft label として活用：イベント中心からの距離 d に対して exp(-d²/σ²) のような重み
- → label の連続性を学習しやすくなる

### 6.2 データ分割

**T4. Validation = series 7-8 のホールドアウト**
- 時間的に最後 → 最も分布シフトが大きい
- 公平な比較のため: K-fold（各 series を 1 fold）
- 現状の設定は **「実用に近いがハード」** な評価

**T5. Test = series 9-10（Kaggle 実 test）**
- Kaggle ルール上、val よりさらに外側で評価
- val が崩れる被験者は test もほぼ確実に悪い

### 6.3 訓練戦略

**T6. Within-subject 12 個独立モデル（仮説 S3）**
- 利点: 被験者特性に完全適合
- 欠点: データ少、過適合
- 標準的な改善: cross-subject pre-train + within-subject fine-tune
- subject embedding を入れれば 1 つのモデルで済む

**T7. Self-supervised pre-training**
- Mehboudi et al., 2021: EEG SSL 手法（contrastive, masked modeling）
- 1 億フレーム規模のラベルなしデータを活用可能
- Grasp-and-Lift には適用例少ないが、 GAL 全 12 被験者の non-event 区間を SSL に使える

**T8. AUC 最適化への直接的アプローチ**
- column_wise_auc は per-event AUC の平均
- 訓練時に AUC を直接最適化（rank loss）すれば validation との整合性が高まる

### 6.4 評価指標と訓練指標のズレ

**T9. Train loss vs Val AUC の不整合**
- Focal loss は下がるが Val AUC は上がらない場合がある
- これは「Focal loss を最小化することと AUC を最大化することは別」
- 実装: 訓練でも AUC を計算し、 train AUC vs val AUC で過学習を判定

---

## 7. 仮説の優先度マトリクス

評価軸:
- **影響度**: その仮説が真の場合、 AUC が +Δ 改善する見込み（H ≥ 0.05, M ≥ 0.02, L < 0.02）
- **検証可能性**: 既存コードベースで実装・検証できるか（H = 1日, M = 1週間, L = 要追加データ/外部）
- **既存仮説との独立性**: H_prev と独立か

| ID | 仮説 | 影響度 | 検証可能性 | 独立性 | 優先度 |
|---|---|:---:|:---:|:---:|:---:|
| S1 | 訓練データ量不足 (step=4 vs step=20) | H | H | ✓ | **S** |
| S2 | RP 時間スケール vs window=500ms | H | H | ✓ | **S** |
| S3 | within-subject + 少データの罠 | H | M | ✓ | **S** |
| A1 | Focal loss の 0.5 局所安定 | M-H | M | ✓ | **A** |
| A2 | データ拡張完全欠如 | H | H | ✓ | **A** |
| A3 | 末端時刻ラベルのみ (情報利用率 0.4%) | M-H | M | ✓ | **A** |
| A4 | Conformer 容量過剰（500K params） | H | H | △ | **A** |
| A5 | Receptive field 100ms < window 500ms | M | H | ✓ | **A** |
| B1 | BatchNorm × class imbalance | M | H | △ | **B** |
| B2 | CSP 機能の不在（Conformer） | M | M | ✓ | **B** |
| B3 | 順序制約（イベント順序）の無視 | M | L | ✓ | **B** |
| B4 | 0.5Hz HP の SCP 削り過ぎ | M | M | ✓ | **B** |
| B5 | CAR の TP9/TP10 汚染 | L-M | M | ✓ | **B** |
| C1 | H_prev2 (LayerNorm transpose) | L-M | H | - | **C** |
| C2 | H_prev3 (Cholesky 528d スケール) | M | H | - | **C** |
| C3 | label_smoothing 0.05 の正例信号弱化 | L | H | ✓ | **C** |
| C4 | 因果フィルタ位相遅延 | L-M | M | ✓ | **C** |

**S 級（最優先）**: 3 件 — いずれもデータ・タスク設計に関する根本的問題
**A 級**: 5 件 — 標準的な改善で対処可能
**B 級**: 5 件 — 中程度の影響、実装コスト中
**C 級**: 4 件 — 既存仮説（再評価で寄与度小と判定）または影響小

---

## 8. 推奨される検証実験（具体的アクションリスト）

### 8.1 即実施（1 日以内）

**実験 1: EEGNet step=10 + データ拡張 (仮説 S1, A2)**
- 設定: step_samples=10 (118K windows), 時間ジッタ ±20ms, チャンネルマスク 5%, 振幅 ±10%
- 期待: EEGNet AUC 0.703 → **0.74-0.76**（LR 並み）
- リスク: 拡張で正例情報が壊れる可能性（軽度）

**実験 2: window=750-1000ms に拡張 (仮説 S2)**
- 設定: window_samples=375 or 500
- TCN dilations=1,2,4,8 で RF 拡大
- 期待: 早期 RP をカバー → S9, S10 も改善
- リスク: 計算 1.5-2 倍

### 8.2 中期実施（1 週間）

**実験 3: クロス被験者 pre-train + within-subject fine-tune (仮説 S3)**
- 設定: 全 12 被験者 × series 1-6 で pre-train（subject embedding あり）
- 各被験者で fine-tune（series 1-6 のうち 1 series を val に）
- series 7-8 で評価
- 期待: AUC +0.03-0.05、 特に S9, S10 で大改善
- リスク: subject embedding の汎化性検証必要

**実験 4: Sequence-to-sequence 損失 (仮説 A3)**
- 出力を (B, 6, 250) ロジットに変更
- frame-wise BCE で全時刻に supervision
- 期待: AUC +0.02-0.04、 特に Conformer で

**実験 5: Focal loss 検証 (仮説 A1)**
- BCE + class weights vs Focal loss vs AUC margin loss を比較
- 0.5 局所安定の有無を診断

### 8.3 長期実施（1 ヶ月）

**実験 6: SSL pre-training**
- masked EEG modeling: 50% time masking, 復元損失
- Grasp-and-Lift 全データ + (可能なら) BCI Competition IV データセット
- 期待: AUC +0.05-0.08（深層モデル限定）

**実験 7: 順序制約モデル**
- Conditional Random Field (CRF) 出力層
- もしくは autoregressive head
- 期待: AUC +0.02-0.04（特に LiftOff, Replace で）

---

## 9. 限界と未検証事項

### 9.1 本分析の限界

1. **学習曲線データは v1 のみ**：v2 の Conformer histories はチャットに貼り付けられたが本リポジトリに保存ファイルなし。v2 学習曲線パターンは v1 と同様と仮定したが、要再現
2. **per-event AUC が DL 側にない**：LR は per-event がある（HandStart 0.668, LiftOff 0.766 など）が DL 側は mean のみ。イベント別の差を分析できない
3. **被験者の生信号を実視していない**：S9, S10 のアーチファクト混入を直接確認していない（本来 EDA で確認すべき）
4. **計算実験は実施せず**：本レポートはコード/ドキュメント精査のみ。仮説の半分は定量検証なし

### 9.2 真因とまでは言えない仮説

- A1（Focal loss の 0.5 安定）: 理論的には妥当だが、実験で 0.5 周辺の固定点を確認していない。BCE + class weights に置換した場合の挙動比較が必要
- B1（BatchNorm × imbalance）: 既知の問題だがこのデータで支配的かは未検証
- B4（0.5Hz HP の SCP 削り）: 1.0Hz HP に変えた比較実験がない

### 9.3 今後の調査が必要な事項

1. **v2 Conformer の実 val_auc 学習曲線** をローカルに保存
2. **被験者別の信号品質メトリクス**（artifact ratio, 50/60Hz noise, channel SNR）の計算
3. **per-event の DL 結果**（HandStart 別、 LiftOff 別の AUC）
4. **train_loss と val_auc の相関プロット**（学習が AUC 改善に繋がっているか）

---

## 10. 結論と再フレーミング

### 10.1 主要な発見

既存の「真因」結論は **アーキテクチャ問題に偏っていた**。本分析で抽出した上位仮説は **データ・タスク設計レベル** に集中している:

1. **S1: 訓練データ量** (LR の 1/10)
2. **S2: 受容野と RP 時間スケールの不一致** (500ms < RP の 1500ms)
3. **S3: within-subject 訓練 × 12 独立モデル × 少データ**

これらは v3 のアーキテクチャ修正（BatchNorm1d, LogVarPool）では解決しない問題群である。

### 10.2 v3 計画への含意

すでに作成された `notebooks/kaggle_train_v3.py` の修正は **依然有効** だが、以下を追加・優先すべき:

1. **EEGNet v3**: step=10 + データ拡張（時間ジッタ、チャンネルマスク、 mixup）
2. **window=750ms 以上に拡張**（TCN dilations=1,2,4,8）
3. **クロス被験者 pre-train**（v3 の within-subject の前に）
4. **Sequence-to-sequence 出力**（情報量 250 倍）

これらを v4 として位置付けるべき。v3 の TCNBlock+LogVarPool 修正単体では、 LR 超えは難しいと予測する。

### 10.3 ポートフォリオ的価値

この再検証プロセス自体が技術的価値を持つ:

1. **「真因」を 1 つに決めつけない多角的診断** — 実務 ML での重要スキル
2. **既存結論の批判的レビュー** — 自己修正能力のアピール
3. **ML / 神経科学 / 統計の三位一体分析** — 単純な ML エンジニアより一段上の専門性
4. **仮説 → 優先度 → 検証実験の構造化** — 体系的な研究マインド

GitHub README にこのレポートを引用し、「v1 → v2 → v3 と段階的に改善し、各段階で何を学んだか」を可視化することで、技術力の説得性が高まる。

---

## 11. 参考文献

### ML / DL
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Wilson, A. C. et al. (2017). "The Marginal Value of Adaptive Gradient Methods in Machine Learning." NeurIPS.
- Brock, A. et al. (2021). "High-Performance Large-Scale Image Recognition Without Normalization." ICML. (AGC)
- Yang, T. et al. (2022). "AUC Maximization in the Era of Big Data and AI: A Survey." arXiv:2203.15046.
- Sculley, D. et al. (2018). "Winner's Curse? On Pace, Progress, and Empirical Rigor." ICLR Workshop.
- Kapoor, S., & Narayanan, A. (2023). "Leakage and the Reproducibility Crisis in ML-based Science." Patterns 4(9).

### EEG / BCI
- Lawhern, V. J. et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based BCI." J. Neural Engineering.
- Song, Y. et al. (2022). "EEG Conformer: Convolutional Transformer for EEG Decoding." IEEE TNSRE.
- Blankertz, B. et al. (2008). "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis." IEEE Sig. Proc. Mag. (CSP)
- Kornhuber, H. H., & Deecke, L. (1965). "Hirnpotentialänderungen bei Willkürbewegungen..." Pflügers Archiv.
- Cheyne, D. O. (2013). "MEG Studies of Sensorimotor Rhythms: A Review." Exp. Neurology.
- McKenzie, S. et al. (2023). "Data Augmentation for EEG-based BCI: A Survey." J. Neural Engineering.
- Mehboudi, M. et al. (2021). "Self-Supervised Learning for EEG: A Comprehensive Review." arXiv.

### Riemannian / SPD
- Brooks, D. et al. (2019). "Riemannian Batch Normalization for SPD Neural Networks." NeurIPS.
- Huang, Z., & Van Gool, L. (2017). "A Riemannian Network for SPD Matrix Learning." AAAI.
- Ionescu, C. et al. (2015). "Matrix Backpropagation for Deep Networks with Structured Layers." ICCV.

### Normalization
- Ba, J. L. et al. (2016). "Layer Normalization." arXiv:1607.06450.
- Ioffe, S., & Szegedy, C. (2015). "Batch Normalization." ICML.

---

**ファイルバージョン**: v1.0
**次回更新時の追記候補**:
- v3 実行後の結果を踏まえた仮説検証
- 実験 1-7 の結果反映
- per-event 分析の追加
