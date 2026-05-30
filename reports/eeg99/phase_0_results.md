# Phase 0 Results Report

**Phase**: 0 — 基盤構築 + 文献集中レビュー  
**完了日**: 2026-05-17  
**担当セッション**: 2026-05-14 (初回) + 2026-05-17 (完了)

---

## 達成度サマリー

| 成果物 | ステータス | 補足 |
|---|:---:|---|
| `src/eeg99/` パッケージ骨格 | ✅ 完了 | 57 tests all passing |
| `docs/literature_review_master.md` | ✅ 完了 | 15 本 + §16 Synthesis + §17 Open Questions |
| Cat & Dog solution 解読 | ✅ 完了 | 3-level pipeline 構造を文書化 |
| master plan v1.1 refine | ✅ 完了 | Open Questions 5件追加 |
| Phase 1 入口仕様確定 | ✅ 完了 | 最初のアクション 5件特定 |

---

## 数値ベンチマーク (Phase 0)

| 指標 | 値 | 備考 |
|---|:---:|---|
| eeg99 package import tests | 57/57 pass | 全 49 module + utility tests |
| 文献数 | 15 本 | 全て ArXiv ID or DOI 確認済み |
| Adoption 判断 | ADOPT:6 / ADAPT:4 / REFERENCE:5 | §0 テーブル参照 |
| GPU 消費 | 0h | Phase 0 はコード/文書のみ |

---

## Key Findings (Phase 0 で判明した重要事項)

### F1: Cat & Dog の因果差分
Cat & Dog の AUC 0.981 は **非因果 window** を含む設計 (XDawn ERP: イベント前後の窓を全期間平均)。我々の因果版では 0.02-0.03 の差が生じる可能性がある。**因果性を保ちつつ 0.97+ を達成することが、むしろポートフォリオ的に強い主張**になる。

### F2: Cat & Dog の feature set
3 種の特徴が核心:
1. Riemannian tangent space features (covariance → SPD manifold → Euclidean)
2. Filter bank features (10 lowpass: 0.5-30 Hz → concat)
3. FBL_delay features (5 past samples × 200-sample stride = 2 sec history)

→ 我々の Component 2 (FBCSP), Component 3 (Riemannian) が直接対応

### F3: Ensemble 構造の確信
50+ models を 3-level ensemble で統合した設計は、我々の Component 8-9 の設計を支持。「多様性」の重要性が文献からも確認された。単一アーキテクチャの改善より ensemble diversity の方が AUC 伸長に効果的。

### F4: SubjectAdaptiveFiLM の独自性確認
FiLM の EEG への適用は先行研究で確認されず。subject embedding による conditional normalization は本プロジェクトの独自貢献として主張可能。

### F5: CausalBENDR の設計制約
BENDR は双方向 Transformer を使用。因果版では「未来 token のみを mask & 予測」する設計が必要。これは BENDR の情報量を制限するため、Phase 3 で小規模 ablation が必須。

---

## 仮説検証ステータス

| 仮説 | Phase 0 での判断 | 根拠 |
|---|:---:|---|
| Ensemble 重視が正しい | **支持** | Cat & Dog の 50+ model ensemble が 0.981 に必要 |
| FiLM が被験者差を解決できる | **独自性確認、未検証** | 先行研究なし、Phase 2 で検証 |
| SSL pre-training が有効 | **支持 (他ドメイン)** | BENDR は大規模 EEG で有効; motor imagery への転用は Phase 3 で |
| Hierarchical Conformer が v2 崩壊を解決 | **設計根拠あり、未検証** | Song 2022 + HCFT 2025 の設計を採用; Phase 2, W4 で検証 |

---

## Phase 1 への引継ぎ

### 優先タスク (W2)

1. **data/loader.py 実装**: CSV → SubjectSeries DataFrame → parquet cache
   - 入力: `train/subj{N}_series{M}_data.csv` + `*_events.csv`
   - 出力: `SubjectSeries(subject_id, series_id, eeg, events, sfreq=500)`

2. **data/preprocess.py 実装**: causal signal conditioning
   - causal HP filter: fc=0.5 Hz (Butterworth order=5, causal)
   - causal LP filter: fc=45 Hz (Butterworth order=5, causal)
   - CAR (Common Average Reference)
   - causal running z-score (window=5s)

3. **data/filterbank.py 実装**: 5 帯域並列 causal filterbank
   - delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-13 Hz, beta: 13-30 Hz, gamma: 30-45 Hz

4. **data/dataset.py 実装**: multi-window PyTorch Dataset
   - window sizes: 125, 250, 500, 1000, 2000 samples (0.25s - 4s)
   - subject_id を各 sample に付与

5. **data/cache.py 実装**: parquet キャッシュ
   - 前処理済みデータをキャッシュ、hash で無効化

### 成功基準 (Phase 1 完了時)
- `tests/eeg99/test_data.py`: 90%+ coverage
- 全 12 被験者 × 8 series の読み込みが < 30 秒
- causal-leak テスト: 前処理後の各サンプルが未来情報を含まないことを検証

---

## 反省・学習事項

1. **文献調査の効率**: WebSearch を並列実行することで 15 本を 1 セッションで確認可能だった。今後も並列化を積極活用。
2. **Cat & Dog の非因果性**: 最初から「非因果」を想定すべきだった。今後は文献採用時に因果性チェックを最初に行う。
3. **Codebase 骨格の先行構築**: Phase 0 で骨格を先に作ったことで、以降のセッションで「どこに書くか」が明確になる。良い判断だった。

---

**レポートバージョン**: 1.0  
**次回更新**: Phase 1 完了時
