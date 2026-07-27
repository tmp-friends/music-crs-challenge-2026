# docs/final_submission — 最終提出アーキテクチャまとめ

RecSys Challenge 2026 Music-CRS の最終提出
（`submission_exp116_lgbm6_nocs_B.zip`、Blind B composite 0.4998 / Industry Track 8 位・全体 16 位）
の実装ドキュメント置き場。論文執筆・code submission の技術リファレンスとして使う。

## このフォルダの md

| ファイル | 内容 |
|---|---|
| [architecture.md](architecture.md) | 最終提出の全体像。3 段パイプライン（BM25 アンカー → 12 source + 73 次元特徴量 + LightGBM LambdaRank 6-seed → Claude response 生成）、Blind B 3 提出の比較、cross_session 除外判断、再現コマンド |
| [anchor_pipeline.md](anchor_pipeline.md) | 第 1 段「BM25 アンカー」の詳細。8 lexical source → RRF → 1 段目 LightGBM の構成と、2 段目で primary として保持される設計の意図 |
| [candidate_sources_train_sample_walkthrough.md](candidate_sources_train_sample_walkthrough.md) | 補助 candidate source 4 系統（continuity / QNM / entity / transition）を train 実サンプル 1 session で walkthrough。GT が各 source のどこに入るかを実測 |
| [qa_and_posthoc_diagnostics.md](qa_and_posthoc_diagnostics.md) | 実装詳細（stage-1/2 の source 表・特徴量・label・学習設定）+ 設計 Q&A 7 本（primary/source 分離の理由、top-20 の根拠、Dev で学習する理由 等）+ コンペ終了後の事後診断実験（primary 深さ / label 学習曲線 / primary 廃止 flattened 構成 / embedding 特徴量 / Blind B cold session トレース） |

## 関連ドキュメント（フォルダ外）

- 評価整合性監査論文: [docs/paper/paper_evaluation_alignment_audit.md](../paper/paper_evaluation_alignment_audit.md) /
  [同 日本語版](../paper/paper_evaluation_alignment_audit_ja.md)
- 評価指標と composite の定義: [docs/evaluation_framework.md](../evaluation_framework.md)
- 最終 run の完全コマンド・レポート・validation・LB 結果:
  [mcrs/experiments/exp116_gbdt_family_ensemble/README.md](../../mcrs/experiments/exp116_gbdt_family_ensemble/README.md)
  （「LightGBM 6-seed / cross_session 完全 drop Blind B hedge」節が最終採用 run）
- Blind B 頑健性 smoke test: [exp/smoke_blindB/REPORT.md](../../exp/smoke_blindB/REPORT.md)
- candidate source デモスクリプト: [EDA/20260720_candidate_sources_demo.py](../../EDA/20260720_candidate_sources_demo.py)

## 最終提出の要点（1 分版)

- **Ranking**: 全 47,071 track catalog から、BM25/exact-match アンカー（候補プール top500 →
  上位 20 件を union へ）+ 12 の
  sparse/structural source（各 top100）で候補 union（平均 ~372/target、Dev recall 54.2%）を作り、
  source 順位のみの 73 次元特徴量で LightGBM LambdaRank（6-seed 平均）が top-20 を決める。
  dense/neural retriever・外部データ不使用。
- **Blind B 対応**: cold user 40/80 で前提が崩れた cross_session source を train/inference
  両方から完全除外（実測 0.50 vs 0.47 で正解）。
- **Response**: ranking 確定後、frozen Claude Opus 4.8 が top1 metadata に grounding した
  応答を 80 行全新規生成（leak-safe、謝罪禁止、構造 rotation で Distinct-2 確保）。
