# 最終提出アーキテクチャ（Blind B final: `submission_exp116_lgbm6_nocs_B.zip`）

RecSys Challenge 2026 Music-CRS の最終提出（final leaderboard 採用 submission）の実装内容を、
コード実体と紐づけてまとめる技術リファレンス。

## 最終結果

| 項目 | 値 |
|---|---|
| 採用 submission | `exp/inference/blindset_B/submission_exp116_lgbm6_nocs_B.zip` |
| nDCG@20 | **0.352800** |
| Catalog Diversity | 0.027533 |
| Lexical Diversity | 0.731453 |
| LLM-judge | 4.30 |
| Composite | **0.499799** |
| 順位 | **Industry Track 8 位 / 全体 16 位（40 teams）** |

Blind B は team 最大 3 提出。3 枠の構成と結果:

| submission | 構成 | Codabench score |
|---|---|---:|
| `lxct6_B` | LightGBM+XGBoost+CatBoost+TabM 4-family × 6-seed、cross_session 込み（leak-safe 修正版） | (primary 提出) |
| `lgbm6_B` | LightGBM 単独 6-seed、cross_session 込み（leak-safe 修正版） | 0.47 |
| **`lgbm6_nocs_B`（最終採用）** | **LightGBM 単独 6-seed、cross_session 完全除外（12 source）** | **0.50** |

## パイプライン全体（retrieve → rank → respond の 3 段）

```text
対話履歴 + (あれば) user_profile
   │
   ├─ [1] BM25 アンカー (exp015 B):
   │      BM25 4 view + exact-match 4 field → RRF union top500
   │      → 1段目 LightGBM (binary logloss) → anchor ranking
   │
   ├─ [2] 補助 candidate source 12 種 (各 top100):
   │      continuity ×4 / QNM ×3 / entity ×4 / transition ×1
   │
   ├─ [3] union + within-session 既出除外 → 73 次元 source-rank 特徴量
   │      → LightGBM LambdaRank 6-seed 平均 → top-20 ranking
   │
   └─ [4] ranking 確定後、Claude Opus 4.8 (frozen) が
          top1 metadata に grounding した response を 80 行全新規生成
```

recommendation path（1–3 段）は CPU の BM25S + LightGBM のみ。dense retrieval・fine-tuned
neural retriever・外部データは一切使わない（合法・leak-safe。実験の淘汰過程はメモリ/実験ログ参照）。

## [1] BM25 アンカー

詳細: [anchor_pipeline.md](anchor_pipeline.md)。
config は [exp015_B_ll_t500_B.yaml](../../mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_B_ll_t500_B.yaml)
（`enabled_sources` 8 種、`retrieval_topk: 500`、`union_mode: rrf`、`track_split_types: all_tracks`）。
2 段目には `--primary "exp015B=exp/inference/blindset_B/exp015_B_ll_t500_B.json"` として渡す。

## [2] 補助 candidate source 12 種

実サンプルでの挙動: [candidate_sources_train_sample_walkthrough.md](candidate_sources_train_sample_walkthrough.md)。

| 系統 | source 名 | 実装 | 中身 |
|---|---|---|---|
| Session-local continuity ×4 | `cont_artist` / `cont_album` / `cont_aa` / `cont_aat` | exp021 `continuity_candidates.py` | 既出 music の artist/album/tag を catalog 展開（popularity 順、within-session 既出除外） |
| Query-neighbor memory ×3 | `qnm_cur` / `qnm_rp` / `qnm_hist` | exp021 `query_neighbor_memory.py` | request を bm25s で train task に照合し近傍 task の GT を伝播（train label のみ index） |
| Entity ×4 | `ent_cur` / `ent_tight` / `ent_r2` / `ent_r4` | exp023 `entity_candidates.py` | user 発話中の track/artist/album phrase を catalog 照合（観測窓 4 種） |
| Transition ×1 | `trans_r1` | exp024 `transition_memory.py` | train の「直前 music entity → 次 GT」遷移統計 |

各 source top100。Dev 8000 target での 12-source pool は平均 ~372 候補/target、
candidate recall 54.16%（4,333/8,000）。wide mixed pool（recall 66.14% / ~1,015 候補）は
positive density 低下で Blind A 実測負け → 高精度 pool を採用（oracle recall ≠ realized nDCG）。

**cross_session を外した理由**（13 番目の source だったが最終提出では除外）:
Blind B は cold user 40/80 で user_id が空。leak-safe 修正（空 id を pool 除外）後も
warm 側 21/38 が cross-session 候補ゼロで実効性がほぼ消滅
（[exp/smoke_blindB/REPORT.md](../../exp/smoke_blindB/REPORT.md)）。分布シフト hedge として
train/inference 両方から完全除外した 12-source 構成が実測で勝った（0.50 vs 0.47）。

## [3] 特徴量と reranker

特徴量は source 上の「順位」だけから作る 73 次元
（[rank_source_lgbm_mixed_topk.py](../../mcrs/experiments/exp027_wide_source_lgbm/rank_source_lgbm_mixed_topk.py)
の `build_features` / `feature_names`）:

- anchor 5: `primary_inv_rank` / `primary_is_top1` / `primary_in_top5` / `primary_in_top10` / `primary_in_top20`
- 各 source 5 × 12 = 60: `{src}_inv_rank` / `{src}_in_top5` / `{src}_in_top20` / `{src}_rank_clipped` / `{src}_rank_pct`
- 集約 8: `source_hit_rate` / `source_hit_count` / `source_best_inv_rank` / `source_rrf_sum_k60` /
  `source_rrf_sum_k200` / `turn_number` / `is_early_turn` / `is_followup_turn`

reranker（[ensemble_rerank.py](../../mcrs/experiments/exp116_gbdt_family_ensemble/ensemble_rerank.py)、
`--families "lgbm"`）:

- LightGBM LambdaRank、`num_boost_round=450` / `num_leaves=31` / `min_data_in_leaf=80` / `lambda_l2=8.0`
- seed `20260616`–`20260621` の 6 モデル raw score 平均で final ranking
- 学習データ: candidate_rows **2,949,717** / positive groups **4,333**（group = session-turn、正例は GT 1 track）
- Blind B 推論は `--skip_oof`（full-fit のみ）。within-session 既出は
  `--exclude_json exp/smoke_blindB/sources/within_session_exclude_blindB.json` で候補段から除外

model-family ensemble（XGBoost/CatBoost/TabM 追加、lxct6）は Dev OOF で LightGBM 単独を下回り
（0.17953 vs 0.18173）、Blind B 実測でも nocs 構成の LightGBM 単独が最良だった。

## [4] Response 生成

[generate_responses_v2.py](../../mcrs/experiments/exp063_response_quality/generate_responses_v2.py)
`--backend claude --model claude-opus-4-8`（frozen、fine-tuning なし）。

- prompt 入力は leak-safe: current turn までの対話 + 入力側 user_profile（あれば）+
  final top1 track の metadata（title/artist/album/release date/整形済み genre・mood tag）のみ
- 謝罪表現の禁止・metadata grounding の強制・書き出しスタイルの決定的 rotation（Distinct-2 対策）
- **submission ごとに 80 行すべて新規生成**（過去 response の流用・top1 変更行のみの template 差し替えは禁止。
  ranking と response の整合を保ち composite を submission 間で公平比較するため）
- 空応答は grounded fallback で必ず埋める（validator が hard error 扱いのため）

## 再現コマンド

完全なコマンドライン（source パス 12 本込み）とレポート・validation 出力は
[exp116 README「LightGBM 6-seed / cross_session 完全 drop Blind B hedge」節](../../mcrs/experiments/exp116_gbdt_family_ensemble/README.md)
を参照。要点:

```bash
# 1) reranker（12 source, LightGBM 6-seed, OOF skip）
.venv/bin/python mcrs/experiments/exp116_gbdt_family_ensemble/ensemble_rerank.py \
  --families "lgbm" \
  --seeds "20260616,20260617,20260618,20260619,20260620,20260621" \
  --skip_oof \
  --primary "exp015B=..." --blind_primary "exp015B=exp/inference/blindset_B/exp015_B_ll_t500_B.json" \
  --source ... (12 本) --blind_source ... (12 本) \
  --exclude_json exp/smoke_blindB/sources/within_session_exclude_blindB.json \
  --num_boost_round 450 --num_leaves 31 --min_data_in_leaf 80 --lambda_l2 8.0 --num_threads 8 \
  --output_blind_tracks .../blindB_tracks_ens_blindB_lgbm6_nocs.json

# 2) response 全行新規生成
.venv/bin/python mcrs/experiments/exp063_response_quality/generate_responses_v2.py \
  --backend claude --model claude-opus-4-8 \
  --tracks_json .../blindB_tracks_ens_blindB_lgbm6_nocs.json \
  --output exp/inference/blindset_B/exp116_lgbm6_nocs_B.json \
  --test_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-B --batch_size 6
```

提出制約（top-20 ranked / 全 catalog / ZIP root=prediction.json / ファイル名 63 字以下）は
validator で全 pass 済み（records 80 / catalog 外 0 / 空 response 0）。
