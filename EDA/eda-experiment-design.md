# Music-CRS EDA 実験設計

Last updated: 2026-06-04

## Summary

本ドキュメントは RecSys Challenge 2026: Music-CRS の問題設計と TalkPlayData-Challenge dataset を踏まえ、コンペ全体の実験方針を決めるための EDA 計画、優先順位、成果物、次実験への接続を定義する。

主目的は、個別の既存実験系列に閉じず、Music-CRS の改善余地を「意図理解」「候補生成」「personalization」「metadata / embedding 利用」「rerank」「response 品質」に分解し、どの情報源をどの条件で使うべきかを判断できる診断情報を作ること。現行 best は比較対象の一つとして扱うが、EDA の中心軸にはしない。

## Problem Design

Music-CRS は Conversational Music Recommendation の課題であり、各 session-turn で以下を同時に出力する。

- `predicted_track_ids`: `all_tracks` catalog 内 track_id の ranked top 20。
- `predicted_response`: 会話文脈、ユーザープロファイル、推薦 track に沿った自然言語応答。

入力文脈は multi-turn dialogue、user profile、listening history、track metadata、pre-extracted embeddings を含む。推論時は候補 track を必ず全 catalog から検索し、`test_tracks` や評価対象 track のみに絞ってはいけない。

評価は以下の composite score で行われる。

| Metric | Weight | EDA 上の意味 |
|---|---:|---|
| nDCG@20 | 0.50 | 正解 track を top 20 の上位に置けるか。最優先の診断対象。 |
| Catalog Diversity | 0.10 | 人気 track への過集中を避けられているか。 |
| Lexical Diversity | 0.10 | response が過度に定型化していないか。 |
| LLM-as-a-Judge | 0.30 | response の personalization / explanation quality。Blind で重要。 |

## Dataset Baseline Facts

ローカル EDA で確認済みの主要値:

| Dataset | Split | Rows / Sessions | Users | Note |
|---|---|---:|---:|---|
| Conversation | train | 15,199 | 8,591 | 学習・パターン分析用 |
| Conversation | dev_test | 1,000 | 500 | ローカル評価・チューニング用 |
| Conversation | blind_a | 80 | 58 | interim leaderboard 用 |
| Track Metadata | all_tracks | 47,071 | - | 推薦対象の全 catalog |
| Track Metadata | test_tracks | 7,405 | - | `all_tracks` の部分集合。推薦候補制限には使わない |
| User Metadata | all_users | 8,772 | - | user profile DB |

Embedding は track 側に 6 種類、user 側に `cf-bpr` がある。track embeddings には 492-616 件の zero vector があり、user `test_cold` は zero vector であるため、embedding feature の有効性は warm / cold と missing vector bucket で分けて見る。

## Design Principles

- nDCG@20 に直結する retrieval / rerank 診断を最優先する。
- `track_split_types: ["all_tracks"]` を全 EDA と実験で維持する。
- train labels は学習・分析用、dev_test labels は検証用に限定し、Blind の ground truth 推測や future turn leakage を避ける。
- 既存 EDA 成果物は再利用し、不足している診断だけを追加する。
- official baseline、local current best、強い単純 prior の複数基準で比較できるように、bucket 別の recall / nDCG / candidate coverage を残す。

## EDA Workstreams

### 1. Data Inventory And Schema Check

目的:

- dataset split、行数、schema、ID coverage を再検証する。
- local docs の値と実データが一致しているかを確認する。
- 推薦対象が `all_tracks` であることを以後の診断の前提として固定する。

確認項目:

- conversation split ごとの session 数、user 数、turn 数、music turn 数。
- `conversation.music.content` の track_id が `all_tracks` に存在するか。
- `test_tracks` が `all_tracks` に包含されるか。
- user_id が `all_users` と user embeddings にどの程度存在するか。
- track_id が metadata と各 embedding split にどの程度存在するか。

成果物:

- `EDA/tables/data_inventory.csv`
- `EDA/tables/id_coverage.csv`
- `EDA/summary/input-data-eda-summary.md` の更新または追記

Gate:

- missing track_id が `all_tracks` に存在しない場合は、以降の retrieval 診断前に原因を特定する。

### 2. Conversation Structure EDA

目的:

- target turn の位置、role pattern、goal category が retrieval difficulty にどう影響するかを把握する。

確認項目:

- split 別の turns/session、music turns/session。
- target turn number の分布。
- user / assistant / music role の並び。
- `conversation_goal.category`、`specificity`、`listener_goal` の分布。
- user profile の age_group、country、gender、preferred_language、preferred_musical_culture の分布。

追加 bucket:

- early turn / middle turn / late turn。
- explicit request: artist/title/genre/mood が明示されている query。
- broad request: mood、activity、recommend something 系。
- follow-up request: previous recommendation への継続・修正。

成果物:

- `EDA/tables/conversation_stats.csv`
- `EDA/tables/turn_stats.csv`
- `EDA/tables/goal_stats.csv`
- `EDA/figures/conversation_distributions.png`
- `EDA/figures/user_profile_distributions.png`

### 3. Label And Music Continuity EDA

目的:

- 正解 track が会話内・ユーザー履歴内でどの signal に寄っているかを把握する。
- exact track 再聴より artist / album / tag continuity を candidate source や rerank feature に使えるか判断する。

既存観察:

- 同一 session 内 exact track 再聴率は train/dev とも 0.00%。
- 同一 session 内 artist 継続率は train 56.88%、dev_test 40.29%。
- 同一 user の過去 session 由来 exact track 再聴率は、過去 session が存在する event に限ると train 10.82%、dev_test 7.00%。

追加確認項目:

- turn number 別の artist / album / tag continuity。
- goal category 別の artist / album / tag continuity。
- user warm / cold 別の過去 session exact track / artist continuity。
- `prior_artist@K`、`prior_album@K`、`prior_tag@K` の actionability。
- continuity signal が正解 track を candidate pool に入れる効果と、ranking 上位化する効果を分けて測る。

成果物:

- `EDA/tables/relisten_rates_by_turn.csv`
- `EDA/tables/relisten_rates_by_goal.csv`
- `EDA/tables/relisten_rates_by_user_split.csv`
- `EDA/tables/relisten_actionability.csv`
- `EDA/summary/relisten-eda-summary.md` の更新

Decision:

- session-local exact replay は candidate source として採用しない。
- prior artist / album は candidate source または rerank feature の候補として維持する。
- tag continuity は broad な補助 feature として扱い、単独 source 化は慎重にする。

### 4. Catalog And Metadata EDA

目的:

- metadata field の欠損、重複、分布、正解 track との分布差を確認し、BM25 corpus と rerank feature の採否を決める。

確認項目:

- `track_name`、`artist_name`、`album_name`、`tag_list`、`release_date`、`popularity`、`duration` の欠損率。
- all_tracks と train/dev gold tracks の popularity 分布差。
- top artist / album / tag の集中度。
- duplicate track metadata と同名曲・同一 artist の扱い。
- release year bucket と user request / goal category の関係。

成果物:

- `EDA/tables/track_metadata_missing.csv`
- `EDA/tables/track_metadata_duplicates.csv`
- `EDA/tables/top_tracks_artists_tags.csv`
- `EDA/tables/music_track_frequency.csv`
- `EDA/figures/track_metadata_distributions.png`

Experiment Connection:

- BM25 corpus fields の比較: `track_name` / `artist_name` / `album_name` baseline に `tag_list`、`release_date` を足す。
- popularity は catalog bias を強めやすいため、主要 source ではなく fallback、prior、または mild feature として評価する。

### 5. Embedding EDA

目的:

- track / user embeddings の欠損、zero vector、norm、近傍品質を把握し、dense retrieval と rerank feature の信頼できる適用範囲を決める。

確認項目:

- embedding type ごとの dim、null count、zero vector count、norm 分布。
- zero vector track が gold track に出る頻度。
- user `train` / `test_warm` / `test_cold` の norm 分布。
- `metadata-qwen3_embedding_0.6b`、`lyrics-qwen3_embedding_0.6b`、`attributes-qwen3_embedding_0.6b` の nearest neighbor qualitative sample。
- `cf-bpr` user-track similarity が warm user でのみ有効か。

成果物:

- `EDA/tables/track_embedding_stats.csv`
- `EDA/tables/track_embedding_norms.csv`
- `EDA/tables/user_embedding_stats.csv`
- `EDA/tables/user_embedding_norms.csv`
- `EDA/figures/embedding_norms.png`

Decision:

- zero vector bucket は retrieval / rerank metrics を別集計する。
- user CF は cold / warm と zero vector bucket を分け、candidate source、feature-only、fallback のどれが適切かを個別に評価する。

### 6. Retrieval Candidate Diagnostics

目的:

- 各 candidate source が正解 track をどの程度 candidate pool に入れられるかを、source 別・bucket 別に測る。
- lexical、metadata、dense、history、popularity の各 source が効く query 条件を特定し、Blind B 向けの候補生成 mix を決める。

対象 source:

- `bm25_recent_turns`
- `bm25_full_context`
- `metadata_bm25_full_context`
- `metadata_bm25_current_user_turn`
- `exact_current_union`
- `dense_text_recent`
- optional: `user_cf_bpr`
- optional: `history_cf_bpr_item2item`
- popularity / goal prior

Metrics:

- source 別 recall@20 / 50 / 100 / 500。
- union candidate recall@K。
- source overlap。
- gold candidate rank distribution。
- candidate 入り失敗率。
- candidate には入ったが final top20 に落ちた率。

Buckets:

- turn number。
- goal category。
- user warm / cold。
- explicit artist/title request。
- broad mood/activity request。
- previous artist continuity あり/なし。
- zero vector gold track あり/なし。
- popularity decile。

成果物:

- `EDA/tables/retrieval_recall_by_source.csv`
- `EDA/tables/retrieval_recall_by_bucket.csv`
- `EDA/tables/source_overlap.csv`
- `EDA/tables/candidate_failure_cases.csv`
- `EDA/summary/retrieval-diagnostics-summary.md`

Gate:

- source 単体でなく、query bucket ごとの hit / miss で採否を決める。
- 追加 source は union recall を上げても final nDCG を下げる可能性があるため、candidate 入り改善と rerank 後順位改善を分けて評価する。

### 7. Rerank Diagnostics

目的:

- candidate generation の問題と reranker の問題を分離する。
- feature-based、embedding-based、LLM reranker がどの bucket で正例を上げ下げしているかを確認する。

確認項目:

- gold in candidate pool かつ final top20 外の dev tasks。
- reranker score と candidate source rank の関係。
- feature importance / SHAP / permutation など、使える手段で feature contribution を確認する。
- candidate topK、source mix、train group policy の差分。
- lexical-heavy / dense-heavy / history-heavy / popularity-heavy bucket 別 nDCG 差。

Metrics:

- valid nDCG@1 / 10 / 20。
- gold candidate rank before rerank。
- gold rank after rerank。
- per-bucket delta vs official baseline / local best / strong prior。

成果物:

- `EDA/tables/rerank_bucket_metrics.csv`
- `EDA/tables/rerank_failure_cases.csv`
- `EDA/tables/lightgbm_feature_importance.csv`
- `EDA/summary/rerank-diagnostics-summary.md`

Experiment Connection:

- candidate source の失敗であれば retrieval 実験へ、candidate 入り後の順位失敗であれば rerank 実験へ接続する。
- user / history / popularity signal は、source として混ぜる場合と feature-only で使う場合を分離して評価する。

### 8. Response Quality EDA

目的:

- recommendation ranking を固定した状態で、response の personalization、explanation consistency、lexical diversity を評価する。

確認項目:

- response が推薦 track の artist / title / genre と矛盾していないか。
- user profile と listener_goal への参照が自然か。
- response template の重複、Distinct-2、長さ分布。
- submission / local prediction の judge score が高い response と低い response の差分。

Metrics:

- response length。
- Distinct-1 / Distinct-2。
- repeated phrase rate。
- track metadata mention consistency。
- optional self-judge rubric: personalization / explanation quality / contradiction。

成果物:

- `EDA/tables/response_length_stats.csv`
- `EDA/tables/response_lexical_diversity.csv`
- `EDA/tables/response_consistency_samples.csv`
- `EDA/summary/response-quality-summary.md`

Gate:

- nDCG を動かさない JSON 固定条件でのみ response variant を比較する。
- `llm_judge_score >= 4.30` または lexical 維持で composite 改善が見込める variant を採用候補にする。

## Execution Order

| Priority | Workstream | Reason | Output |
|---:|---|---|---|
| P0 | Retrieval Candidate Diagnostics | nDCG@20 改善に直結。source ごとの有効 bucket を特定する。 | `retrieval-diagnostics-summary.md` |
| P0 | Rerank Diagnostics | candidate 入り失敗と rerank 失敗を分離する。 | `rerank-diagnostics-summary.md` |
| P1 | Label And Music Continuity EDA | artist / album continuity を source または feature に落とす判断材料。 | `relisten-eda-summary.md` |
| P1 | Catalog And Metadata EDA | corpus field と metadata feature の採否を決める。 | metadata tables / figures |
| P1 | Embedding EDA | dense / CF feature の適用範囲を切る。 | embedding tables / figures |
| P2 | Conversation Structure EDA | bucket 設計と query classifier の基礎。 | conversation tables / figures |
| P2 | Response Quality EDA | ranking 固定後の composite 改善。 | `response-quality-summary.md` |
| P3 | Data Inventory And Schema Check | 既存確認済み。新データ公開時に再実行。 | inventory tables |

## Initial Commands

既存 EDA を再実行または更新する場合の入口:

```bash
python EDA/20260526_relisten_eda.py
```

新規 diagnostic script を追加する場合の推奨配置:

```text
EDA/
├── yyyymmdd_retrieval_diagnostics.py
├── yyyymmdd_rerank_diagnostics.py
├── yyyymmdd_response_quality_eda.py
├── summary/
├── tables/
└── figures/
```

大きな実験コードや model artifact を伴う場合は、`mcrs/experiments/expNNN_slug/` 側に置き、`EDA/` には実行ファイル、`EDA/tables/`、`EDA/figures/`、`EDA/summary/` だけを残す。

## Validation Checklist

EDA から実験へ進む前に以下を確認する。

- `track_split_types == ["all_tracks"]`。
- predicted / candidate track_id は `all_tracks` に存在する。
- train label と dev label を混ぜて学習評価していない。
- Blind A/B の future turn や ground truth を参照していない。
- zero vector / cold user bucket を分けて report している。
- official baseline、local current best、単純 prior のどれとの差分かを明記している。

## Next Actions

1. `EDA/summary/strategy-eda-summary.md` を入口にし、既存 EDA から実験方針に直結する signal / decision / gate を整理する。
2. `retrieval_diagnostics.py` を作成し、source 別 recall@20/50/100/500 と bucket 別 recall を出す。
3. `rerank_diagnostics.py` を作成し、candidate 入り失敗と rerank 失敗を dev tasks で分離する。
4. response quality EDA を ranking 固定条件で実行し、personalization / explanation consistency / lexical diversity の改善余地を測る。
5. 診断結果から、Blind B 向け retrieval / rerank / response の優先順位を `mcrs/experiments/experiment-plan.md` に反映する。

