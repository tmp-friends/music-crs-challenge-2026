# Music-CRS strategy EDA summary

Last updated: 2026-06-04

## Purpose

この EDA は、特定の exp 系列ではなく、コンペ全体の実験方針を決めるための入口にする。目的は、Music-CRS の改善余地を `retrieval`、`rerank`、`personalization`、`response` に分解し、どの signal を primary source / feature / fallback / 不採用候補にするかを決めること。

詳細な decision matrix は `EDA/tables/strategy_eda_decision_matrix.csv` に置く。

## Key observations

- 同一 session 内の exact track repeat は train/dev とも 0.00%。同じ曲をそのまま再提示する candidate source は主軸にしない。
- dev_test の同一 session artist continuity は 40.29%、album continuity は 28.23%。artist / album は query view、metadata retrieval、rerank feature の重要候補。
- dev_test の user-history exact repeat は過去 session がある event に限ると 7.00%。warm user 専用の personalization signal として扱い、cold user には広げない。
- track metadata は主要列の欠損がほぼなく、`track_name`、`artist_name`、`album_name`、`tag_list`、`popularity`、`duration` は安定して使える。
- track embeddings は 492-616 件の zero vector がある。dense retrieval / embedding feature は zero-vector bucket を必ず分けて評価する。
- user `test_cold` の cf-bpr は 129 rows 中 104 rows が zero vector、median norm も 0.0。user CF は cold user には主 signal として使わない。

## Experiment policy

| Area | Policy |
|---|---|
| Retrieval | metadata + lexical を base にし、artist / album continuity と dense source を bucket 別に足す。source 追加は union recall と final nDCG を分けて判断する。 |
| Rerank | candidate 入り失敗と rerank 失敗を分離する。history / CF / popularity は source と feature-only の両方で評価する。 |
| Personalization | warm user では user history / artist continuity を使い、cold user は dialogue + metadata を優先する。 |
| Response | ranking 固定条件で response だけを比較し、推薦 track との整合性、profile 参照、Distinct-2 を見る。 |
| Diversity | popularity は強い主 objective ではなく、fallback または mild prior として扱う。 |

## Immediate EDA backlog

1. Retrieval source diagnostics: source 別 recall@20/50/100/500、union recall、source overlap、bucket 別 miss を出す。
2. Rerank diagnostics: gold in candidate だが top20 外の tasks を抽出し、source rank と reranker score の関係を見る。
3. Query bucket EDA: explicit artist/title、genre/mood/activity、follow-up、early/late turn、warm/cold user を分類する。
4. Response quality EDA: ranking 固定で length、Distinct-1/2、metadata mention consistency、profile mention を測る。

## Decision gates

- 新しい candidate source は、全体 recall だけでなく bucket 別の nDCG@20 改善で採否を決める。
- warm user 向け signal は cold user bucket を悪化させない gate を置く。
- embedding 系は zero vector bucket を別集計し、zero vector が原因の miss を retrieval failure と混同しない。
- response 実験は `predicted_track_ids` 固定を原則にし、ranking 変化と judge/lexical 変化を混ぜない。
