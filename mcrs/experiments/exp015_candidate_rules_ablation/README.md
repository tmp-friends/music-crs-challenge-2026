# exp015_candidate_rules_ablation

## Summary

exp015 は、過去 Kaggle recommender competition の上位解法でよく使われた **multi-source candidate generation + GBDT ranker** の考え方を Music-CRS に移植し、候補生成ルールを ablation する実験。

exp014 系で分かったことは、単に候補 source を増やすだけでは top20 を汚しやすいということ。特に `user_cf_bpr` / `history_cf_bpr_item2item` / `popularity_fallback` は candidate recall ceiling を上げても、LGBM 後の top20 や Blind A judge を悪化させることがあった。exp015 では、強い source を細かく分解し、intent gate と quota で noisy source の混入位置を制御する。

Base ranking は、Blind A `ndcg@20` の最高値を更新した **`submission_exp015_A_ll_t500_A.zip`** を使う。これは exp014d `s05 logloss top500` と同じ score / 同じ reranker model だが、exp015 の `source_profile: A` 実装を通るため、candidate rule ablation の base として実装内容を追いやすい。Composite best は引き続き exp014 `s06 top500 fallback` だが、exp015 は candidate rule ablation なので、ranking base は nDCG best の logloss に寄せる。

## Background from Kaggle Solutions

調査した過去事例から、exp015 に持ち込むべきパターンは以下。

- H&M 11 位解法は、候補生成を「直近人気」「年齢層別人気」「過去購入」「過去購入カテゴリ内人気」「過去購入 item 類似」「同一商品コード」「カテゴリ one-hot 類似」に分け、候補生成後に LightGBM 系 ranker で並べ替えている。候補の良し悪しは recall / precision で見る方針も明記されている。Reference: https://blog.recruit.co.jp/data/articles/kaggle-h-and-m/
- H&M 23 位解法は、past purchased、item/user CF、LightGCN、different color、weekly / segment popularity、BERT sentence vector などを別 candidate strategy として扱っている。特に `purchase count by segment` / `popular rank by segment` は Music-CRS の文脈付き popularity に近い。Reference: https://speakerdeck.com/kuto5046/h-and-m-23th-place-solution
- OTTO 系解法では、session-level co-visitation / Word2Vec / vector recall などで候補を作り、GBDT ranker へ渡す二段構成が一般的。Music-CRS では session 内 music turn、類似 track、co-occurrence 的な履歴類似 source に対応する。Reference: https://github.com/nicolaivicol/otto-recommender
- Instacart 解法は、過去購入候補と manual feature engineering + XGBoost / CatBoost が中心。Music-CRS では「過去 track そのもの」より、過去 track の artist/tag/audio/CF 近傍を source 分解する方が安全。Reference: https://queirozf.com/entries/winning-solutions-overview-kaggle-instacart-competition

## Hypothesis

exp014 の all9 ablation で s06 以降が崩れた主因は、候補 source の意味が粗すぎて、noisy source が RRF / candidate rank 上で強く混ざったことだと考える。

改善仮説:

1. `exact_current_union` を field 別 source に分解すると、track / artist / album の強い exact signal と、広すぎる tag exact を LGBM が別々に扱える。
2. `metadata_bm25_full_context` を field / query view 別に分解すると、tag noise と artist/album intent を切り分けられる。
3. global popularity ではなく、artist / tag / user segment / history segment に条件付けた popularity を tail source として入れると、cold / ambiguous turn の recall を上げつつ top20 noise を抑えられる。
4. history CF / audio / metadata 類似は、current turn が "similar", "more like", "again" 系のときだけ quota を与えると、exp014 s08 のような無条件混入の悪化を避けられる。
5. RRF equal-weight ではなく quota-based union を使うと、source provenance を残しながら noisy source の上位侵入を制御できる。

## Base: exp015 A Logloss Top500

exp015 の initial base は `submission_exp015_A_ll_t500_A.zip` として扱う。実体は exp014d の `s05 logloss top500` Blind A config / model を exp015 の `source_profile: A` pipeline に載せ替えた control run で、score は exp014d と同じ。

| Item | Value |
|---|---|
| Base artifact | `exp/inference/blindset_A/submission_exp015_A_ll_t500_A.zip` |
| Base JSON | `exp/inference/blindset_A/exp015_A_ll_t500_A.json` |
| Base config | `mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_A_ll_t500_A.yaml` |
| Equivalent historical artifact | `exp/inference/blindset_A/submission_exp014d_s05_ll_t500_A.zip` |
| Model artifact referenced | `mcrs/experiments/exp014d_lgbm_objective_comparison/results/top500/ll/s05_exact_current/lgbm_s05_ll_top500_sampled.txt` |
| Objective | LightGBM binary `logloss` |
| Candidate topK | 500 |
| Sources | `bm25_recent_turns`, `bm25_full_context`, `metadata_bm25_full_context`, `metadata_bm25_current_user_turn`, `exact_current_union` |
| Source profile | `A` |
| Union mode | `rrf` |
| Track split | `all_tracks` |

Blind A result of the base:

```text
ndcg@20           0.3799
catalog_diversity 0.0289
lexical_diversity 0.6166
llm_judge_score   4.0500
composite_score   0.4833
```

Comparison:

- vs composite best exp014 `s06 top500 fallback`: nDCG `+0.0157`, composite `-0.0119`
- exp015 target: keep or improve nDCG while recovering judge / lexical through cleaner candidate rules and later response stabilization.

## Planned Candidate Rules

| Rule | Candidate source design | Expected effect | Risk |
|---|---|---|---|
| Exact field split | `exact_track_current`, `exact_artist_current`, `exact_album_current`, `exact_tag_current` | 明示 intent の保護、tag noise の分離 | source 数増加で feature sparsity が増える |
| Metadata BM25 field split | artist/album/text fields と tags-only を分ける | broad tag match を LGBM が落としやすい | BM25 index / feature 実装が増える |
| Segment popularity | `popular_by_artist`, `popular_by_tag`, `popular_by_user_country_age`, `popular_by_history_artist` | global popularity より文脈に合う fallback | segment が粗いと popularity bias になる |
| History source split | `history_same_artist`, `history_same_tag`, `history_cf_mean`, `history_audio_mean`, `history_recent_neighbors` | exp014 s08 の粗い history source を診断可能にする | similar intent 以外で noise |
| Intent gate | explicit / mood-tag / similar / ambiguous buckets で source 有効化 | source 混入を query に合わせる | rule classifier の誤判定 |
| Quota union | topK 内の source 別枠を固定し、RRF は枠内順位に使う | noisy source の上位侵入を抑える | quota tuning が必要 |
| Tail-only popularity | popularity は topK 不足時または下位枠だけに入れる | cold turn の候補不足対策 | top20 には効きにくい |

## Experiment Matrix

最初は Dev で candidate / rerank だけを評価し、Blind A artifact は gate 通過後に作る。

| Run | Base | Rule changes | Purpose |
|---|---|---|---|
| A | `submission_exp015_A_ll_t500_A.zip` | no change | copied control / current base |
| B | A | exact field split | exact 分解の単独効果 |
| C | B | metadata BM25 field split | metadata noise の分離 |
| D | C | segment popularity tail source | global popularity の置き換え |
| E | D | history source split | history 類似の source 別寄与 |
| F | E | intent gate | source を query bucket 別に制御 |
| G | F | quota-based union | top500 内 source 枠を制御 |
| H | G | topK 500 vs 1000 sweep | recall ceiling と noise の trade-off |

## Implementation Status

Implemented code:

- `mcrs/experiments/exp015_candidate_rules_ablation/candidate_sources.py`
  - exp014 `WideCandidateBuilder` をコピーして、source 名を field / rule 別に拡張した。
  - `track_split_types: ["all_tracks"]` を維持する。
- intent classifier
  - `candidate_sources.py` 内で current user turn から `explicit_entity`, `mood_or_genre`, `similarity_request`, `ambiguous` を判定する軽量 rule を実装した。
- `mcrs/experiments/exp015_candidate_rules_ablation/pipeline.py`
  - exp014 pipeline と同じ output schema を保ち、candidate builder / LightGBM reranker を exp015 専用実装に差し替える。
- `mcrs/experiments/exp015_candidate_rules_ablation/run_retrain.sh`
  - exp015 A logloss を base objective とし、各 candidate rule run で train/dev parquet と model を作る。
- `mcrs/experiments/exp015_candidate_rules_ablation/run_blindA.sh`
  - gate 通過 run のみ Blind A JSON / ZIP を作る。

Copied into exp015:

- feature builder: `features.py`
- dataset builder / train / apply scripts: `build_dataset.py`, `train_lgbm.py`, `apply_lgbm.py`
- shard merge helper: `merge_dataset_shards.py`
- LightGBM reranker: `lightgbm_ltr.py`
- positive-group train policy: exp014b と同じく train split は positive candidate を含む group のみ。
- objective: exp015 A / exp014d best と同じ binary `logloss` を primary。比較が必要な場合のみ `lambdarank` を secondary。

## Metrics and Gates

Primary Dev metric:

- `all_tasks_ndcg@20`

Secondary Dev metrics:

- `candidate_recall@20/100/500`
- `valid_groups_with_positive`
- `valid_ndcg@20_positive_groups`
- `oracle_ndcg@20`
- `unique_top20`
- bucket metrics by intent type
- source active rate / source unique gold hits
- top20 source provenance

Integrity checks:

- `unknown_candidate_count == 0`
- `track_split_types == ["all_tracks"]`
- future turn を query / feature に使わない
- train labels は train split のみ
- Blind labels は使わない
- Blind A ZIP root は `prediction.json` のみ
- `predicted_track_ids` は各 row 20 件、重複なし、catalog 内 track のみ

Dev gate:

```text
all_tasks_ndcg@20 > exp015 A logloss Dev all_tasks_ndcg@20
and candidate_recall@500 >= copied base
and unknown_candidate_count == 0
```

Blind A gate:

```text
primary: ndcg@20 > 0.3799
secondary: composite_score > 0.4952
fallback adopt: ndcg improves but composite does not; keep as ranking candidate for response stabilization
```

## Command Plan

Preflight:

```bash
.venv/bin/python -m py_compile \
  mcrs/experiments/exp014_lightgbm_ltr_reranker/train_lgbm.py
```

Copied base Blind A smoke:

```bash
python run_inference_blindset.py \
  --tid exp015_A_ll_t500_A \
  --batch_size 4
```

Planned full retrain command after exp015 scripts are implemented:

```bash
TOPKS=500 ONLY_RUNS=A,B,C,D,E,F,G \
  OBJECTIVE=logloss FORCE=0 \
  bash mcrs/experiments/exp015_candidate_rules_ablation/run_retrain.sh
```

Planned Blind A generation after Dev gate:

```bash
TOPKS=500 ONLY_RUNS=<best_run> \
  OBJECTIVE=logloss BATCH_SIZE=4 FORCE=0 \
  bash mcrs/experiments/exp015_candidate_rules_ablation/run_blindA.sh
```

## Planned Artifacts

- `mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_A_ll_t500_A.yaml`
- `mcrs/experiments/exp015_candidate_rules_ablation/results/exp015_candidate_rules_summary.csv`
- `mcrs/experiments/exp015_candidate_rules_ablation/results/top500/{run}/train.parquet`
- `mcrs/experiments/exp015_candidate_rules_ablation/results/top500/{run}/dev.parquet`
- `mcrs/experiments/exp015_candidate_rules_ablation/results/top500/{run}/lgbm_logloss.txt`
- `mcrs/experiments/exp015_candidate_rules_ablation/results/top500/{run}/metrics.json`
- `mcrs/experiments/exp015_candidate_rules_ablation/results/top500/{run}/feature_importance.csv`
- `exp/inference/blindset_A/exp015_{run}_ll_t500_A.json`
- `exp/inference/blindset_A/submission_exp015_{run}_ll_t500_A.zip`

## Leaderboard Result

Blind A LB result for all runs A〜G (submitted 2026-06-06)。run slug は `A=base`, `B=exact_split`, `C=metadata_split`, `D=segment_pop`, `E=history_split`, `F=intent_gate`, `G=quota_union`。

| Run | Slug | ndcg@20 | catalog_div | lexical_div | llm_judge | composite |
|---|---|---|---|---|---|---|
| **B** | exact_split | **0.3869** | 0.0293 | 0.6267 | 3.9000 | 0.4766 |
| **A** | base | 0.3799 | 0.0289 | 0.6166 | **4.1500** | **0.4908** |
| C | metadata_split | 0.3296 | 0.0281 | 0.5787 | 3.9500 | 0.4467 |
| D | segment_pop | 0.2723 | 0.0269 | 0.6198 | 3.2000 | 0.3658 |
| G | quota_union | 0.2420 | 0.0243 | 0.6142 | 3.4500 | 0.3686 |
| E | history_split | 0.2132 | 0.0255 | 0.6085 | 3.6000 | 0.3650 |
| F | intent_gate | 0.2676 | 0.0260 | 0.6038 | 2.7500 | 0.3280 |

run A (base) の生 score block（exp014d s05 logloss と同一 config / model だが judge 再採点で composite が更新された）:

```text
ndcg@20                   0.3799
catalog_diversity         0.0289
lexical_diversity         0.6166
llm_judge_score           4.1500
composite_score           0.4908
```

run B (exact_split, nDCG best) の生 score block:

```text
ndcg@20                   0.3869
catalog_diversity         0.0293
lexical_diversity         0.6267
llm_judge_score           3.9000
composite_score           0.4766
```

Validation for the local ZIP:

```text
zip entries: prediction.json only
records: 80
bad_len: 0
bad_unique: 0
```

### LB からの解釈

- **B (exact field split) が nDCG@20 best = 0.3869** で base A (0.3799) を +0.0070 上回った。Dev の `all_tasks_ndcg@20` でも B が base 超え (0.14442 vs 0.13951) だったので、Dev→LB で方向が一致。`exact_current_union` を track/artist/album/tag に分割する効果は外部評価でも確認できた。
- ただし **composite は A (0.4908) が最高**で、B (0.4766) は llm_judge_score が下がった (4.1500 → 3.9000) ぶん composite で負ける。nDCG-best と composite-best が割れる exp014d と同じ構図。
- C 以降は nDCG / composite ともに base 未満。Dev での悪化順 (D/E/F/G) が LB でもほぼ再現し、**segment popularity / history split / intent gate / quota union を足すほど top20 が崩れる**ことを外部評価で確証した。特に F (intent_gate) は judge=2.7500 と最低で、source gating の誤判定が response 品質まで巻き込んだ可能性。
- 結論: 細粒度化で現状効くのは **exact field split のみ**。それ以外の source 追加系は今回の定義では LB 逆効果。ただし **B (exact_split) を以後の ranking base として採用**する。C〜G で悪化した source 系統も、未検証の組み合わせ・新規 source がまだ多いため **打ち切らず**、source 定義 / gating / feature 化方法を変えて継続検討する。
- 目標: Blind A **`ndcg@20` 0.500 超え**。現状 best 0.3869 から +0.11 以上必要なので、candidate source 追加・改良と feature / reranker 改善を積み上げ、B を base にした response stabilization で composite も引き上げる。

## Interpretation Plan

- B/C が改善: exact / metadata source 分解を採用し、field-specific source rank feature を exp014 系へ戻す。
- D が改善: global `popularity_fallback` は廃止し、segment popularity を tail source として採用する。
- E が recall だけ改善して nDCG が落ちる: history 類似は source ではなく feature-only または intent-gated source にする。
- F/G が改善: Blind B 向けには source を増やすより、intent gate + quota を primary candidate builder にする。
- nDCG は改善するが composite が落ちる: exp014d と同じく ranking candidate として保持し、response stabilization を別実験で実施する。

## Next Actions

LB 確定後の方針（base = B exact_split, target `ndcg@20` > 0.500）:

1. **B を ranking base に固定**し、以後の candidate / rerank 実験は B 比較で評価する。
2. B base のまま **judge を落とさない response stabilization** を別実験で組み、composite で current best (0.4952) 超えを狙う。
3. C〜G の悪化 source は打ち切らず、**source 定義 / gating / feature 化を変えて再挑戦**する。例: history / segment popularity は source 直挿しではなく feature-only か intent-gated source に変える。
4. **未検証 source の追加**を継続する（audio 類似、co-listen / session CF、artist-graph、tag taxonomy 展開など）。recall ceiling ではなく top20 precision を上げる source を優先する。
5. `ndcg@20` 0.500 に向け、candidate source 改良 + feature fusion + reranker (objective / depth / negative sampling) をまとめて積み上げる。
