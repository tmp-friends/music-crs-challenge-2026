# exp021_candidate_fusion

## Summary

docs/reference の Kaggle 解法で繰り返し出てくる retrieve -> rerank -> calibration / label propagation を、Music-CRS の候補生成へ移植した。既存 candidate source の単純 sweep は exp014〜exp019c でかなり試しているため、exp021 では **train task 近傍から正解 track を伝播する query-neighbor memory (QNM)** と、EDA で強かった **session-local artist / album continuity source** を新規 candidate source として作成し、exp019c Dev best に薄く RRF 注入した。

比較対象は exp019c final F4 + listwise orderavg tail:

- Dev `all_tasks_ndcg@20=0.19670257787442233`
- Artifact: `mcrs/experiments/exp019c_llm_reranker_calibration/results/dev_tracks_exp019c_finalF4_orderavg_fk10_w0915_p1.json`

## Hypothesis

- Kaggle の類似問題では、test query に近い train example の label を候補化する nearest-neighbor / label-memory が strong candidate source になる。
- Music-CRS でも train dialogue の request text と user profile/listening context が近ければ、正解 track・artist・tag が伝播可能なはず。
- EDA では session-local exact repeat は 0% だが、artist / album continuity は Dev で高い。直前までに提示した music track の artist / album から全 catalog へ展開すれば、既存 source が落とす follow-up turn を拾える可能性がある。
- ただし Blind A では Dev label-tuned fusion が崩れやすいので、Dev best と top1 protected の 2 系統を作る。

## Changes

- Code:
  - `query_label_memory.py`: n-gram label memory。smoke では弱く、full 採用せず。
  - `query_neighbor_memory.py`: train task BM25 近傍から label / artist expansion / tag expansion を集約する新規 candidate source。
  - `continuity_candidates.py`: session history の music track から artist / album / tag を直接 catalog 展開する EDA 起点 candidate source。
  - `greedy_rrf_sweep.py`: source top-k と source reuse を指定できるようにし、QNM source の RRF 注入を探索。
- Data/features:
  - Train labels のみを memory 化。Dev/Blind の正解・未来 turn・`thought` は使わない。
  - track は全 catalog から返す。
- Response:
  - Blind A は exp019c finalF4 orderavg の非空 response を key で引き継ぎ。top1 protected 版は `--require_top1_match` で整合性を確認。

## Configuration

- Date: 2026-06-16
- Eval dataset: `devset`, `blindset_A`
- Conversation dataset: `talkpl-ai/TalkPlayData-Challenge-Dataset`
- Blind A dataset: `talkpl-ai/TalkPlayData-Challenge-Blind-A`
- Base prediction: exp019c finalF4 orderavg p1
- Track split: `all_tracks`
- QNM variants: `current`, `recent_profile`, `history_music`
- QNM settings: `top_neighbors=240`, `topk=500`, `artist_top_tracks=120`, `tag_top_tracks=24`, `tag_weight_scale=0.25`
- Continuity variants: `album`, `artist`, `album_artist`, `artist_album`, `recent2_album_artist`, `recent1_album_artist`, `album_artist_tag`

## Commands

```bash
.venv/bin/python mcrs/experiments/exp021_candidate_fusion/query_neighbor_memory.py \
  --base_candidate_parquet mcrs/experiments/exp016_dense_retriever_ft/results/b2dense_nocap_s360/dev.parquet \
  --output_dir mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory \
  --topk 500 \
  --variants current,recent_profile,history_music \
  --top_neighbors 240 \
  --artist_top_tracks 120 \
  --tag_top_tracks 24 \
  --tag_weight_scale 0.25 \
  --n_threads 12 \
  --chunksize 64
```

```bash
.venv/bin/python mcrs/experiments/exp021_candidate_fusion/continuity_candidates.py \
  --output_dir mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates \
  --topk 500
```

```bash
.venv/bin/python mcrs/experiments/exp021_candidate_fusion/continuity_candidates.py \
  --skip_dev \
  --blind_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-A \
  --variants album,artist,album_artist \
  --topk 20 \
  --output_dir mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates_blindA_top20
```

```bash
.venv/bin/python mcrs/experiments/exp021_candidate_fusion/query_neighbor_memory.py \
  --skip_dev \
  --blind_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-A \
  --output_dir mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory_blindA \
  --topk 500 \
  --variants current,recent_profile,history_music \
  --top_neighbors 240 \
  --artist_top_tracks 120 \
  --tag_top_tracks 24 \
  --tag_weight_scale 0.25 \
  --n_threads 12 \
  --chunksize 64
```

## Dev Results

QNM source standalone:

| Variant | nDCG@20 | hit@20 | hit@100 | hit@500 | base miss recovered@500 |
|---|---:|---:|---:|---:|---:|
| current | 0.0158285157 | 0.04525 | 0.159375 | 0.32025 | 0.130084 |
| recent_profile | 0.0252466707 | 0.06950 | 0.217625 | 0.396625 | 0.173062 |
| history_music | 0.0414734647 | 0.10950 | 0.284750 | 0.485500 | 0.193400 |

Fusion:

| Variant | Dev nDCG@20 | Delta vs exp019c | Blind A top1 changed | Note |
|---|---:|---:|---:|---|
| exp019c finalF4 orderavg p1 | 0.1967025779 | -- | -- | previous Dev best |
| exp021 QNM top20 reuse greedy RRF | **0.1970058104** | **+0.0003032325** | 2/80 | Dev best, response mismatch risk |
| exp021 QNM top20 reuse p1 greedy RRF | 0.1969462064 | +0.0002436285 | 0/80 | safer Blind A artifact |

Dev best steps:

1. `qnm_current`, `fusion_k=20`, `primary_weight=0.88`, `protect_top=1`
2. `qnm_recent_profile`, `fusion_k=15`, `primary_weight=0.925`, `protect_top=0`
3. `qnm_current`, `fusion_k=10`, `primary_weight=0.75`, `protect_top=10`
4. `qnm_recent_profile`, `fusion_k=5`, `primary_weight=0.97`, `protect_top=0`
5. `qnm_current`, `fusion_k=5`, `primary_weight=0.92`, `protect_top=10`
6. `qnm_history_music`, `fusion_k=2`, `primary_weight=0.985`, `protect_top=15`

Fold deltas for Dev best: `[+0.0000062, +0.0011333, +0.0000129, +0.0005492, -0.0001854]`。1 fold は negative のため Blind A では risky。

Continuity source standalone:

| Variant | nDCG@20 | hit@20 | hit@100 | hit@500 | active rate |
|---|---:|---:|---:|---:|---:|
| album_artist | 0.1519857781 | 2508 | 3243 | 3544 | 0.875 |
| artist_album | 0.1443294763 | 2457 | 3246 | 3546 | 0.875 |
| album_artist_tag | 0.1340052659 | 2235 | 3321 | 4079 | 0.875 |
| recent2_album_artist | 0.1332548179 | 2242 | 2831 | 3124 | 0.875 |
| album | 0.1330360738 | 2080 | 2342 | 2621 | 0.875 |
| artist | 0.1204756538 | 2166 | 3201 | 3521 | 0.875 |
| recent1_album_artist | 0.1197009330 | 1992 | 2473 | 2760 | 0.875 |

Continuity fusion:

| Variant | Dev nDCG@20 | Delta | Fold min delta | Blind A top1 changed | Note |
|---|---:|---:|---:|---:|---|
| QNM Dev best | 0.1970058104 | -- | -0.0001854 | 2/80 | previous exp021 best |
| continuity top20 reuse | 0.1981343702 | +0.0011285598 | +0.0001097 | 0/80 | strong Dev update |
| continuity p1 safe | 0.1978844032 | +0.0009381968 vs QNM p1 | +0.0002703 | 0/80 | safer submit candidate |
| continuity top100 tail | **0.1981492892** | +0.0000149190 vs continuity top20 | -0.0000056 | 0/80 | Dev-highest but tiny / risky |

Continuity top20 Dev best steps:

1. `cont_album_artist`, `fusion_k=15`, `primary_weight=0.70`, `protect_top=1`
2. `cont_artist`, `fusion_k=1`, `primary_weight=0.925`, `protect_top=15`
3. `cont_album`, `fusion_k=15`, `primary_weight=0.70`, `protect_top=15`
4. `cont_album`, `fusion_k=10`, `primary_weight=0.95`, `protect_top=0`
5. `cont_album`, `fusion_k=10`, `primary_weight=0.95`, `protect_top=0`
6. `cont_album`, `fusion_k=2`, `primary_weight=0.95`, `protect_top=15`

## Artifacts

- Dev best tracks: `mcrs/experiments/exp021_candidate_fusion/results/dev_tracks_exp021_qnm_top20_reuse_greedy_rrf.json`
- Dev safer tracks: `mcrs/experiments/exp021_candidate_fusion/results/dev_tracks_exp021_qnm_top20_reuse_p1_greedy_rrf.json`
- Blind A Dev-best tracks: `mcrs/experiments/exp021_candidate_fusion/results/blindA_tracks_exp021_qnm_top20_reuse_greedy_rrf.json`
- Blind A safer tracks: `mcrs/experiments/exp021_candidate_fusion/results/blindA_tracks_exp021_qnm_top20_reuse_p1_greedy_rrf.json`
- Submission JSON (Dev best): `exp/inference/blindset_A/exp021_qnm_devbest_A.json`
- Submission ZIP (Dev best): `exp/inference/blindset_A/submission_exp021_qnm_devbest_A.zip`
- Submission JSON (safer p1): `exp/inference/blindset_A/exp021_qnm_p1_A.json`
- Submission ZIP (safer p1): `exp/inference/blindset_A/submission_exp021_qnm_p1_A.zip`
- Dev continuity top20 tracks: `mcrs/experiments/exp021_candidate_fusion/results/dev_tracks_exp021_continuity_top20_reuse_greedy_rrf.json`
- Dev continuity p1-safe tracks: `mcrs/experiments/exp021_candidate_fusion/results/dev_tracks_exp021_continuity_top20_reuse_p1_safe_greedy_rrf.json`
- Dev continuity top100-tail tracks: `mcrs/experiments/exp021_candidate_fusion/results/dev_tracks_exp021_continuity_top100_tail_after_devbest_rrf.json`
- Submission JSON (continuity p1-safe): `exp/inference/blindset_A/exp021_cont_p1safe_A.json`
- Submission ZIP (continuity p1-safe): `exp/inference/blindset_A/submission_exp021_cont_p1safe_A.zip`
- Submission JSON (continuity Dev best): `exp/inference/blindset_A/exp021_cont_devbest_A.json`
- Submission ZIP (continuity Dev best): `exp/inference/blindset_A/submission_exp021_cont_devbest_A.zip`
- Submission JSON (continuity top100 tail): `exp/inference/blindset_A/exp021_cont_top100_A.json`
- Submission ZIP (continuity top100 tail): `exp/inference/blindset_A/submission_exp021_cont_top100_A.zip`

## Validation

```text
validate_submission.py exp/inference/blindset_A/exp021_qnm_p1_A.json
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/submission_exp021_qnm_p1_A.zip
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/exp021_qnm_devbest_A.json
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/submission_exp021_qnm_devbest_A.zip
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/submission_exp021_cont_p1safe_A.zip --catalog cache/bm25/track_name_artist_name_album_name_tag_list/track_ids.json
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/submission_exp021_cont_devbest_A.zip --catalog cache/bm25/track_name_artist_name_album_name_tag_list/track_ids.json
records: 80
OK: 提出制約をすべて満たす。

validate_submission.py exp/inference/blindset_A/submission_exp021_cont_top100_A.zip --catalog cache/bm25/track_name_artist_name_album_name_tag_list/track_ids.json
records: 80
OK: 提出制約をすべて満たす。
```

ZIP root はどちらも `prediction.json` のみ。submission zip 名は 63 文字未満。追加で
`cache/bm25/track_name_artist_name_album_name_tag_list/track_ids.json`（47,071 tracks）を
catalog として指定した検証も JSON / ZIP の 4 件すべてで OK。

## Sample Prediction Output

```json
[
  {
    "session_id": "9c37dcd7-d7c2-4686-8541-1e37c4814a09",
    "user_id": "c3233a5c-da6c-42a3-9459-83ee1134e207",
    "turn_number": 1,
    "track_count": 20,
    "top3": [
      "91e3b516-8d17-4031-b75f-4022c5fab076",
      "c1e7b924-053c-4c35-8519-d5f1cb2eaa25",
      "59f8be05-193a-4112-87f3-1affe08d9865"
    ],
    "resp_len": 381
  }
]
```

## Interpretation

QNM は standalone では弱いが、history_music variant は base miss の 19.3% を top500 で回収し、既存 source とは異なる候補を持っていた。最終的な top20 fusion では平均 overlap20 が 20.0 で、主な効果は新規 track 追加ではなく top20 内の順位再調整だった。

Continuity source は EDA どおり QNM より強く、`album_artist` 単体で nDCG@20 `0.15199`、hit@500 `3544/8000`。top20 fusion は QNM best から +0.00113 と、この turn の中では最も大きい改善だった。5 fold すべて positive で、Blind A top1 も動かさなかったため、QNM より採用優先度は高い。

Dev highest は top100 tail の `0.1981492892` だが、top20 continuity からの差は +0.0000149 で 1 fold negative。Blind A で `ndcg@20 >= 0.5` を達成できる根拠はまだない。提出するなら、まず top1 mismatch 0 の `submission_exp021_cont_p1safe_A.zip`、次に `submission_exp021_cont_devbest_A.zip`、top100 tail は提出枠に余裕がある場合の確認用。

## Leaderboard Result

- Submitted: yes
- Date: 2026-06-17

### Blind A Summary

| Artifact | ndcg@20 | catalog_diversity | lexical_diversity | llm_judge_score | composite_score |
|---|---:|---:|---:|---:|---:|
| `submission_exp021_qnm_devbest_A.zip` | 0.3408 | 0.0304 | 0.5837 | 4.0000 | 0.4568 |
| `submission_exp021_qnm_p1_A.zip` | 0.3409 | 0.0304 | 0.5837 | 3.9500 | 0.4531 |
| `submission_exp021_cont_top100_A.zip` | 0.3565 | 0.0301 | 0.5837 | 4.0500 | 0.4684 |
| `submission_exp021_cont_p1safe_A.zip` | **0.3600** | 0.0301 | 0.5837 | 3.9000 | 0.4589 |
| `submission_exp021_cont_devbest_A.zip` | 0.3565 | 0.0301 | 0.5837 | 4.0000 | 0.4646 |

#### `submission_exp021_qnm_devbest_A.zip`

```text
00:08:06 [INFO] ========== Final Score ==========
00:08:06 [INFO]   ndcg@20                   0.3408
00:08:06 [INFO]   catalog_diversity         0.0304
00:08:06 [INFO]   lexical_diversity         0.5837
00:08:06 [INFO]   llm_judge_score           4.0000
00:08:06 [INFO]   composite_score           0.4568
00:08:06 [INFO] =================================
```

#### `submission_exp021_qnm_p1_A.zip`

```text
00:08:03 [INFO] ========== Final Score ==========
00:08:03 [INFO]   ndcg@20                   0.3409
00:08:03 [INFO]   catalog_diversity         0.0304
00:08:03 [INFO]   lexical_diversity         0.5837
00:08:03 [INFO]   llm_judge_score           3.9500
00:08:03 [INFO]   composite_score           0.4531
00:08:03 [INFO] =================================
```

#### `submission_exp021_cont_top100_A.zip`

```text
14:19:04 [INFO] ========== Final Score ==========
14:19:04 [INFO]   ndcg@20                   0.3565
14:19:04 [INFO]   catalog_diversity         0.0301
14:19:04 [INFO]   lexical_diversity         0.5837
14:19:04 [INFO]   llm_judge_score           4.0500
14:19:04 [INFO]   composite_score           0.4684
14:19:04 [INFO] =================================
```

#### `submission_exp021_cont_p1safe_A.zip`

```text
14:19:12 [INFO] ========== Final Score ==========
14:19:12 [INFO]   ndcg@20                   0.3600
14:19:12 [INFO]   catalog_diversity         0.0301
14:19:12 [INFO]   lexical_diversity         0.5837
14:19:12 [INFO]   llm_judge_score           3.9000
14:19:12 [INFO]   composite_score           0.4589
14:19:12 [INFO] =================================
```

#### `submission_exp021_cont_devbest_A.zip`

```text
14:19:28 [INFO] ========== Final Score ==========
14:19:28 [INFO]   ndcg@20                   0.3565
14:19:28 [INFO]   catalog_diversity         0.0301
14:19:28 [INFO]   lexical_diversity         0.5837
14:19:28 [INFO]   llm_judge_score           4.0000
14:19:28 [INFO]   composite_score           0.4646
14:19:28 [INFO] =================================
```

Interpretation: QNM は Blind A で `ndcg@20` が `0.3408-0.3409` に留まり、continuity 系が上回った。continuity では p1safe が `ndcg@20=0.3600` で最良だが、judge が `3.9000` に落ち composite は `0.4589`。top100 は nDCG では同等以下だが judge `4.0500` により composite `0.4684` が exp021 内最高。

## Next Actions

- exp021 単体では continuity top100 / devbest / p1safe の差は Blind A noise floor 内。以後は exp022 の continuity LGBM を優先する。
- Continuity は source として強いので、次は LightGBM candidate build に feature/source として入れるか、turn>=2 bucket 専用 gate を作る。
