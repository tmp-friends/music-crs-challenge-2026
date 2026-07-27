# exp024_transition_memory

## 目的

train split の「prior music entity -> next label track」遷移を候補 source 化し、exp023 に tail RRF で足す。

## 実行メモ

初回の `recent1,recent2` 実行では `recent2` の index 構築が重く、途中で中断した。`transition_memory.py` は構築済み index を dev / Blind で使い回すよう修正済み。最終実行は `recent1` のみに絞った。

```bash
.venv/bin/python mcrs/experiments/exp024_transition_memory/transition_memory.py \
  --blind_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-A \
  --topk 100 \
  --variants recent1 \
  --output_dir mcrs/experiments/exp024_transition_memory/results/transition_memory_top100

.venv/bin/python mcrs/experiments/exp021_candidate_fusion/greedy_rrf_sweep.py \
  --primary exp023=mcrs/experiments/exp023_entity_candidates/results/dev_tracks_exp023_entity_after_exp022_oof_rrf.json \
  --source transition_recent1=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100/dev_tracks_exp024_transition_recent1.json \
  --blind_primary exp023=exp/inference/blindset_A/exp023_entity_A.json \
  --blind_source transition_recent1=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100/blindA_tracks_exp024_transition_recent1.json \
  --source_topk 100 --fusion_ks 1,2,3,5,10,20,60 \
  --primary_weights 0.940:0.995:0.005 --protect_tops 1,5,10,15 --max_steps 2 \
  --output_json mcrs/experiments/exp024_transition_memory/results/transition_recent1_after_exp023_rrf.json \
  --output_dev_tracks mcrs/experiments/exp024_transition_memory/results/dev_tracks_exp024_transition_after_exp023_rrf.json \
  --output_blind_tracks mcrs/experiments/exp024_transition_memory/results/blindA_tracks_exp024_transition_after_exp023_rrf.json
```

## 結果

- `recent1` standalone: nDCG@20 `0.0036967252`, hit@20 `87`, recall@100 `0.056875`
- exp023 primary: `0.2000946602`
- final Dev: `0.2001032073` (`+0.0000085471`)
- adopted step: `transition_recent1`, `fusion_k=60`, `primary_weight=0.98`, `protect_top=1`
- Blind A change: `changed=23/80`, `top1_changed=0`, `avg_overlap20=20.0`

## Artifacts

- Transition summary: `mcrs/experiments/exp024_transition_memory/results/transition_memory_top100/transition_memory_summary.json`
- RRF report: `mcrs/experiments/exp024_transition_memory/results/transition_recent1_after_exp023_rrf.json`
- Dev final: `mcrs/experiments/exp024_transition_memory/results/dev_tracks_exp024_transition_after_exp023_rrf.json`
- Blind A JSON: `exp/inference/blindset_A/exp024_transition_A.json`
- Blind A zip: `exp/inference/blindset_A/submission_exp024_transition_A.zip`

## Validation

`exp024_transition_A.json` と `submission_exp024_transition_A.zip` は catalog 47,071 tracks 照合で `records: 80`, `OK`。ZIP root は `prediction.json`。response attach は `top1_mismatch=0`, `empty_response=0`。

## Leaderboard Result

- Submitted: yes
- Date: 2026-06-17
- Artifact: `exp/inference/blindset_A/submission_exp024_transition_A.zip`

```text
15:42:39 [INFO] ========== Final Score ==========
15:42:39 [INFO]   ndcg@20                   0.3739
15:42:39 [INFO]   catalog_diversity         0.0303
15:42:39 [INFO]   lexical_diversity         0.5837
15:42:39 [INFO]   llm_judge_score           4.0000
15:42:39 [INFO]   composite_score           0.4734
15:42:39 [INFO] =================================
15:42:39 [INFO] scores.json written.
```

Interpretation: ranking は exp023 と同等だが judge が `4.1000` から `4.0000` に下がり composite は `-0.0075`。transition memory は Blind A では採用しない。
