# exp023_entity_candidates

## 目的

現在 turn / recent user text から track metadata entity phrase を拾い、catalog candidate source として exp022 OOF に足す。

## 実行コマンド

```bash
.venv/bin/python mcrs/experiments/exp023_entity_candidates/entity_candidates.py \
  --blind_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-A \
  --topk 100 \
  --output_dir mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100

.venv/bin/python mcrs/experiments/exp021_candidate_fusion/greedy_rrf_sweep.py \
  --primary exp022_oof=mcrs/experiments/exp022_continuity_lgbm/results/dev_tracks_exp022_cont_lgbm_oof.json \
  --source ent_current=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/dev_tracks_exp023_entity_current.json \
  --source ent_recent2=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/dev_tracks_exp023_entity_recent2.json \
  --source ent_recent4=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/dev_tracks_exp023_entity_recent4.json \
  --source ent_tight=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/dev_tracks_exp023_entity_current_tight.json \
  --blind_primary exp022=exp/inference/blindset_A/exp022_cont_lgbm_A.json \
  --blind_source ent_current=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/blindA_tracks_exp023_entity_current.json \
  --blind_source ent_recent2=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/blindA_tracks_exp023_entity_recent2.json \
  --blind_source ent_recent4=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/blindA_tracks_exp023_entity_recent4.json \
  --blind_source ent_tight=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/blindA_tracks_exp023_entity_current_tight.json \
  --source_topk 100 --fusion_ks 1,2,3,5,10,20,60 \
  --primary_weights 0.940:0.995:0.005 --protect_tops 1,5,10,15 --max_steps 3 \
  --output_json mcrs/experiments/exp023_entity_candidates/results/entity_after_exp022_oof_rrf.json \
  --output_dev_tracks mcrs/experiments/exp023_entity_candidates/results/dev_tracks_exp023_entity_after_exp022_oof_rrf.json \
  --output_blind_tracks mcrs/experiments/exp023_entity_candidates/results/blindA_tracks_exp023_entity_after_exp022_rrf.json
```

## 結果

Standalone source は弱いが、tail に少量入れると微増した。

- exp022 OOF primary: `0.2000784994`
- final Dev: `0.2000946602` (`+0.0000161608`)
- adopted steps: `ent_tight`, `ent_current`, `ent_recent4`
- Blind A change: `changed=59/80`, `top1_changed=0`, `avg_overlap20=20.0`

## Artifacts

- Source summary: `mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100/entity_candidates_summary.json`
- RRF report: `mcrs/experiments/exp023_entity_candidates/results/entity_after_exp022_oof_rrf.json`
- Dev final: `mcrs/experiments/exp023_entity_candidates/results/dev_tracks_exp023_entity_after_exp022_oof_rrf.json`
- Blind A JSON: `exp/inference/blindset_A/exp023_entity_A.json`
- Blind A zip: `exp/inference/blindset_A/submission_exp023_entity_A.zip`

## Validation

`exp023_entity_A.json` と `submission_exp023_entity_A.zip` は catalog 47,071 tracks 照合で `records: 80`, `OK`。ZIP root は `prediction.json`。response attach は `top1_mismatch=0`, `empty_response=0`。

## Leaderboard Result

- Submitted: yes
- Date: 2026-06-17
- Artifact: `exp/inference/blindset_A/submission_exp023_entity_A.zip`

```text
15:42:34 [INFO] ========== Final Score ==========
15:42:34 [INFO]   ndcg@20                   0.3739
15:42:34 [INFO]   catalog_diversity         0.0303
15:42:34 [INFO]   lexical_diversity         0.5837
15:42:34 [INFO]   llm_judge_score           4.1000
15:42:34 [INFO]   composite_score           0.4809
15:42:34 [INFO] =================================
```

Interpretation: Dev では exp022 OOF から微増したが、Blind A では exp022 より `ndcg@20 -0.0004`, composite `-0.0001`。entity tail は採用差分としては noise 内で、exp022 を優先する。
