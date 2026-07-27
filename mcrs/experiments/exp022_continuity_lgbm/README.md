# exp022_continuity_lgbm

## 目的

exp021 の continuity / query-neighbor memory source を rank-only feature として LightGBM ranker に入れ、top20 の順位改善を狙う。

## 実行コマンド

```bash
.venv/bin/python mcrs/experiments/exp022_continuity_lgbm/rank_source_lgbm.py \
  --primary primary=mcrs/experiments/exp021_candidate_fusion/results/dev_tracks_exp021_continuity_top100_tail_after_devbest_rrf.json \
  --source cont_album=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates/dev_tracks_exp021_cont_album.json \
  --source cont_artist=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates/dev_tracks_exp021_cont_artist.json \
  --source cont_album_artist=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates/dev_tracks_exp021_cont_album_artist.json \
  --source cont_album_artist_tag=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates/dev_tracks_exp021_cont_album_artist_tag.json \
  --source qnm_current=mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory/dev_tracks_exp021_qnm_current.json \
  --source qnm_recent_profile=mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory/dev_tracks_exp021_qnm_recent_profile.json \
  --source qnm_history_music=mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory/dev_tracks_exp021_qnm_history_music.json \
  --blind_primary primary=mcrs/experiments/exp021_candidate_fusion/results/blindA_tracks_exp021_continuity_top100_tail_after_devbest_rrf.json \
  --blind_source cont_album=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album.json \
  --blind_source cont_artist=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_artist.json \
  --blind_source cont_album_artist=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album_artist.json \
  --blind_source cont_album_artist_tag=mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album_artist_tag.json \
  --blind_source qnm_current=mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_current.json \
  --blind_source qnm_recent_profile=mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_recent_profile.json \
  --blind_source qnm_history_music=mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_history_music.json \
  --source_topk 100 --protect_top 1 \
  --output_report mcrs/experiments/exp022_continuity_lgbm/results/exp022_continuity_lgbm_report.json \
  --output_dev_full mcrs/experiments/exp022_continuity_lgbm/results/dev_tracks_exp022_cont_lgbm_full.json \
  --output_dev_oof mcrs/experiments/exp022_continuity_lgbm/results/dev_tracks_exp022_cont_lgbm_oof.json \
  --output_blind_tracks mcrs/experiments/exp022_continuity_lgbm/results/blindA_tracks_exp022_cont_lgbm_full.json
```

## 結果

- anchor exp021: `0.1981492892`
- full-fit Dev: `0.2243695306` (`+0.0262202414`)
- 5-fold OOF Dev: `0.2000784994` (`+0.0019292102`)
- Blind A change: `changed=77/80`, `top1_changed=0`, `avg_overlap20=18.65`

full-fit は in-sample が強すぎるため、以降の Dev 比較では OOF を主系列として使う。

## Artifacts

- Dev OOF: `mcrs/experiments/exp022_continuity_lgbm/results/dev_tracks_exp022_cont_lgbm_oof.json`
- Blind A JSON: `exp/inference/blindset_A/exp022_cont_lgbm_A.json`
- Blind A zip: `exp/inference/blindset_A/submission_exp022_cont_lgbm_A.zip`
- Report: `mcrs/experiments/exp022_continuity_lgbm/results/exp022_continuity_lgbm_report.json`

## Validation

```bash
.venv/bin/python .agents/skills/music-crs-dev-cycle/scripts/validate_submission.py exp/inference/blindset_A/exp022_cont_lgbm_A.json --catalog cache/bm25/track_name_artist_name_album_name_tag_list/track_ids.json
.venv/bin/python .agents/skills/music-crs-dev-cycle/scripts/validate_submission.py exp/inference/blindset_A/submission_exp022_cont_lgbm_A.zip --catalog cache/bm25/track_name_artist_name_album_name_tag_list/track_ids.json
zipinfo -1 exp/inference/blindset_A/submission_exp022_cont_lgbm_A.zip
```

両方 `records: 80`, `OK`。ZIP root は `prediction.json`。

## Leaderboard Result

- Submitted: yes
- Date: 2026-06-17
- Artifact: `exp/inference/blindset_A/submission_exp022_cont_lgbm_A.zip`

```text
15:42:14 [INFO] ========== Final Score ==========
15:42:14 [INFO]   ndcg@20                   0.3743
15:42:14 [INFO]   catalog_diversity         0.0303
15:42:14 [INFO]   lexical_diversity         0.5837
15:42:14 [INFO]   llm_judge_score           4.1000
15:42:14 [INFO]   composite_score           0.4810
15:42:14 [INFO] =================================
```

Interpretation: 今回提示された exp021-025 の中では Blind A `ndcg@20` / composite ともに最高。exp021 continuity p1safe から nDCG `+0.0143`、exp021 top100 から composite `+0.0126`。ただし exp015 B の nDCG gate `0.3869` は未達。
