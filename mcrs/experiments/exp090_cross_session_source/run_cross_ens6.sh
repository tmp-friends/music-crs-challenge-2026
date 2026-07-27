#!/usr/bin/env bash
# exp090 cross-session source の 6-seed ensemble（exp058 と同一 denoise）+ Blind A tracks。
# exp058 の 12 source に cross_session を足し、6 seed の score-average で再ランキング。
# OOF は single-seed で floor クリア済みなので --skip_oof で Blind/full のみ。
set -euo pipefail
cd /home/tomoya/kaggle/music-crs-baselines

OUT=mcrs/experiments/exp090_cross_session_source/results
D21=mcrs/experiments/exp021_candidate_fusion/results
D23=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100
D24=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100

.venv/bin/python mcrs/experiments/exp058_wide100_seed_ensemble/seed_ensemble_wide100.py \
  --seeds "20260616,20260617,20260618,20260619,20260620,20260621" \
  --skip_oof \
  --primary "exp015B=mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json" \
  --blind_primary "exp015B=exp/inference/blindset_A/exp015_B_ll_t500_A.json" \
  --source "cont_aat=${D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist_tag.json" \
  --source "cont_aa=${D21}/continuity_candidates/dev_tracks_exp021_cont_album_artist.json" \
  --source "cont_album=${D21}/continuity_candidates/dev_tracks_exp021_cont_album.json" \
  --source "cont_artist=${D21}/continuity_candidates/dev_tracks_exp021_cont_artist.json" \
  --source "qnm_rp=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_recent_profile.json" \
  --source "qnm_hist=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_history_music.json" \
  --source "qnm_cur=${D21}/query_neighbor_memory/dev_tracks_exp021_qnm_current.json" \
  --source "ent_cur=${D23}/dev_tracks_exp023_entity_current.json" \
  --source "ent_tight=${D23}/dev_tracks_exp023_entity_current_tight.json" \
  --source "ent_r2=${D23}/dev_tracks_exp023_entity_recent2.json" \
  --source "ent_r4=${D23}/dev_tracks_exp023_entity_recent4.json" \
  --source "trans_r1=${D24}/dev_tracks_exp024_transition_recent1.json" \
  --source "cross_session:50=${OUT}/dev_tracks_cross_session.json" \
  --blind_source "cont_aat=${D21}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album_artist_tag.json" \
  --blind_source "cont_aa=${D21}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album_artist.json" \
  --blind_source "cont_album=${D21}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album.json" \
  --blind_source "cont_artist=${D21}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_artist.json" \
  --blind_source "qnm_rp=${D21}/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_recent_profile.json" \
  --blind_source "qnm_hist=${D21}/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_history_music.json" \
  --blind_source "qnm_cur=${D21}/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_current.json" \
  --blind_source "ent_cur=${D23}/blindA_tracks_exp023_entity_current.json" \
  --blind_source "ent_tight=${D23}/blindA_tracks_exp023_entity_current_tight.json" \
  --blind_source "ent_r2=${D23}/blindA_tracks_exp023_entity_recent2.json" \
  --blind_source "ent_r4=${D23}/blindA_tracks_exp023_entity_recent4.json" \
  --blind_source "trans_r1=${D24}/blindA_tracks_exp024_transition_recent1.json" \
  --blind_source "cross_session:50=${OUT}/blindA_tracks_cross_session.json" \
  --protect_top 0 \
  --num_boost_round 450 \
  --oof_num_boost_round 240 \
  --num_leaves 31 \
  --min_data_in_leaf 80 \
  --lambda_l2 8.0 \
  --num_threads 8 \
  --output_report "${OUT}/ranker/report_cross_ens6.json" \
  --output_dev_full "${OUT}/ranker/dev_full_cross_ens6.json" \
  --output_blind_tracks "${OUT}/ranker/blindA_tracks_cross_ens6.json"
