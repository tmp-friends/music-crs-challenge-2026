#!/usr/bin/env bash
# exp058 wide100 seed ensemble driver。
# 引数:
#   $1 = seeds (カンマ区切り)         例: 20260616
#   $2 = extra flags                 例: --skip_oof
#   $3 = tag suffix (出力 slug)      例: s1
set -euo pipefail
cd /home/tomoya/kaggle/music-crs-baselines

SEEDS="${1:-20260616,20260617,20260618,20260619,20260620,20260621}"
EXTRA="${2:-}"
TAG="${3:-ens6}"

D21=mcrs/experiments/exp021_candidate_fusion/results
D21B=mcrs/experiments/exp021_candidate_fusion/results
D23=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100
D24=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100
OUT=mcrs/experiments/exp058_wide100_seed_ensemble/results

.venv/bin/python mcrs/experiments/exp058_wide100_seed_ensemble/seed_ensemble_wide100.py \
  --seeds "${SEEDS}" ${EXTRA} \
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
  --blind_source "cont_aat=${D21B}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album_artist_tag.json" \
  --blind_source "cont_aa=${D21B}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album_artist.json" \
  --blind_source "cont_album=${D21B}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_album.json" \
  --blind_source "cont_artist=${D21B}/continuity_candidates_blindA_top100/blindA_tracks_exp021_cont_artist.json" \
  --blind_source "qnm_rp=${D21B}/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_recent_profile.json" \
  --blind_source "qnm_hist=${D21B}/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_history_music.json" \
  --blind_source "qnm_cur=${D21B}/query_neighbor_memory_blindA/blindA_tracks_exp021_qnm_current.json" \
  --blind_source "ent_cur=${D23}/blindA_tracks_exp023_entity_current.json" \
  --blind_source "ent_tight=${D23}/blindA_tracks_exp023_entity_current_tight.json" \
  --blind_source "ent_r2=${D23}/blindA_tracks_exp023_entity_recent2.json" \
  --blind_source "ent_r4=${D23}/blindA_tracks_exp023_entity_recent4.json" \
  --blind_source "trans_r1=${D24}/blindA_tracks_exp024_transition_recent1.json" \
  --output_report "${OUT}/report_${TAG}.json" \
  --output_dev_full "${OUT}/dev_full_${TAG}.json" \
  --output_dev_oof "${OUT}/dev_oof_${TAG}.json" \
  --output_blind_tracks "${OUT}/blindA_tracks_${TAG}.json"
