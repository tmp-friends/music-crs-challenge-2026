#!/usr/bin/env bash
# exp116 lxct6 (lgbm+xgb+catboost+tabm 各 6-seed 対称) を Blind B に適用する。
# 学習側（dev --source / params / exclude の Dev 部分）は run_ens_lxct6.sh と byte-identical。
# blind 側のみ Blind B 用の再生成 source / 修正版 cross_session / Blind B exclude に差し替える。
#   - blind_primary : exp015 B を Blind B で推論した候補（exp/inference/blindset_B/）
#   - blind_source  : exp/smoke_blindB/sources/ 配下で Blind B 再生成したもの
#                     （continuity/qnm/entity/transition は --blind_dataset_name Blind-B で生成、
#                       cross_session は空 user_id を no-pool 化した leak-safe 修正版）
#   - exclude_json  : Dev + Blind B の within-session 既出 track（Blind B 分を regenerate）
# REDUCED=1 で軽量設定（1 seed / 低 round / OOF skip）にし、本番前の統合 dry-run に使える。
set -euo pipefail
cd /home/tomoya/kaggle/music-crs-baselines

OUT=mcrs/experiments/exp116_gbdt_family_ensemble/results
CS=mcrs/experiments/exp090_cross_session_source/results
D21=mcrs/experiments/exp021_candidate_fusion/results
D23=mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100
D24=mcrs/experiments/exp024_transition_memory/results/transition_memory_top100

# Blind B 再生成 source の置き場
BB=exp/smoke_blindB/sources
BB_CONT=${BB}/continuity
BB_QNM=${BB}/qnm
BB_ENT=${BB}/entity
BB_TR=${BB}/transition
BB_CS=${BB}/cross_session
BB_PRIMARY=exp/inference/blindset_B/exp015_B_ll_t500_B.json
BB_EXCLUDE=${BB}/within_session_exclude_blindB.json

REDUCED="${REDUCED:-0}"
if [ "${REDUCED}" = "1" ]; then
  # 統合 dry-run: 各 family 1 seed・低 round・OOF skip。join/shape と 80 行出力の検証用。
  SEEDS="20260616"; XN=1; CN=1; TN=1
  OOF_ARGS="--skip_oof"
  ROUNDS="--num_boost_round 60 --xgb_rounds 60 --catboost_iters 120 --tabm_epochs 2"
  SUFFIX="blindB_dryrun"
else
  SEEDS="20260616,20260617,20260618,20260619,20260620,20260621"; XN=6; CN=6; TN=6
  OOF_ARGS="--oof_lgbm_n 6 --oof_xgb_n 6 --oof_catboost_n 6 --oof_tabm_n 6"
  ROUNDS="--num_boost_round 450 --oof_num_boost_round 240 --xgb_rounds 450 --catboost_iters 1000 --tabm_epochs 6"
  SUFFIX="blindB_lxct6"
fi

.venv/bin/python mcrs/experiments/exp116_gbdt_family_ensemble/ensemble_rerank.py \
  --families "lgbm,xgb,catboost,tabm" \
  --seeds "${SEEDS}" \
  --xgb_n ${XN} --catboost_n ${CN} --tabm_n ${TN} \
  ${OOF_ARGS} \
  --xgb_device cuda --catboost_task_type GPU --tabm_device cuda \
  --primary "exp015B=mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json" \
  --blind_primary "exp015B=${BB_PRIMARY}" \
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
  --source "cross_session:50=${CS}/nopad/dev_tracks_cross_session.json" \
  --blind_source "cont_aat=${BB_CONT}/blindA_tracks_exp021_cont_album_artist_tag.json" \
  --blind_source "cont_aa=${BB_CONT}/blindA_tracks_exp021_cont_album_artist.json" \
  --blind_source "cont_album=${BB_CONT}/blindA_tracks_exp021_cont_album.json" \
  --blind_source "cont_artist=${BB_CONT}/blindA_tracks_exp021_cont_artist.json" \
  --blind_source "qnm_rp=${BB_QNM}/blindA_tracks_exp021_qnm_recent_profile.json" \
  --blind_source "qnm_hist=${BB_QNM}/blindA_tracks_exp021_qnm_history_music.json" \
  --blind_source "qnm_cur=${BB_QNM}/blindA_tracks_exp021_qnm_current.json" \
  --blind_source "ent_cur=${BB_ENT}/blindA_tracks_exp023_entity_current.json" \
  --blind_source "ent_tight=${BB_ENT}/blindA_tracks_exp023_entity_current_tight.json" \
  --blind_source "ent_r2=${BB_ENT}/blindA_tracks_exp023_entity_recent2.json" \
  --blind_source "ent_r4=${BB_ENT}/blindA_tracks_exp023_entity_recent4.json" \
  --blind_source "trans_r1=${BB_TR}/blindA_tracks_exp024_transition_recent1.json" \
  --blind_source "cross_session:50=${BB_CS}/blindA_tracks_cross_session.json" \
  --exclude_json "${BB_EXCLUDE}" \
  --allow_short_sources \
  --protect_top 0 \
  ${ROUNDS} \
  --num_leaves 31 --min_data_in_leaf 80 --lambda_l2 8.0 --num_threads 8 \
  --output_report "${OUT}/ranker/report_ens_${SUFFIX}.json" \
  --output_dev_full "${OUT}/ranker/dev_full_ens_${SUFFIX}.json" \
  --output_blind_tracks "${OUT}/ranker/blindB_tracks_ens_${SUFFIX}.json"
