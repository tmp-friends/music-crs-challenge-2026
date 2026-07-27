#!/usr/bin/env bash
# Blind B smoke: collapse リスクの高い候補 source generator を Blind B で再生成する。
# 目的は「cold(空 user_id)/ single-turn セッションが候補プールを silent に失わないか」の計測。
# 既存 Blind A 成果物を上書きしないよう output_dir を smoke 専用に分離する。
# dev 再生成は --skip_dev で回避。transition は train index 構築が重く collapse リスクも低いので除外。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BLIND="talkpl-ai/TalkPlayData-Challenge-Blind-B"
ROOT="exp/smoke_blindB/sources"
PY=".venv/bin/python"
mkdir -p "${ROOT}"

run() {  # name cmd...
  local name="$1"; shift
  echo "===== [${name}] start ====="
  if "$@"; then echo "===== [${name}] OK ====="; else echo "===== [${name}] FAILED (exit $?) ====="; fi
}

# 1. continuity（in-session music 履歴ベース。single-turn cold で履歴ゼロ→collapse 候補）
run continuity ${PY} mcrs/experiments/exp021_candidate_fusion/continuity_candidates.py \
  --skip_dev --blind_dataset_name "${BLIND}" --topk 100 \
  --output_dir "${ROOT}/continuity"

# 2. qnm（current / recent_profile / history_music。profile・履歴依存）
run qnm ${PY} mcrs/experiments/exp021_candidate_fusion/query_neighbor_memory.py \
  --skip_dev --blind_dataset_name "${BLIND}" --topk 100 \
  --variants "current,recent_profile,history_music" \
  --output_dir "${ROOT}/qnm"

# 3. entity（会話 entity ベース。user 依存は低いが cold で会話のみから抽出できるか確認）
run entity ${PY} mcrs/experiments/exp023_entity_candidates/entity_candidates.py \
  --skip_dev --blind_dataset_name "${BLIND}" --topk 100 \
  --variants "current,recent2,recent4,current_tight" \
  --output_dir "${ROOT}/entity"

# 4. cross_session（user_id 依存。cold=空 user_id は "" キーに集約される最大の collapse 源）
#    bankable は --no_pad（nopad）を使う。dev pool + Blind B から cross 候補を作る。
run cross_session ${PY} mcrs/experiments/exp090_cross_session_source/cross_session_candidates.py \
  --blind_dataset_name "${BLIND}" --no_pad --cap 50 \
  --output_dir "${ROOT}/cross_session"

echo "ALL DONE"
