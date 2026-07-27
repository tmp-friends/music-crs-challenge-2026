#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

EXP_DIR="mcrs/experiments/exp015_candidate_rules_ablation"
RESULT_ROOT="${RESULT_ROOT:-${EXP_DIR}/results}"
TOPKS="${TOPKS:-500}"
ONLY_RUNS="${ONLY_RUNS:-A,B,C,D,E,F,G}"
OBJECTIVE="${OBJECTIVE:-logloss}"
FORCE="${FORCE:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
DEV_LIMIT="${DEV_LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_ROWS_PER_GROUP="${MAX_ROWS_PER_GROUP:-120}"
HEAD_NEGATIVES="${HEAD_NEGATIVES:-80}"

RUN_IDS=("A" "B" "C" "D" "E" "F" "G")
RUN_SLUGS=("base" "exact_split" "metadata_split" "segment_pop" "history_split" "intent_gate" "quota_union")

should_run() {
  local run_id="$1"
  if [ "${ONLY_RUNS}" = "all" ]; then
    return 0
  fi
  local normalized=",${ONLY_RUNS// /},"
  [[ "${normalized}" == *",${run_id},"* ]]
}

ensure_config() {
  local run_id="$1"
  local topk="$2"
  local output_config="$3"
  .venv/bin/python - "$run_id" "$topk" "$output_config" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

from mcrs.experiments.exp015_candidate_rules_ablation.candidate_sources import source_profile_to_sources

run_id = sys.argv[1]
topk = int(sys.argv[2])
output_config = Path(sys.argv[3])
base_config = Path("mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_base_s05_ll_t500_A.yaml")

config = OmegaConf.load(base_config)
config.test_dataset_name = "talkpl-ai/TalkPlayData-Challenge-Dataset"
config.conversation_dataset_name = "talkpl-ai/TalkPlayData-Challenge-Dataset"
config.source_profile = run_id
config.union_mode = "quota" if run_id == "G" else "rrf"
config.enabled_sources = list(source_profile_to_sources(run_id))
config.retrieval_topk = topk
config.track_split_types = ["all_tracks"]
output_config.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config=config, f=output_config)
print(output_config)
PY
}

write_summary_csv() {
  .venv/bin/python - "$RESULT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("top*/*/metrics.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "candidate_topk": data.get("candidate_topk", ""),
        "run_id": data.get("run_id", ""),
        "source_profile": data.get("source_profile", ""),
        "objective": data.get("objective", ""),
        "enabled_sources": ",".join(data.get("enabled_sources", [])),
        "valid_groups_total": data.get("valid_groups_total", ""),
        "valid_groups_with_positive": data.get("valid_groups_with_positive", ""),
        "valid_ndcg@20_positive_groups": data.get("valid_ndcg@20_positive_groups", ""),
        "all_tasks_ndcg@20": data.get("all_tasks_ndcg@20", ""),
        "all_tasks_recall@20": data.get("all_tasks_recall@20", ""),
        "candidate_recall@20": data.get("candidate_recall@20", ""),
        "candidate_recall@100": data.get("candidate_recall@100", ""),
        "candidate_recall@500": data.get("candidate_recall@500", ""),
        "oracle_ndcg@20": data.get("oracle_ndcg@20", ""),
        "unique_top20": data.get("unique_top20", ""),
        "train_unknown_candidate_count": data.get("train_unknown_candidate_count", ""),
        "dev_unknown_candidate_count": data.get("dev_unknown_candidate_count", ""),
    })
out = root / "exp015_candidate_rules_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "candidate_topk",
    "run_id",
    "source_profile",
    "objective",
    "enabled_sources",
    "valid_groups_total",
    "valid_groups_with_positive",
    "valid_ndcg@20_positive_groups",
    "all_tasks_ndcg@20",
    "all_tasks_recall@20",
    "candidate_recall@20",
    "candidate_recall@100",
    "candidate_recall@500",
    "oracle_ndcg@20",
    "unique_top20",
    "train_unknown_candidate_count",
    "dev_unknown_candidate_count",
]
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote_summary={out} rows={len(rows)}")
PY
}

combine_metrics() {
  .venv/bin/python - "$@" <<'PY'
import json
import sys
from pathlib import Path

run_id, objective, topk = sys.argv[1], sys.argv[2], int(sys.argv[3])
build_path = Path(sys.argv[4])
train_metrics_path = Path(sys.argv[5])
prediction_path = Path(sys.argv[6])
output_path = Path(sys.argv[7])

build = json.loads(build_path.read_text(encoding="utf-8"))
train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
dev_build = build.get("dev", {})
train_build = build.get("train", {})
valid = train_metrics.get("valid", {})
top20 = [track_id for row in predictions for track_id in row.get("predicted_track_ids", [])[:20]]
data = {
    "experiment": "exp015_candidate_rules_ablation",
    "run_id": run_id,
    "source_profile": run_id,
    "objective": objective,
    "candidate_topk": topk,
    "enabled_sources": build.get("enabled_sources", []),
    "train_groups_before_filter": train_build.get("groups_before_filter", ""),
    "train_groups_after_filter": train_metrics.get("train_groups_after_filter", ""),
    "train_groups_dropped_no_positive": train_metrics.get("train_groups_dropped_no_positive", ""),
    "valid_groups_total": train_metrics.get("valid_groups_total", dev_build.get("groups_after_filter", "")),
    "valid_groups_with_positive": train_metrics.get("valid_groups_with_positive", valid.get("groups_with_positive", "")),
    "valid_ndcg@20_positive_groups": train_metrics.get("valid_ndcg@20_positive_groups", valid.get("ndcg@20", "")),
    "all_tasks_ndcg@20": train_metrics.get("all_tasks_ndcg@20", ""),
    "all_tasks_recall@20": train_metrics.get("all_tasks_recall@20", ""),
    "candidate_recall@20": dev_build.get("candidate_recall@20", ""),
    "candidate_recall@100": dev_build.get("candidate_recall@100", ""),
    "candidate_recall@500": dev_build.get("candidate_recall@500", ""),
    "oracle_ndcg@20": dev_build.get("oracle_ndcg@20", ""),
    "unique_top20": len(set(top20)),
    "train_unknown_candidate_count": train_build.get("unknown_candidate_count", ""),
    "dev_unknown_candidate_count": dev_build.get("unknown_candidate_count", ""),
}
output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote_metrics={output_path}")
PY
}

mkdir -p "${RESULT_ROOT}"
IFS=',' read -r -a TOPK_LIST <<< "${TOPKS// /}"

for topk in "${TOPK_LIST[@]}"; do
  for index in "${!RUN_IDS[@]}"; do
    run_id="${RUN_IDS[$index]}"
    slug="${RUN_SLUGS[$index]}"
    if ! should_run "${run_id}"; then
      echo "SKIP run ${run_id} by ONLY_RUNS=${ONLY_RUNS}"
      continue
    fi

    run_dir="${RESULT_ROOT}/top${topk}/${run_id}_${slug}"
    config_path="${EXP_DIR}/configs/exp015_${run_id}_t${topk}_dev.yaml"
    train_data="${run_dir}/train_${run_id}_top${topk}.parquet"
    dev_data="${run_dir}/dev_${run_id}_top${topk}.parquet"
    build_summary="${run_dir}/build_${run_id}_top${topk}_summary.json"
    model_path="${run_dir}/lgbm_${run_id}_ll_top${topk}.txt"
    train_metrics="${run_dir}/lgbm_${run_id}_ll_top${topk}_metrics.json"
    importance_path="${run_dir}/lgbm_${run_id}_ll_top${topk}_feature_importance.csv"
    prediction_path="${run_dir}/lgbm_${run_id}_ll_top${topk}_dev_predictions.json"
    score_path="${run_dir}/lgbm_${run_id}_ll_top${topk}_dev_scores.parquet"
    combined_metrics="${run_dir}/metrics.json"

    mkdir -p "${run_dir}"
    ensure_config "${run_id}" "${topk}" "${config_path}"

    limit_args=()
    if [ -n "${TRAIN_LIMIT}" ]; then limit_args+=(--train_limit "${TRAIN_LIMIT}"); fi
    if [ -n "${DEV_LIMIT}" ]; then limit_args+=(--dev_limit "${DEV_LIMIT}"); fi

    if [ "${FORCE}" = "1" ] || [ ! -f "${train_data}" ] || [ ! -f "${dev_data}" ]; then
      .venv/bin/python "${EXP_DIR}/build_dataset.py" \
        --candidate_config "${config_path}" \
        --train_output "${train_data}" \
        --dev_output "${dev_data}" \
        --summary_output "${build_summary}" \
        --candidate_topk "${topk}" \
        --retrieval_batch_size "${BATCH_SIZE}" \
        --require_positive \
        --train_sample_max_rows_per_group "${MAX_ROWS_PER_GROUP}" \
        --train_sample_head_negatives "${HEAD_NEGATIVES}" \
        "${limit_args[@]}"
    fi

    if [ "${FORCE}" = "1" ] || [ ! -f "${model_path}" ]; then
      .venv/bin/python "${EXP_DIR}/train_lgbm.py" \
        --train_data "${train_data}" \
        --valid_data "${dev_data}" \
        --model_output "${model_path}" \
        --metrics_output "${train_metrics}" \
        --feature_importance_output "${importance_path}" \
        --objective "${OBJECTIVE}" \
        --num_boost_round 2000 \
        --early_stopping_rounds 100 \
        --log_period 50
    fi

    if [ "${FORCE}" = "1" ] || [ ! -f "${prediction_path}" ]; then
      .venv/bin/python "${EXP_DIR}/apply_lgbm.py" \
        --model "${model_path}" \
        --candidate_data "${dev_data}" \
        --prediction_output "${prediction_path}" \
        --score_output "${score_path}" \
        --topk 20
    fi

    combine_metrics "${run_id}" "${OBJECTIVE}" "${topk}" "${build_summary}" "${train_metrics}" "${prediction_path}" "${combined_metrics}"
    write_summary_csv
  done
done

write_summary_csv
