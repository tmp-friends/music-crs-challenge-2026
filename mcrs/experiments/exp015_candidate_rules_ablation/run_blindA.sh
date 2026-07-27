#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

EXP_DIR="mcrs/experiments/exp015_candidate_rules_ablation"
RESULT_ROOT="${RESULT_ROOT:-${EXP_DIR}/results}"
EVAL_DATASET="${EVAL_DATASET:-blindset_A}"
TOPKS="${TOPKS:-500}"
ONLY_RUNS="${ONLY_RUNS:-A}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FORCE="${FORCE:-0}"
OUT_BASE="exp/inference/${EVAL_DATASET}"

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
  local model_path="$3"
  local tid="exp015_${run_id}_ll_t${topk}_A"
  local output_config="${EXP_DIR}/configs/${tid}.yaml"
  .venv/bin/python - "$run_id" "$topk" "$model_path" "$output_config" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

from mcrs.experiments.exp015_candidate_rules_ablation.candidate_sources import source_profile_to_sources

run_id = sys.argv[1]
topk = int(sys.argv[2])
model_path = sys.argv[3]
output_config = Path(sys.argv[4])
base_config = Path("mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_base_s05_ll_t500_A.yaml")

config = OmegaConf.load(base_config)
config.experiment_module = "mcrs.experiments.exp015_candidate_rules_ablation.pipeline"
config.source_profile = run_id
config.union_mode = "quota" if run_id == "G" else "rrf"
config.enabled_sources = list(source_profile_to_sources(run_id))
config.retrieval_topk = topk
config.reranker_model_name = model_path
config.track_split_types = ["all_tracks"]
output_config.parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config=config, f=output_config)
print(output_config)
PY
}

validate_prediction_and_zip() {
  local json_path="$1"
  local zip_path="$2"
  .venv/bin/python - "$json_path" "$zip_path" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

json_path = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
rows = json.loads(json_path.read_text(encoding="utf-8"))
bad_len = sum(1 for row in rows if len(row.get("predicted_track_ids", [])) != 20)
bad_unique = sum(
    1
    for row in rows
    if len(row.get("predicted_track_ids", [])) != len(set(row.get("predicted_track_ids", [])))
)
with zipfile.ZipFile(zip_path) as zf:
    entries = zf.namelist()
if entries != ["prediction.json"]:
    raise SystemExit(f"bad zip entries: {entries}")
if bad_len or bad_unique:
    raise SystemExit(f"bad_len={bad_len} bad_unique={bad_unique}")
print(f"validated {zip_path} records={len(rows)}")
PY
}

mkdir -p "${OUT_BASE}"
IFS=',' read -r -a TOPK_LIST <<< "${TOPKS// /}"

for topk in "${TOPK_LIST[@]}"; do
  for index in "${!RUN_IDS[@]}"; do
    run_id="${RUN_IDS[$index]}"
    slug="${RUN_SLUGS[$index]}"
    if ! should_run "${run_id}"; then
      continue
    fi
    run_dir="${RESULT_ROOT}/top${topk}/${run_id}_${slug}"
    if [ "${run_id}" = "A" ]; then
      model_path="mcrs/experiments/exp014d_lgbm_objective_comparison/results/top500/ll/s05_exact_current/lgbm_s05_ll_top500_sampled.txt"
    else
      model_path="${run_dir}/lgbm_${run_id}_ll_top${topk}.txt"
    fi
    if [ ! -f "${model_path}" ]; then
      echo "Missing model: ${model_path}" >&2
      exit 1
    fi
    tid="exp015_${run_id}_ll_t${topk}_A"
    json_path="${OUT_BASE}/${tid}.json"
    zip_path="${OUT_BASE}/submission_${tid}.zip"
    ensure_config "${run_id}" "${topk}" "${model_path}"
    if [ "${FORCE}" = "1" ] || [ ! -f "${json_path}" ]; then
      .venv/bin/python run_inference_blindset.py \
        --tid "${tid}" \
        --eval_dataset "${EVAL_DATASET}" \
        --batch_size "${BATCH_SIZE}"
    fi
    if [ "${FORCE}" = "1" ] || [ ! -f "${zip_path}" ]; then
      tmp_dir="$(mktemp -d)"
      cp "${json_path}" "${tmp_dir}/prediction.json"
      (cd "${tmp_dir}" && zip -q -FS "${OLDPWD}/${zip_path}" prediction.json)
      rm -rf "${tmp_dir}"
    fi
    validate_prediction_and_zip "${json_path}" "${zip_path}"
  done
done
