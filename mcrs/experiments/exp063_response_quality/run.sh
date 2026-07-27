#!/usr/bin/env bash
# exp062 response-quality: exp058 の固定 ranking に対し response だけ再生成する。
# ranking(predicted_track_ids)は触らないので nDCG/catalog_div は exp058 のまま。狙いは judge(0.30)+lexical(0.10)。
set -euo pipefail
cd "$(dirname "$0")/../../.."  # repo root

GEN=mcrs/experiments/exp063_response_quality/generate_responses_v2.py
EVAL=mcrs/experiments/exp063_response_quality/eval_responses.py
BASE=exp/inference/blindset_A/exp058_wide100_ens6_A.json   # ranking 固定の入力（current base/gate）
OUTDIR=exp/inference/blindset_A

# --- V1: local Qwen3.5-4B + 改良 prompt -------------------------------------
.venv/bin/python "$GEN" \
  --tracks_json "$BASE" \
  --output "$OUTDIR/exp063_v2_qwen4b_A.json" \
  --backend qwen --lm_type Qwen/Qwen3.5-4B \
  --batch_size 8 --max_new_tokens 320

# --- V-Gemini: gemini-3.1-pro-preview（最新 Pro, judge と同一 family）---------
# 認証: repo root .env に GEMINI_API_KEY=... を置く（.env は gitignore 済）。
# SDK: uv pip install google-genai（導入済）。
.venv/bin/python "$GEN" \
  --tracks_json "$BASE" \
  --output "$OUTDIR/exp063_v2_gemini_pro_A.json" \
  --backend gemini --lm_type gemini-3.1-pro-preview \
  --max_new_tokens 400 --temperature 0.7  # ~100-115語のrich profile用

# --- local 指標（lexical/apology/opening/ranking 保全）------------------------
.venv/bin/python "$EVAL" \
  "$BASE" \
  "$OUTDIR/exp063_v2_qwen4b_A.json" \
  "$OUTDIR/exp063_v2_gemini_pro_A.json" \
  --baseline "$BASE"

# --- submission ZIP（root=prediction.json, 名前<=63字）------------------------
for v in qwen4b gemini_pro; do
  .venv/bin/python - "$OUTDIR/exp063_v2_${v}_A.json" "$OUTDIR/submission_exp063_v2_${v}_A.zip" <<'PY'
import json, sys, zipfile, os
src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
tmp = "/tmp/prediction.json"; json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z: z.write(tmp, "prediction.json")
os.remove(tmp); print("wrote", dst)
PY
  .venv/bin/python scripts/validate_submission.py "$OUTDIR/submission_exp063_v2_${v}_A.zip"
done
