#!/usr/bin/env bash
# exp116 ranking から Blind A submission を作る（response 全行新規生成 + zip + validate）。
# 使い方: build_submission.sh <ranking_tracks.json> <tag>
#   <tag> は出力名の識別子（例 lxc / lxct）。ファイル名は 63 字以下に収まる短さにする。
# response は最終 ranking の top1 に合わせ 80 行すべてを Claude backend で新規生成（leak-safe、
# response 全行再生成ルール）。過去 response の流用・splice は一切しない。
set -euo pipefail
cd "$(dirname "$0")/../../.."  # repo root
source .venv/bin/activate 2>/dev/null || true

RANK="$1"
TAG="$2"
OUT="exp/inference/blindset_A/exp116_${TAG}_A.json"
ZIP="exp/inference/blindset_A/submission_exp116_${TAG}_A.zip"
MODEL="${3:-claude-opus-4-8}"
VAL="scripts/validate_submission.py"

# 1) 80 行すべてを、この ranking の top1 に合わせて新規生成。
python mcrs/experiments/exp063_response_quality/generate_responses_v2.py \
  --backend claude --model "$MODEL" \
  --tracks_json "$RANK" --output "$OUT" --batch_size 6

# 2) submission ZIP（root=prediction.json）。eval より先に必ず作る。
python - "$OUT" "$ZIP" <<'PY'
import json, sys, zipfile
data = json.load(open(sys.argv[1], encoding="utf-8"))
tmp = "/tmp/prediction.json"
json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED) as z:
    z.write(tmp, "prediction.json")
print("wrote", sys.argv[2], "| zip name len", len(__import__("os").path.basename(sys.argv[2])))
PY

# 3) format 検証（80 件 / top-20 / ranked / root=prediction.json / 名前<=63字）。
python "$VAL" "$OUT"
python "$VAL" "$ZIP"
