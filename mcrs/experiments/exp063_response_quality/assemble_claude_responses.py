"""Claude agent が生成した response part を結合し、exp058 ranking 固定の submission JSON を作る。

ranking(predicted_track_ids)は tracks_json(exp058)からそのままコピー。predicted_response だけ
Claude 生成で差し替える。空 response は fallback で埋め、scaffold は postclean で除去（全 backend 共通）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from mcrs.db_item import MusicCatalogDB

import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from generate_responses_v2 import postclean_response, fallback_response


def main(args):
    """tracks_json の各 task に Claude 応答を結びつけ、submission JSON を書く。"""
    tracks = json.load(open(args.tracks_json, encoding="utf-8"))
    item_db = MusicCatalogDB(
        args.item_db_name, ["all_tracks"],
        ["track_name", "artist_name", "album_name", "release_date"],
    )

    # part ファイル（task_id -> response）を全て読み込んでマージ。
    resp_by_task = {}
    for path in sorted(glob.glob(args.parts_glob)):
        part = json.load(open(path, encoding="utf-8"))
        for k, v in part.items():
            resp_by_task[k] = v
        print(f"[assemble] {path}: {len(part)} responses")

    out, n_missing, n_backfill = [], 0, 0
    for r in tracks:
        key = f"{r['session_id']}::{r['turn_number']}"
        tids = r["predicted_track_ids"]
        resp = postclean_response(resp_by_task.get(key, ""))
        if not resp or not resp.strip():
            resp = fallback_response(item_db.metadata_dict, str(tids[0]))
            if key not in resp_by_task:
                n_missing += 1
            n_backfill += 1
        out.append({
            "session_id": r["session_id"],
            "user_id": r["user_id"],
            "turn_number": r["turn_number"],
            "predicted_track_ids": tids[:20],  # ranking は exp058 のまま不変
            "predicted_response": resp,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[assemble] wrote {len(out)} records ({n_missing} missing task_ids, {n_backfill} backfilled) -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_json", default="exp/inference/blindset_A/exp058_wide100_ens6_A.json")
    p.add_argument("--parts_glob", default="mcrs/experiments/exp063_response_quality/results/claude_resp_part*.json")
    p.add_argument("--output", default="exp/inference/blindset_A/exp063_v2_claude_A.json")
    p.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    main(p.parse_args())
