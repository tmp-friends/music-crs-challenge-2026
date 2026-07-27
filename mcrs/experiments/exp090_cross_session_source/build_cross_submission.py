"""exp090 cross-session ens6 ranking に response を付与し Blind A submission を作る。

ranking は cross-ens6 の Blind A tracks。response は top1 が exp063-long（=exp058 ranking）と一致する
行は exp063-long をそのまま流用し、cross-session で top1 が変わった行は grounded な metadata template
へ差し替える（response が推薦 track と矛盾しないように。判定軸は ndcg なので response は再最適化しない）。
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile

from mcrs.db_item import MusicCatalogDB

import sys as _sys

_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp063_response_quality"))
from generate_responses_v2 import postclean_response, fallback_response, _first_str  # noqa: E402


def metadata_template(item_db, track_id):
    """top1 変更行用の grounded response（推薦 track と矛盾しない最小説明）。"""
    md = item_db.metadata_dict.get(str(track_id), {})
    title = _first_str(md.get("track_name")) or "this track"
    artist = _first_str(md.get("artist_name"))
    album = _first_str(md.get("album_name"))
    by = f" by {artist}" if artist else ""
    frm = f" from {album}" if album else ""
    return (
        f'I think you\'ll really enjoy "{title}"{by}{frm} — it lines up well with what you\'ve been '
        f"exploring across your sessions. Want a few more along these lines?"
    )


def main(args):
    """cross-ens6 tracks に response を付け submission JSON / ZIP を書く。"""
    tracks = json.load(open(args.tracks_json, encoding="utf-8"))
    e63 = {
        (str(r["session_id"]), int(r["turn_number"])): r
        for r in json.load(open(args.exp063_json, encoding="utf-8"))
    }
    item_db = MusicCatalogDB(
        args.item_db_name, ["all_tracks"],
        ["track_name", "artist_name", "album_name", "release_date"],
    )

    out, n_keep, n_template = [], 0, 0
    for r in tracks:
        key = (str(r["session_id"]), int(r["turn_number"]))
        tids = [str(t) for t in r["predicted_track_ids"][:20]]
        src = e63.get(key)
        # exp063-long の top1 と一致するなら response 流用、違えば template。
        if src and [str(t) for t in src["predicted_track_ids"][:1]] == tids[:1]:
            resp = postclean_response(src.get("predicted_response", ""))
            n_keep += 1
        else:
            resp = metadata_template(item_db, tids[0])
            n_template += 1
        if not resp or not resp.strip():
            resp = fallback_response(item_db.metadata_dict, tids[0])
        out.append({
            "session_id": str(r["session_id"]),
            "user_id": str(r.get("user_id", src.get("user_id", "") if src else "")),
            "turn_number": int(r["turn_number"]),
            "predicted_track_ids": tids,
            "predicted_response": resp,
        })

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    json.dump(out, open(args.output_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[build] {len(out)} records | response kept {n_keep}, template(top1 changed) {n_template}")
    print(f"[build] json -> {args.output_json}")
    print(f"[build] zip  -> {args.output_zip} (name len {len(os.path.basename(args.output_zip))})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_json", default="mcrs/experiments/exp090_cross_session_source/results/ranker/blindA_tracks_cross_ens6.json")
    p.add_argument("--exp063_json", default="exp/inference/blindset_A/exp063_v2_claude_long_A.json")
    p.add_argument("--output_json", default="exp/inference/blindset_A/exp090_cross_session_ens6_A.json")
    p.add_argument("--output_zip", default="exp/inference/blindset_A/submission_exp090_cross_ens6_A.zip")
    p.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    main(p.parse_args())
