"""全 80 task の leak-safe な response 生成コンテキストを 1 file にまとめる（Claude agent 生成用）。

Claude サブエージェント（Agent tool）に response を書かせるための入力。各 task は exp058 の固定 top1
推薦に対して、current turn までの対話 + 入力側 user_profile + 推薦 track(clean tag) + 書き出しスタイル
を持つ。ranking は触らない。リーク禁止: conversation_goal / goal_progress_assessments は不使用。
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

from mcrs.db_item import MusicCatalogDB

import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from generate_responses_v2 import assign_style, clean_tags, _first_str  # 生成器と同一ロジックを共有


def compact_history(convs, item_db, max_turns=8):
    """current turn 直前までの履歴を agent 用に短い行列へ畳む（music role は曲名へ展開）。"""
    lines = []
    for m in convs:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "music":
            md = item_db.metadata_dict.get(content, {})
            name = _first_str(md.get("track_name")) or content
            artist = _first_str(md.get("artist_name")) or "?"
            content = f'[system recommended: "{name}" by {artist}]'
            role = "assistant"
        lines.append({"role": role if role in {"user", "assistant"} else "assistant", "text": content})
    return lines[-max_turns:]


def main(args):
    """exp058 tracks の top1 と Blind A 文脈から、80 task の生成入力 JSON を書く。"""
    tracks = json.load(open(args.tracks_json, encoding="utf-8"))
    tracks_by_key = {(str(r["session_id"]), int(r["turn_number"])): r for r in tracks}
    item_db = MusicCatalogDB(
        args.item_db_name, ["all_tracks"],
        ["track_name", "artist_name", "album_name", "release_date"],
    )
    db = load_dataset(args.test_dataset_name, split=args.split)

    out = []
    for it in db:
        convs = list(it["conversations"])
        last = convs[-1]
        key = (str(it["session_id"]), int(last["turn_number"]))
        rec = tracks_by_key.get(key)
        if not rec or not rec.get("predicted_track_ids"):
            continue
        tid = str(rec["predicted_track_ids"][0])
        md = item_db.metadata_dict.get(tid, {})
        release = str(md.get("release_date") or "")
        year = release[:4] if release[:4].isdigit() else ""
        up = it.get("user_profile") or {}
        out.append({
            "task_id": f"{key[0]}::{key[1]}",
            "user_query": str(last["content"]),
            "history": compact_history(convs[:-1], item_db),
            "user_profile": {
                k: up.get(k) for k in
                ["age", "gender", "country_name", "preferred_language", "preferred_musical_culture"]
            },
            "recommended_track": {
                "title": _first_str(md.get("track_name")),
                "artist": _first_str(md.get("artist_name")),
                "album": _first_str(md.get("album_name")),
                "year": year,
                "tags": clean_tags(md.get("tag_list"), max_tags=8),
            },
            "style_directive": assign_style(key[0], key[1]),
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[gen-context] wrote {len(out)} tasks -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_json", default="exp/inference/blindset_A/exp058_wide100_ens6_A.json")
    p.add_argument("--output", default="mcrs/experiments/exp063_response_quality/results/claude_gen_input.json")
    p.add_argument("--test_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--split", default="test")
    p.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    main(p.parse_args())
