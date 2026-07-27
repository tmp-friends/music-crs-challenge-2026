"""variant 群を Gemini-judge 代理（Claude subagent）に渡すためのコンパクト採点入力を作る。

公式 judge は Blind set の response を text-only（ground truth 非参照）で Personalization /
Explanation Quality を 1-5 採点する。これを再現するため、各 sample task について
「対話文脈 + 推薦 track + 各 variant の応答（ラベル匿名）」を 1 つの JSON にまとめる。
ground truth は一切含めない（judge も見ない設計）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from datasets import load_dataset

from mcrs.db_item import MusicCatalogDB

import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from generate_responses_v2 import clean_tags  # 生成器と同じ tag クリーニングを judge 提示にも適用


def short_history(convs, item_db, max_turns=6):
    """current turn 直前までの履歴を、judge が読める短い行列へ畳む（リーク無し）。"""
    lines = []
    for m in convs:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "music":
            try:
                md = item_db.metadata_dict.get(content, {})
                name = md.get("track_name")
                artist = md.get("artist_name")
                name = (name[0] if isinstance(name, list) and name else name) or content
                artist = (artist[0] if isinstance(artist, list) and artist else artist) or "?"
                content = f"[recommended track: \"{name}\" by {artist}]"
            except Exception:  # noqa: BLE001
                content = "[recommended track]"
            role = "assistant"
        lines.append(f"{role}: {content}")
    return lines[-max_turns:]


def rec_track_str(item_db, tid):
    """推薦 top1 を judge 提示用に短く整形（title/artist/tags）。"""
    md = item_db.metadata_dict.get(tid, {})
    def first(v):
        return (v[0] if isinstance(v, list) and v else v) or ""
    tags = clean_tags(md.get("tag_list"), max_tags=8)
    return {
        "title": first(md.get("track_name")),
        "artist": first(md.get("artist_name")),
        "tags": tags,
    }


def main(args):
    """sample task ごとに文脈 + 各 variant 応答をまとめた judge 入力 JSON を書く。"""
    item_db = MusicCatalogDB(
        args.item_db_name, ["all_tracks"],
        ["track_name", "artist_name", "album_name", "release_date"],
    )
    db = load_dataset(args.test_dataset_name, split=args.split)
    ctx = {}
    for it in db:
        convs = list(it["conversations"])
        last = convs[-1]
        key = (str(it["session_id"]), int(last["turn_number"]))
        ctx[key] = {
            "user_query": str(last["content"]),
            "history": short_history(convs[:-1], item_db),
            "profile": {
                k: (it.get("user_profile") or {}).get(k)
                for k in ["age", "gender", "country_name", "preferred_musical_culture"]
            },
        }

    variants = {}
    for spec in args.variants:
        label, path = spec.split("=", 1)
        variants[label] = {(r["session_id"], r["turn_number"]): r for r in json.load(open(path))}

    keys = sorted(ctx.keys())
    # 決定的サンプリング（md5 で安定選抜）。
    keys = sorted(keys, key=lambda k: hashlib.md5(f"{k}".encode()).hexdigest())[: args.sample]

    out = []
    for key in keys:
        rec_any = next((v.get(key) for v in variants.values() if v.get(key)), None)
        if not rec_any:
            continue
        top1 = str(rec_any["predicted_track_ids"][0])
        responses = {}
        for label, vmap in variants.items():
            r = vmap.get(key)
            if r:
                responses[label] = r["predicted_response"]
        out.append({
            "task_id": f"{key[0]}::{key[1]}",
            "user_query": ctx[key]["user_query"],
            "history": ctx[key]["history"],
            "user_profile": ctx[key]["profile"],
            "recommended_track": rec_track_str(item_db, top1),
            "responses": responses,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[judge-input] wrote {len(out)} tasks x {len(variants)} variants -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", required=True, help="label=path.json の列")
    p.add_argument("--output", default="mcrs/experiments/exp063_response_quality/results/judge_input.json")
    p.add_argument("--sample", type=int, default=24)
    p.add_argument("--test_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--split", default="test")
    p.add_argument("--item_db_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    main(p.parse_args())
