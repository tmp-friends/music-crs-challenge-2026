"""within-session（同一セッション内で既に提示済み）の music track を最終 top20 から除外する後処理。

exp073 調査: within-session の既出 music は次 turn の正解と絶対一致しない（Dev 0/7000）＝除外しても
正解を消す危険ゼロの free rule。ranker は top20 しか出さない（Dev primary が 20 件のため topk≤20）ので、
既出を落として空いた底枠は backfill primary（exp015B blind, 500 件/行）の rank 順から、未収載かつ
非既出の候補で埋める。除外で繰り上がる上位は ndcg を非減少にし、底枠 backfill は position~20（低 weight）。

リーク禁止: 現タスクの最終 turn より前（turn < final）の music role のみを既出として除外。future/GT 不使用。
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset


def prior_music_ids(convs, last_turn):
    """最終 turn 未満の music track_id 集合（within-session 既出）を返す。"""
    return {
        str(m.get("content", ""))
        for m in convs
        if m.get("role") == "music" and int(m.get("turn_number", 0)) < last_turn
    }


def main(args):
    """top20 ranking から within-session 既出を除外し、底枠を backfill primary で埋めて top20 を書く。"""
    ranking = json.load(open(args.tracks_json, encoding="utf-8"))
    backfill = {
        (str(r["session_id"]), int(r["turn_number"])): [str(t) for t in r["predicted_track_ids"]]
        for r in json.load(open(args.backfill_primary, encoding="utf-8"))
    }
    db = load_dataset(args.dataset_name, split=args.split)
    prior_by_key = {}
    for it in db:
        convs = list(it["conversations"])
        last_turn = int(convs[-1]["turn_number"])
        prior_by_key[(str(it["session_id"]), last_turn)] = prior_music_ids(convs, last_turn)

    out = []
    n_excluded_rows = 0
    n_excluded_tracks = 0
    n_backfilled = 0
    for r in ranking:
        key = (str(r["session_id"]), int(r["turn_number"]))
        prior = prior_by_key.get(key, set())
        ids = [str(t) for t in r["predicted_track_ids"]]
        kept = [t for t in ids if t not in prior]
        dropped = len(ids) - len(kept)
        if dropped:
            n_excluded_rows += 1
            n_excluded_tracks += dropped
        # 空いた底枠を backfill primary（exp015B, rank 順）の未収載・非既出候補で埋める。
        if len(kept) < 20:
            kept_set = set(kept)
            for t in backfill.get(key, []):
                if len(kept) >= 20:
                    break
                if t not in kept_set and t not in prior:
                    kept.append(t); kept_set.add(t); n_backfilled += 1
        top20 = kept[:20]
        assert len(top20) == 20, f"{key}: only {len(top20)} after exclusion/backfill"
        out.append({
            "session_id": r["session_id"],
            "user_id": r.get("user_id", ""),
            "turn_number": r["turn_number"],
            "predicted_track_ids": top20,
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[within-excl] {len(out)} rows | rows with exclusion {n_excluded_rows} | tracks excluded {n_excluded_tracks} | backfilled slots {n_backfilled}")
    print(f"[within-excl] -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tracks_json", default="mcrs/experiments/exp090_cross_session_source/results/ranker/blindA_tracks_cross_clean_ens6.json")
    p.add_argument("--backfill_primary", default="exp/inference/blindset_A/exp015_B_ll_t500_A.json",
                   help="底枠 backfill 用の rank 順候補（exp015B blind, 500件/行）")
    p.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--split", default="test")
    p.add_argument("--output", default="mcrs/experiments/exp090_cross_session_source/results/ranker/blindA_tracks_cross_clean_excl_top20.json")
    main(p.parse_args())
