"""within-session 既出 music の per-key 除外 JSON を Dev + Blind A 分まとめて生成する。

ranker / seed_ensemble の `--exclude_json` に渡し、候補 union 構築段で「同一セッション内で既に提示済みの
music track」を除外する（exp073: within-session 既出は次 turn の正解と絶対一致しない＝Dev 0/7000＝除外は
guaranteed-wrong 除去で ndcg 非減少）。後処理 backfill ではなく候補段で抜く正攻法（user 指摘）。

リーク禁止: 現タスクの最終 turn より前（turn < target_turn）の music role のみ既出として列挙。
Dev は全 turn の各 music target、Blind は最終 turn 1 タスク（隠し最終 turn は除外対象に出さない）。
"""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset


def music_turns(convs):
    """(turn_number, track_id) の music 列を返す。"""
    return [
        (int(m.get("turn_number", 0)), str(m.get("content", "")))
        for m in convs
        if m.get("role") == "music"
    ]


def main(args):
    """Dev + Blind A の per-key within-session 既出 track を 1 つの exclude JSON に書く。"""
    out = []

    # Dev: 全 music turn が予測対象。各 target turn の prior = それより前の turn の music。
    dev = load_dataset(args.dev_dataset_name, split=args.split)
    for it in dev:
        convs = list(it["conversations"])
        mt = music_turns(convs)
        target_turns = sorted({t for t, _ in mt})
        for t in target_turns:
            prior = sorted({c for (u, c) in mt if u < t and c})
            out.append({"session_id": str(it["session_id"]), "turn_number": t, "exclude_track_ids": prior})

    # Blind A: 予測対象は会話の最終 turn。prior = それ未満の visible music。
    blind = load_dataset(args.blind_dataset_name, split=args.split)
    for it in blind:
        convs = list(it["conversations"])
        last_turn = int(convs[-1]["turn_number"])
        prior = sorted({c for (u, c) in music_turns(convs) if u < last_turn and c})
        out.append({"session_id": str(it["session_id"]), "turn_number": last_turn, "exclude_track_ids": prior})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False)
    nz = sum(1 for r in out if r["exclude_track_ids"])
    print(f"[exclude-json] {len(out)} keys ({nz} with within-session exclusions) -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dev_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    p.add_argument("--blind_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    p.add_argument("--split", default="test")
    p.add_argument("--output", default="mcrs/experiments/exp090_cross_session_source/results/within_session_exclude.json")
    main(p.parse_args())
