"""Blind B (TalkPlayData-Challenge-Blind-B) の構造 EDA と Blind A / catalog 比較。

2026-06-23 の Blind B 公開を受けた初回 EDA。提出構造・null フィールド・Blind A との
重複/パラフレーズ関係・register シフト・cold start 比率・catalog overlap を計測する。

実行:
    python EDA/20260623_blind_b_eda.py

主な発見（詳細は EDA/summary/20260623_blind_b_eda.md）:
    - Blind B = 80 session / 1 session 1 予測（最終 user turn）。catalog は 100% overlap。
    - メタデータ大幅削除: conversation_goal 0/80, session_date 0/80, goal_progress 0/80,
      thought 全削除, user_profile / user_id は 40/80 のみ存在（残り cold）。
    - Blind A と session_id が 43/80 重複。うち多くは同一 session を口語調に paraphrase
      したもの（GT は両者とも hidden なので leak は無い）。
    - register が口語寄りにシフト（slang marker 約 3 倍）、平均クエリ長やや増。
"""

from __future__ import annotations

import collections
import statistics

from datasets import load_dataset

BLIND_B = "talkpl-ai/TalkPlayData-Challenge-Blind-B"
BLIND_A = "talkpl-ai/TalkPlayData-Challenge-Blind-A"
TRACK_META = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

SLANG_MARKERS = [
    "cheers", "y'know", "ngl", "omg", "mate", "gonna", "wanna",
    "kinda", "sorta", "lol", "tbh", "innit", "dunno", "ya ",
]


def _is_warm(record) -> bool:
    """user_profile が埋まっている（cold でない）session か判定する。"""
    profile = record["user_profile"]
    return bool(profile and profile.get("user_id"))


def scan_structure(name: str, dataset_name: str) -> None:
    """1 dataset の構造統計（null フィールド・turn 分布・role 分布）を出力する。"""
    dataset = load_dataset(dataset_name, split="test")
    n = len(dataset)
    profile_populated = goal_populated = uid_nonempty = 0
    date_nonempty = gpa_nonempty = thought_nonempty = 0
    turn_counts: list[int] = []
    role_counter: collections.Counter = collections.Counter()
    music_turns = sessions_with_music = 0

    for record in dataset:
        if _is_warm(record):
            profile_populated += 1
        if record["conversation_goal"] is not None:
            goal_populated += 1
        if (record["user_id"] or "").strip():
            uid_nonempty += 1
        if (record["session_date"] or "").strip():
            date_nonempty += 1
        if record["goal_progress_assessments"]:
            gpa_nonempty += 1
        conversations = record["conversations"]
        turn_counts.append(len(conversations))
        has_music = False
        for turn in conversations:
            role_counter[turn["role"]] += 1
            if (turn.get("thought") or "").strip():
                thought_nonempty += 1
            if turn["role"] == "music":
                music_turns += 1
                has_music = True
        sessions_with_music += int(has_music)

    print(f"=== {name} (n={n}) ===")
    print(f"  user_profile populated : {profile_populated}/{n}")
    print(f"  conversation_goal !null: {goal_populated}/{n}")
    print(f"  user_id non-empty      : {uid_nonempty}/{n}")
    print(f"  session_date non-empty : {date_nonempty}/{n}")
    print(f"  goal_progress non-empty: {gpa_nonempty}/{n}")
    print(f"  thought non-empty turns: {thought_nonempty}")
    print(
        f"  turns/session: min={min(turn_counts)} max={max(turn_counts)} "
        f"mean={statistics.mean(turn_counts):.2f} median={statistics.median(turn_counts)}"
    )
    print(f"  turn count dist: {dict(sorted(collections.Counter(turn_counts).items()))}")
    print(f"  roles (all turns)      : {dict(role_counter)}")
    print(f"  music turns total      : {music_turns} (sessions with music: {sessions_with_music})")
    print(f"  last-turn roles        : {dict(collections.Counter(r['conversations'][-1]['role'] for r in dataset))}")
    print()


def scan_demographics(name: str, dataset_name: str) -> None:
    """populated profile のみで人口統計の分布を出力する。"""
    dataset = load_dataset(dataset_name, split="test")
    fields = ["country_name", "age_group", "gender", "preferred_language", "preferred_musical_culture"]
    counters = {field: collections.Counter() for field in fields}
    populated = 0
    for record in dataset:
        if not _is_warm(record):
            continue
        populated += 1
        profile = record["user_profile"]
        for field in fields:
            counters[field][profile.get(field)] += 1
    print(f"=== {name} demographics (populated={populated}) ===")
    for field in fields:
        print(f"  {field}: {counters[field].most_common(6)}")
    print()


def scan_register(name: str, dataset_name: str) -> None:
    """user turn の長さと口語 marker 出現数を計測する。"""
    dataset = load_dataset(dataset_name, split="test")
    lengths: list[int] = []
    slang_hits = 0
    n_user = 0
    for record in dataset:
        for turn in record["conversations"]:
            if turn["role"] != "user":
                continue
            content = turn["content"] or ""
            lengths.append(len(content.split()))
            n_user += 1
            lower = content.lower()
            slang_hits += sum(lower.count(marker) for marker in SLANG_MARKERS)
    print(
        f"{name}: user turns={n_user} | mean words={statistics.mean(lengths):.1f} "
        f"median={statistics.median(lengths)} | slang hits={slang_hits} ({slang_hits / n_user:.2f}/turn)"
    )


def scan_overlap() -> None:
    """Blind A と Blind B の session_id / user_id 重複と paraphrase 関係を計測する。"""
    blind_b = load_dataset(BLIND_B, split="test")
    blind_a = load_dataset(BLIND_A, split="test")
    a_by_sid = {r["session_id"]: r for r in blind_a}
    a_uids = {r["user_id"] for r in blind_a if (r["user_id"] or "").strip()}
    a_sids = set(a_by_sid)

    shared = same_len = target_later = prefix_match = cold_in_shared = 0
    for record in blind_b:
        sid = record["session_id"]
        if sid not in a_by_sid:
            continue
        shared += 1
        ra = a_by_sid[sid]
        if len(record["conversations"]) == len(ra["conversations"]):
            same_len += 1
        ua = [t["content"] for t in ra["conversations"] if t["role"] == "user"]
        ub = [t["content"] for t in record["conversations"] if t["role"] == "user"]
        k = min(len(ua), len(ub))
        if ua[:k] == ub[:k]:
            prefix_match += 1
        if record["conversations"][-1]["turn_number"] > ra["conversations"][-1]["turn_number"]:
            target_later += 1
        if not _is_warm(record):
            cold_in_shared += 1

    print("=== Blind A vs Blind B overlap ===")
    print(f"  shared session_ids: {shared}/80 | B-only: {80 - shared}")
    print(f"  of shared: same length={same_len}, prediction-target later in B={target_later}")
    print(f"  user-turn prefix verbatim match: {prefix_match}/{shared} (残りは paraphrase)")
    print(f"  cold(no profile) among shared: {cold_in_shared}/{shared}")

    # B-only 37 session の user 新規性
    empty_uid = overlap_uid = new_uid = 0
    for record in blind_b:
        if record["session_id"] in a_sids:
            continue
        uid = (record["user_id"] or "").strip()
        if not uid:
            empty_uid += 1
        elif uid in a_uids:
            overlap_uid += 1
        else:
            new_uid += 1
    print(f"  B-only sessions: empty_uid={empty_uid}, uid-overlaps-A={overlap_uid}, genuinely-new-uid={new_uid}")
    print()


def scan_catalog_overlap() -> None:
    """Blind B の文脈 music track_id が catalog に存在するか確認する。"""
    blind_b = load_dataset(BLIND_B, split="test")
    meta = load_dataset(TRACK_META, split="all_tracks")
    catalog = set(meta["track_id"])
    tracks = {
        turn["content"]
        for record in blind_b
        for turn in record["conversations"]
        if turn["role"] == "music" and (turn["content"] or "").strip()
    }
    in_catalog = sum(1 for t in tracks if t in catalog)
    print(f"=== catalog overlap === catalog size={len(catalog)}")
    print(f"  Blind B context tracks: {len(tracks)} unique | in catalog: {in_catalog} "
          f"({100 * in_catalog / max(1, len(tracks)):.1f}%)")
    print()


def main() -> None:
    scan_structure("Blind B", BLIND_B)
    scan_structure("Blind A", BLIND_A)
    scan_demographics("Blind B", BLIND_B)
    scan_demographics("Blind A", BLIND_A)
    scan_overlap()
    scan_catalog_overlap()
    print("=== register shift ===")
    scan_register("Blind B", BLIND_B)
    scan_register("Blind A", BLIND_A)


if __name__ == "__main__":
    main()
