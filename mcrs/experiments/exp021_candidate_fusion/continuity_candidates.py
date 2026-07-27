"""会話内の artist / album continuity を使う候補生成 source。

EDA で確認済みの「同一 session 内では exact track repeat はほぼ無いが、
artist / album は継続しやすい」という性質を、候補生成 source として検証する。
現在 turn より前に提示済みの music track だけを使い、Dev/Blind の正解や未来 turn は
参照しない。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mcrs.experiments.exp021_candidate_fusion.query_label_memory import (
    Task,
    evaluate_predictions,
    load_metadata,
    load_tasks,
    metadata_values,
    stable_float,
    write_tracks_json,
)


@dataclass(frozen=True)
class ContinuitySpec:
    """continuity source の重みと候補展開設定。"""

    name: str
    recent_tracks: int | None
    artist_weight: float
    album_weight: float
    tag_weight: float
    track_weight: float
    artist_top_tracks: int
    album_top_tracks: int
    tag_top_tracks: int
    exclude_prior_tracks: bool = True


def parse_recent_tracks(value: str) -> int | None:
    """CLI の recent_tracks 指定を int / None に変換する。"""
    if value in {"all", "none", ""}:
        return None
    return int(value)


def parse_variants(value: str) -> list[ContinuitySpec]:
    """variant 名を continuity source 設定へ変換する。

    Args:
        value: カンマ区切りの variant 名。

    Returns:
        ContinuitySpec の list。

    Raises:
        ValueError: 未知の variant が指定された場合。
    """
    registry = {
        "album": ContinuitySpec(
            name="album",
            recent_tracks=None,
            artist_weight=0.0,
            album_weight=1.0,
            tag_weight=0.0,
            track_weight=0.0,
            artist_top_tracks=0,
            album_top_tracks=120,
            tag_top_tracks=0,
        ),
        "artist": ContinuitySpec(
            name="artist",
            recent_tracks=None,
            artist_weight=1.0,
            album_weight=0.0,
            tag_weight=0.0,
            track_weight=0.0,
            artist_top_tracks=160,
            album_top_tracks=0,
            tag_top_tracks=0,
        ),
        "album_artist": ContinuitySpec(
            name="album_artist",
            recent_tracks=None,
            artist_weight=0.65,
            album_weight=1.0,
            tag_weight=0.0,
            track_weight=0.0,
            artist_top_tracks=140,
            album_top_tracks=120,
            tag_top_tracks=0,
        ),
        "artist_album": ContinuitySpec(
            name="artist_album",
            recent_tracks=None,
            artist_weight=1.0,
            album_weight=0.6,
            tag_weight=0.0,
            track_weight=0.0,
            artist_top_tracks=180,
            album_top_tracks=100,
            tag_top_tracks=0,
        ),
        "recent2_album_artist": ContinuitySpec(
            name="recent2_album_artist",
            recent_tracks=2,
            artist_weight=0.7,
            album_weight=1.0,
            tag_weight=0.0,
            track_weight=0.0,
            artist_top_tracks=140,
            album_top_tracks=120,
            tag_top_tracks=0,
        ),
        "recent1_album_artist": ContinuitySpec(
            name="recent1_album_artist",
            recent_tracks=1,
            artist_weight=0.7,
            album_weight=1.0,
            tag_weight=0.0,
            track_weight=0.0,
            artist_top_tracks=140,
            album_top_tracks=120,
            tag_top_tracks=0,
        ),
        "album_artist_tag": ContinuitySpec(
            name="album_artist_tag",
            recent_tracks=None,
            artist_weight=0.65,
            album_weight=1.0,
            tag_weight=0.08,
            track_weight=0.0,
            artist_top_tracks=140,
            album_top_tracks=120,
            tag_top_tracks=24,
        ),
    }
    specs: list[ContinuitySpec] = []
    for name in [item.strip() for item in value.split(",") if item.strip()]:
        if name not in registry:
            raise ValueError(f"Unknown variant: {name}. Valid: {sorted(registry)}")
        specs.append(registry[name])
    return specs


def build_value_index(
    metadata: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, list[str]]:
    """metadata field の値から popularity 順 track list への index を作る。

    Args:
        metadata: track_id -> metadata。
        field: `artist_name` / `album_name` / `tag_list` などの metadata field。

    Returns:
        正規化済み metadata 値 -> track_id list。
    """
    index: dict[str, list[str]] = defaultdict(list)
    for track_id, item in metadata.items():
        for value in metadata_values(item, field):
            index[value].append(track_id)

    def sort_key(track_id: str) -> tuple[float, str]:
        return (-stable_float(metadata[track_id].get("popularity")), track_id)

    return {key: sorted(set(track_ids), key=sort_key) for key, track_ids in index.items()}


def build_indexes(
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], list[str]]:
    """artist / album / tag index と fallback track list を作る。"""
    artist_index = build_value_index(metadata, "artist_name")
    album_index = build_value_index(metadata, "album_name")
    tag_index = build_value_index(metadata, "tag_list")
    global_tracks = sorted(
        metadata,
        key=lambda track_id: (-stable_float(metadata[track_id].get("popularity")), track_id),
    )
    return artist_index, album_index, tag_index, global_tracks


def prior_music_track_ids(task: Task, *, recent_tracks: int | None) -> list[str]:
    """task history から現在 turn より前の music track_id を recency 順で返す。

    Args:
        task: session-turn task。
        recent_tracks: 直近何曲だけ使うか。None の場合は全履歴を使う。

    Returns:
        新しい順の track_id list。
    """
    track_ids = [
        str(row.get("content", ""))
        for row in task.history
        if row.get("role") == "music" and row.get("content")
    ]
    if recent_tracks is not None:
        track_ids = track_ids[-recent_tracks:]
    return list(reversed(track_ids))


def add_index_scores(
    scores: Counter[str],
    keys: list[str],
    index: dict[str, list[str]],
    *,
    source_weight: float,
    recency_weight: float,
    top_tracks: int,
) -> None:
    """1 metadata field の continuity 展開スコアを加算する。"""
    if source_weight <= 0.0 or top_tracks <= 0:
        return
    for key in keys:
        for rank, track_id in enumerate(index.get(key, [])[:top_tracks], start=1):
            scores[track_id] += source_weight * recency_weight / math.log2(rank + 2.0)


def rank_task(
    task: Task,
    metadata: dict[str, dict[str, Any]],
    artist_index: dict[str, list[str]],
    album_index: dict[str, list[str]],
    tag_index: dict[str, list[str]],
    global_tracks: list[str],
    spec: ContinuitySpec,
    *,
    topk: int,
) -> list[str]:
    """1 task に対して continuity candidates を rank 順に返す。

    Args:
        task: session-turn task。
        metadata: track metadata。
        artist_index: artist -> tracks。
        album_index: album -> tracks。
        tag_index: tag -> tracks。
        global_tracks: fallback 用 popularity 順 track list。
        spec: continuity source 設定。
        topk: 出力件数。

    Returns:
        ranked track_id list。
    """
    prior_ids = prior_music_track_ids(task, recent_tracks=spec.recent_tracks)
    prior_set = set(prior_ids)
    scores: Counter[str] = Counter()

    for recency_rank, prior_track_id in enumerate(prior_ids, start=1):
        item = metadata.get(prior_track_id)
        if item is None:
            continue
        recency_weight = 1.0 / math.log2(recency_rank + 1.0)
        if spec.track_weight > 0.0:
            scores[prior_track_id] += spec.track_weight * recency_weight
        add_index_scores(
            scores,
            metadata_values(item, "album_name"),
            album_index,
            source_weight=spec.album_weight,
            recency_weight=recency_weight,
            top_tracks=spec.album_top_tracks,
        )
        add_index_scores(
            scores,
            metadata_values(item, "artist_name"),
            artist_index,
            source_weight=spec.artist_weight,
            recency_weight=recency_weight,
            top_tracks=spec.artist_top_tracks,
        )
        add_index_scores(
            scores,
            metadata_values(item, "tag_list"),
            tag_index,
            source_weight=spec.tag_weight,
            recency_weight=recency_weight,
            top_tracks=spec.tag_top_tracks,
        )

    if spec.exclude_prior_tracks:
        for track_id in prior_set:
            scores.pop(track_id, None)

    ranked = sorted(
        scores,
        key=lambda track_id: (
            -scores[track_id],
            -stable_float(metadata.get(track_id, {}).get("popularity")),
            track_id,
        ),
    )
    seen = set(ranked)
    for track_id in global_tracks:
        if len(ranked) >= topk:
            break
        if track_id in seen or (spec.exclude_prior_tracks and track_id in prior_set):
            continue
        ranked.append(track_id)
        seen.add(track_id)
    return ranked[:topk]


def summarize_by_turn(
    tasks: list[Task],
    predictions: dict[tuple[str, int], list[str]],
) -> list[dict[str, Any]]:
    """turn_number ごとの hit@20/100/500 と nDCG@20 を集計する。"""
    grouped: dict[int, list[Task]] = defaultdict(list)
    for task in tasks:
        grouped[task.turn_number].append(task)
    rows: list[dict[str, Any]] = []
    for turn_number, group_tasks in sorted(grouped.items()):
        metrics = evaluate_predictions(group_tasks, predictions)
        row = {"turn_number": turn_number, **metrics}
        rows.append(row)
    return rows


def run_variant(
    tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    artist_index: dict[str, list[str]],
    album_index: dict[str, list[str]],
    tag_index: dict[str, list[str]],
    global_tracks: list[str],
    spec: ContinuitySpec,
    *,
    topk: int,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """1 split / 1 variant の prediction と metrics を保存する。"""
    predictions: dict[tuple[str, int], list[str]] = {}
    prior_active = 0
    for task in tqdm(tasks, desc=f"rank {output_prefix}:{spec.name}"):
        if prior_music_track_ids(task, recent_tracks=spec.recent_tracks):
            prior_active += 1
        predictions[(task.session_id, task.turn_number)] = rank_task(
            task,
            metadata,
            artist_index,
            album_index,
            tag_index,
            global_tracks,
            spec,
            topk=topk,
        )
    metrics = evaluate_predictions(tasks, predictions)
    tracks_path = output_dir / f"{output_prefix}_tracks_exp021_cont_{spec.name}.json"
    write_tracks_json(tracks_path, tasks, predictions, topk=topk)
    return {
        "metrics": metrics,
        "prior_active": prior_active,
        "prior_active_rate": prior_active / len(tasks) if tasks else 0.0,
        "by_turn": summarize_by_turn(tasks, predictions),
        "tracks_path": str(tracks_path),
    }


def run_split(
    tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    specs: list[ContinuitySpec],
    *,
    topk: int,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """複数 variant を 1 split に適用する。"""
    artist_index, album_index, tag_index, global_tracks = build_indexes(metadata)
    summary: dict[str, Any] = {}
    for spec in specs:
        summary[spec.name] = run_variant(
            tasks,
            metadata,
            artist_index,
            album_index,
            tag_index,
            global_tracks,
            spec,
            topk=topk,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    return summary


def main() -> None:
    """CLI entrypoint。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--track_metadata_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--track_split_types", default="all_tracks")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--blind_dataset_name")
    parser.add_argument("--blind_split", default="test")
    parser.add_argument("--skip_dev", action="store_true")
    parser.add_argument(
        "--variants",
        default="album,artist,album_artist,artist_album,recent2_album_artist,recent1_album_artist,album_artist_tag",
    )
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--limit_dev_sessions", type=int)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates"),
    )
    args = parser.parse_args()

    split_types = [item.strip() for item in args.track_split_types.split(",") if item.strip()]
    metadata = load_metadata(args.track_metadata_dataset_name, split_types)
    specs = parse_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "topk": args.topk,
        "variants": [spec.__dict__ for spec in specs],
        "dev": {},
    }
    if not args.skip_dev:
        dev_tasks = load_tasks(
            args.conversation_dataset_name,
            args.dev_split,
            limit_sessions=args.limit_dev_sessions,
        )
        report["dev_task_count"] = len(dev_tasks)
        report["dev"] = run_split(
            dev_tasks,
            metadata,
            specs,
            topk=args.topk,
            output_dir=args.output_dir,
            output_prefix="dev",
        )

    if args.blind_dataset_name:
        blind_tasks = load_tasks(args.blind_dataset_name, args.blind_split)
        report["blind_task_count"] = len(blind_tasks)
        report["blind"] = run_split(
            blind_tasks,
            metadata,
            specs,
            topk=args.topk,
            output_dir=args.output_dir,
            output_prefix="blindA",
        )

    report_path = args.output_dir / "continuity_candidates_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "dev": report["dev"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
