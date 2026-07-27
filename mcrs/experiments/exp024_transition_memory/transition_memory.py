"""train split の履歴 music entity から次 track への遷移を候補化する。

session-local continuity は同一 session 内の artist / album 継続を直接使ったが、
この source は train split の session-turn を使って「過去に提示された music track /
artist / album / tag」と「その turn の正解 track」の遷移 map を作る。Dev/Blind では
現在 turn より前の music 履歴だけを key にして候補を生成する。
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
    build_track_lists,
    evaluate_predictions,
    load_metadata,
    load_tasks,
    metadata_values,
    stable_float,
    write_tracks_json,
)


@dataclass(frozen=True)
class TransitionSpec:
    """transition memory source の key/expansion 設定。"""

    name: str
    recent_tracks: int | None
    track_key_weight: float
    artist_key_weight: float
    album_key_weight: float
    tag_key_weight: float
    direct_label_weight: float
    artist_expand_weight: float
    tag_expand_weight: float
    artist_top_tracks: int
    tag_top_tracks: int
    max_postings_per_key: int


@dataclass
class TransitionIndex:
    """履歴 entity key から次 track candidate への index。"""

    postings: dict[str, Counter[str]]
    metadata: dict[str, dict[str, Any]]
    tracks_by_artist: dict[str, list[str]]
    tracks_by_tag: dict[str, list[str]]
    global_tracks: list[str]


def parse_variants(value: str) -> list[TransitionSpec]:
    """variant 名を TransitionSpec に変換する。"""
    registry = {
        "recent1": TransitionSpec(
            name="recent1",
            recent_tracks=1,
            track_key_weight=2.5,
            artist_key_weight=1.6,
            album_key_weight=1.3,
            tag_key_weight=0.25,
            direct_label_weight=2.0,
            artist_expand_weight=0.45,
            tag_expand_weight=0.08,
            artist_top_tracks=80,
            tag_top_tracks=16,
            max_postings_per_key=220,
        ),
        "recent2": TransitionSpec(
            name="recent2",
            recent_tracks=2,
            track_key_weight=2.2,
            artist_key_weight=1.4,
            album_key_weight=1.1,
            tag_key_weight=0.22,
            direct_label_weight=2.0,
            artist_expand_weight=0.45,
            tag_expand_weight=0.08,
            artist_top_tracks=80,
            tag_top_tracks=16,
            max_postings_per_key=220,
        ),
        "all": TransitionSpec(
            name="all",
            recent_tracks=None,
            track_key_weight=1.8,
            artist_key_weight=1.2,
            album_key_weight=0.9,
            tag_key_weight=0.18,
            direct_label_weight=2.0,
            artist_expand_weight=0.45,
            tag_expand_weight=0.08,
            artist_top_tracks=80,
            tag_top_tracks=16,
            max_postings_per_key=220,
        ),
        "artist_album": TransitionSpec(
            name="artist_album",
            recent_tracks=2,
            track_key_weight=0.0,
            artist_key_weight=1.7,
            album_key_weight=1.2,
            tag_key_weight=0.0,
            direct_label_weight=2.0,
            artist_expand_weight=0.35,
            tag_expand_weight=0.0,
            artist_top_tracks=80,
            tag_top_tracks=0,
            max_postings_per_key=220,
        ),
    }
    specs: list[TransitionSpec] = []
    for name in [item.strip() for item in value.split(",") if item.strip()]:
        if name not in registry:
            raise ValueError(f"Unknown transition variant: {name}. Valid: {sorted(registry)}")
        specs.append(registry[name])
    return specs


def prior_music_track_ids(task: Task, *, recent_tracks: int | None) -> list[str]:
    """task history から prior music track_id を新しい順で返す。"""
    track_ids = [
        str(row.get("content", ""))
        for row in task.history
        if row.get("role") == "music" and row.get("content")
    ]
    if recent_tracks is not None:
        track_ids = track_ids[-recent_tracks:]
    return list(reversed(track_ids))


def transition_keys_for_track(
    track_id: str,
    metadata: dict[str, dict[str, Any]],
    spec: TransitionSpec,
) -> list[tuple[str, float]]:
    """1 prior track から transition key と key weight を作る。"""
    item = metadata.get(track_id)
    if item is None:
        return []
    keys: list[tuple[str, float]] = []
    if spec.track_key_weight > 0:
        keys.append((f"track:{track_id}", spec.track_key_weight))
    if spec.artist_key_weight > 0:
        keys.extend((f"artist:{value}", spec.artist_key_weight) for value in metadata_values(item, "artist_name"))
    if spec.album_key_weight > 0:
        keys.extend((f"album:{value}", spec.album_key_weight) for value in metadata_values(item, "album_name"))
    if spec.tag_key_weight > 0:
        keys.extend((f"tag:{value}", spec.tag_key_weight) for value in metadata_values(item, "tag_list"))
    return keys


def add_label_expansion(
    scores: Counter[str],
    label: str,
    metadata: dict[str, dict[str, Any]],
    tracks_by_artist: dict[str, list[str]],
    tracks_by_tag: dict[str, list[str]],
    spec: TransitionSpec,
    *,
    weight: float,
) -> None:
    """transition 先 label とその metadata expansion を scores に足す。"""
    item = metadata.get(label)
    if item is None:
        return
    scores[label] += spec.direct_label_weight * weight
    if spec.artist_expand_weight > 0 and spec.artist_top_tracks > 0:
        for artist in metadata_values(item, "artist_name"):
            for rank, track_id in enumerate(tracks_by_artist.get(artist, [])[: spec.artist_top_tracks], start=1):
                scores[track_id] += spec.artist_expand_weight * weight / math.log2(rank + 2.0)
    if spec.tag_expand_weight > 0 and spec.tag_top_tracks > 0:
        for tag in metadata_values(item, "tag_list"):
            for rank, track_id in enumerate(tracks_by_tag.get(tag, [])[: spec.tag_top_tracks], start=1):
                scores[track_id] += spec.tag_expand_weight * weight / math.log2(rank + 2.0)


def build_label_expansions(
    train_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    tracks_by_artist: dict[str, list[str]],
    tracks_by_tag: dict[str, list[str]],
    spec: TransitionSpec,
) -> dict[str, list[tuple[str, float]]]:
    """train label ごとの展開済み candidate posting を事前計算する。

    Args:
        train_tasks: train split の tasks。
        metadata: track metadata。
        tracks_by_artist: artist -> popularity 順 tracks。
        tracks_by_tag: tag -> popularity 順 tracks。
        spec: transition memory variant。

    Returns:
        label track_id -> (candidate track_id, relative weight) の list。
    """
    labels = sorted({label for task in train_tasks for label in task.labels})
    expansions: dict[str, list[tuple[str, float]]] = {}
    for label in tqdm(labels, desc=f"label expansions:{spec.name}"):
        scores: Counter[str] = Counter()
        add_label_expansion(
            scores,
            label,
            metadata,
            tracks_by_artist,
            tracks_by_tag,
            spec,
            weight=1.0,
        )
        expansions[label] = list(scores.items())
    return expansions


def build_transition_index(
    train_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    spec: TransitionSpec,
) -> TransitionIndex:
    """train tasks から transition memory index を作る。"""
    tracks_by_artist, tracks_by_tag, global_tracks = build_track_lists(metadata)
    label_expansions = build_label_expansions(
        train_tasks,
        metadata,
        tracks_by_artist,
        tracks_by_tag,
        spec,
    )
    postings: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tqdm(train_tasks, desc=f"build transition index:{spec.name}"):
        if not task.labels:
            continue
        prior_ids = prior_music_track_ids(task, recent_tracks=spec.recent_tracks)
        for recency_rank, prior_track_id in enumerate(prior_ids, start=1):
            recency_weight = 1.0 / math.log2(recency_rank + 1.0)
            for key, key_weight in transition_keys_for_track(prior_track_id, metadata, spec):
                for label in task.labels:
                    for track_id, relative_weight in label_expansions.get(label, []):
                        postings[key][track_id] += relative_weight * key_weight * recency_weight

    pruned: dict[str, Counter[str]] = {}
    for key, counter in postings.items():
        pruned[key] = Counter(dict(counter.most_common(spec.max_postings_per_key)))
    return TransitionIndex(
        postings=pruned,
        metadata=metadata,
        tracks_by_artist=tracks_by_artist,
        tracks_by_tag=tracks_by_tag,
        global_tracks=global_tracks,
    )


def rank_task(task: Task, index: TransitionIndex, spec: TransitionSpec, *, topk: int) -> list[str]:
    """1 task に対して transition memory candidates を rank 順に返す。"""
    scores: Counter[str] = Counter()
    for recency_rank, prior_track_id in enumerate(
        prior_music_track_ids(task, recent_tracks=spec.recent_tracks),
        start=1,
    ):
        recency_weight = 1.0 / math.log2(recency_rank + 1.0)
        for key, key_weight in transition_keys_for_track(prior_track_id, index.metadata, spec):
            for label, value in index.postings.get(key, {}).items():
                scores[label] += float(value) * key_weight * recency_weight

    ranked = sorted(
        scores,
        key=lambda track_id: (
            -scores[track_id],
            -stable_float(index.metadata.get(track_id, {}).get("popularity")),
            track_id,
        ),
    )
    seen = set(ranked)
    for track_id in index.global_tracks:
        if len(ranked) >= topk:
            break
        if track_id in seen:
            continue
        ranked.append(track_id)
        seen.add(track_id)
    return ranked[:topk]


def rank_split(
    index: TransitionIndex,
    target_tasks: list[Task],
    spec: TransitionSpec,
    *,
    topk: int,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """構築済み index で 1 split を rank し、metrics と tracks JSON を保存する。

    Args:
        index: train split から構築済みの transition index。
        target_tasks: rank 対象の task list。
        spec: transition memory variant。
        topk: 保存する推薦件数。
        output_dir: 出力ディレクトリ。
        output_prefix: ``dev`` や ``blindA`` などの出力 prefix。

    Returns:
        評価指標、active task 数、出力 path を含む summary。
    """
    predictions: dict[tuple[str, int], list[str]] = {}
    active = 0
    for task in tqdm(target_tasks, desc=f"rank transition {output_prefix}:{spec.name}"):
        if prior_music_track_ids(task, recent_tracks=spec.recent_tracks):
            active += 1
        predictions[(task.session_id, task.turn_number)] = rank_task(task, index, spec, topk=topk)
    metrics = evaluate_predictions(target_tasks, predictions)
    tracks_path = output_dir / f"{output_prefix}_tracks_exp024_transition_{spec.name}.json"
    write_tracks_json(tracks_path, target_tasks, predictions, topk=topk)
    return {
        "metrics": metrics,
        "active_tasks": active,
        "active_rate": active / len(target_tasks) if target_tasks else 0.0,
        "transition_key_count": len(index.postings),
        "tracks_path": str(tracks_path),
    }


def run_variant(
    train_tasks: list[Task],
    target_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    spec: TransitionSpec,
    *,
    topk: int,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """1 split / 1 variant を実行して metrics と tracks JSON を保存する。"""
    index = build_transition_index(train_tasks, metadata, spec)
    return rank_split(
        index,
        target_tasks,
        spec,
        topk=topk,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )


def main() -> None:
    """CLI entrypoint。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--track_metadata_dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--track_split_types", default="all_tracks")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--blind_dataset_name")
    parser.add_argument("--blind_split", default="test")
    parser.add_argument("--variants", default="recent1,recent2,all,artist_album")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("mcrs/experiments/exp024_transition_memory/results/transition_memory_top100"))
    args = parser.parse_args()

    split_types = [item.strip() for item in args.track_split_types.split(",") if item.strip()]
    metadata = load_metadata(args.track_metadata_dataset_name, split_types)
    train_tasks = load_tasks(args.conversation_dataset_name, args.train_split)
    dev_tasks = load_tasks(args.conversation_dataset_name, args.dev_split)
    specs = parse_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "topk": args.topk,
        "track_split_types": split_types,
        "train_task_count": len(train_tasks),
        "dev_task_count": len(dev_tasks),
        "variants": [spec.__dict__ for spec in specs],
        "dev": {},
    }
    if args.blind_dataset_name:
        blind_tasks = load_tasks(args.blind_dataset_name, args.blind_split)
        report["blind_task_count"] = len(blind_tasks)
        report["blind"] = {}
    else:
        blind_tasks = []
    for spec in specs:
        index = build_transition_index(train_tasks, metadata, spec)
        report["dev"][spec.name] = rank_split(
            index,
            dev_tasks,
            spec,
            topk=args.topk,
            output_dir=args.output_dir,
            output_prefix="dev",
        )
        if args.blind_dataset_name:
            report["blind"][spec.name] = rank_split(
                index,
                blind_tasks,
                spec,
                topk=args.topk,
                output_dir=args.output_dir,
                output_prefix="blindA",
            )
    report_path = args.output_dir / "transition_memory_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "dev": report["dev"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
