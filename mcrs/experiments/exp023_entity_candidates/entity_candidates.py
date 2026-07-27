"""metadata entity phrase から候補 track を生成する。

current user turn と直近 user 発話に含まれる track / artist / album / tag 名を
catalog metadata の正規化 phrase index に照合し、該当 entity の track を popularity
順で展開する。BM25 ではなく構造化 metadata entity の candidate source として
standalone 指標と fusion 用 tracks JSON を作る。
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
from mcrs.retrieval_modules.metadata_exact import normalize_text


FIELD_WEIGHTS = {
    "track_name": 4.0,
    "artist_name": 2.8,
    "album_name": 2.2,
    "tag_list": 0.9,
}


@dataclass(frozen=True)
class EntitySpec:
    """entity candidate source の入力 view と展開上限。"""

    name: str
    include_current: bool
    include_recent_user: bool
    recent_user_turns: int
    top_tracks_per_entity: int
    tag_top_tracks_per_entity: int


def parse_variants(value: str) -> list[EntitySpec]:
    """variant 名を EntitySpec に変換する。

    Args:
        value: カンマ区切り variant 名。

    Returns:
        EntitySpec の list。
    """
    registry = {
        "current": EntitySpec(
            name="current",
            include_current=True,
            include_recent_user=False,
            recent_user_turns=0,
            top_tracks_per_entity=180,
            tag_top_tracks_per_entity=60,
        ),
        "recent2": EntitySpec(
            name="recent2",
            include_current=True,
            include_recent_user=True,
            recent_user_turns=2,
            top_tracks_per_entity=180,
            tag_top_tracks_per_entity=60,
        ),
        "recent4": EntitySpec(
            name="recent4",
            include_current=True,
            include_recent_user=True,
            recent_user_turns=4,
            top_tracks_per_entity=180,
            tag_top_tracks_per_entity=60,
        ),
        "current_tight": EntitySpec(
            name="current_tight",
            include_current=True,
            include_recent_user=False,
            recent_user_turns=0,
            top_tracks_per_entity=80,
            tag_top_tracks_per_entity=24,
        ),
    }
    specs: list[EntitySpec] = []
    for name in [item.strip() for item in value.split(",") if item.strip()]:
        if name not in registry:
            raise ValueError(f"Unknown entity variant: {name}. Valid: {sorted(registry)}")
        specs.append(registry[name])
    return specs


def build_field_index(
    metadata: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, list[str]]:
    """metadata field value -> popularity 順 track list の index を作る。"""
    index: dict[str, list[str]] = defaultdict(list)
    for track_id, item in metadata.items():
        for value in metadata_values(item, field):
            if len(value) >= 3:
                index[value].append(track_id)

    def sort_key(track_id: str) -> tuple[float, str]:
        return (-stable_float(metadata[track_id].get("popularity")), track_id)

    return {value: sorted(set(track_ids), key=sort_key) for value, track_ids in index.items()}


def build_indexes(metadata: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """全 field の entity index と popularity fallback を作る。"""
    indexes = {field: build_field_index(metadata, field) for field in FIELD_WEIGHTS}
    global_tracks = sorted(
        metadata,
        key=lambda track_id: (-stable_float(metadata[track_id].get("popularity")), track_id),
    )
    return indexes, global_tracks


def recent_user_text(task: Task, *, turns: int) -> str:
    """task history の直近 user 発話を結合する。"""
    rows = [
        row
        for row in task.history
        if row.get("role") == "user" and row.get("turn_number") is not None
    ]
    rows.sort(key=lambda row: int(row.get("turn_number", 0)))
    return "\n".join(str(row.get("content", "")) for row in rows[-turns:])


def phrase_ngrams(text: str, *, max_n: int = 8) -> set[str]:
    """正規化 text から phrase matching 用 n-gram set を作る。"""
    tokens = normalize_text(text).split()
    phrases: set[str] = set()
    for n in range(1, min(max_n, len(tokens)) + 1):
        for start in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[start : start + n])
            if len(phrase) >= 3:
                phrases.add(phrase)
    return phrases


def matched_entities(
    task: Task,
    spec: EntitySpec,
    indexes: dict[str, dict[str, list[str]]],
) -> dict[str, set[str]]:
    """task の text view に含まれる metadata entity phrase を field 別に返す。"""
    text_parts: list[tuple[str, float]] = []
    if spec.include_current:
        text_parts.append((task.user_query, 1.0))
    if spec.include_recent_user and spec.recent_user_turns > 0:
        text_parts.append((recent_user_text(task, turns=spec.recent_user_turns), 0.45))

    output: dict[str, set[str]] = {field: set() for field in indexes}
    for text, _ in text_parts:
        phrases = phrase_ngrams(text)
        for field, index in indexes.items():
            output[field].update(phrase for phrase in phrases if phrase in index)
    return output


def rank_task(
    task: Task,
    spec: EntitySpec,
    indexes: dict[str, dict[str, list[str]]],
    metadata: dict[str, dict[str, Any]],
    global_tracks: list[str],
    *,
    topk: int,
) -> list[str]:
    """1 task に対する entity candidate ranked list を返す。"""
    scores: Counter[str] = Counter()
    entities = matched_entities(task, spec, indexes)
    for field, phrases in entities.items():
        top_tracks = spec.tag_top_tracks_per_entity if field == "tag_list" else spec.top_tracks_per_entity
        for phrase in phrases:
            for rank, track_id in enumerate(indexes[field].get(phrase, [])[:top_tracks], start=1):
                scores[track_id] += FIELD_WEIGHTS[field] / math.log2(rank + 2.0)

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
        if track_id in seen:
            continue
        ranked.append(track_id)
        seen.add(track_id)
    return ranked[:topk]


def run_variant(
    tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    indexes: dict[str, dict[str, list[str]]],
    global_tracks: list[str],
    spec: EntitySpec,
    *,
    topk: int,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """1 split / 1 variant の prediction JSON と metrics を作る。"""
    predictions: dict[tuple[str, int], list[str]] = {}
    active = 0
    active_by_field: Counter[str] = Counter()
    for task in tqdm(tasks, desc=f"entity rank {output_prefix}:{spec.name}"):
        entities = matched_entities(task, spec, indexes)
        if any(entities.values()):
            active += 1
        for field, values in entities.items():
            if values:
                active_by_field[field] += 1
        predictions[(task.session_id, task.turn_number)] = rank_task(
            task,
            spec,
            indexes,
            metadata,
            global_tracks,
            topk=topk,
        )
    metrics = evaluate_predictions(tasks, predictions)
    tracks_path = output_dir / f"{output_prefix}_tracks_exp023_entity_{spec.name}.json"
    write_tracks_json(tracks_path, tasks, predictions, topk=topk)
    return {
        "metrics": metrics,
        "active_tasks": active,
        "active_rate": active / len(tasks) if tasks else 0.0,
        "active_by_field": dict(active_by_field),
        "tracks_path": str(tracks_path),
    }


def run_split(
    tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    specs: list[EntitySpec],
    *,
    topk: int,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """複数 entity variant を 1 split に適用する。"""
    indexes, global_tracks = build_indexes(metadata)
    return {
        spec.name: run_variant(
            tasks,
            metadata,
            indexes,
            global_tracks,
            spec,
            topk=topk,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
        for spec in specs
    }


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
    parser.add_argument("--variants", default="current,recent2,recent4,current_tight")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--output_dir", type=Path, default=Path("mcrs/experiments/exp023_entity_candidates/results/entity_candidates"))
    args = parser.parse_args()

    split_types = [item.strip() for item in args.track_split_types.split(",") if item.strip()]
    metadata = load_metadata(args.track_metadata_dataset_name, split_types)
    specs = parse_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "topk": args.topk,
        "track_split_types": split_types,
        "variants": [spec.__dict__ for spec in specs],
        "dev": {},
    }
    if not args.skip_dev:
        dev_tasks = load_tasks(args.conversation_dataset_name, args.dev_split)
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

    report_path = args.output_dir / "entity_candidates_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "dev": report.get("dev", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
