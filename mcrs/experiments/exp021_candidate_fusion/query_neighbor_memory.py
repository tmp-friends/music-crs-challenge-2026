"""train query 近傍の label を使う候補生成 source を検証する。

Kaggle の retrieve-rerank 系解法でよく使われる「似た訓練例の正解を候補に混ぜる」
発想を Music-CRS に移した実験。Track catalog を直接検索する既存 BM25 と異なり、
train split の user query / context を BM25 index 化し、dev / blind query の近傍
train task が持つ label track を推薦候補へ変換する。

Dev/Blind の label や assistant thought は使わない。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
from tqdm import tqdm

from mcrs.experiments.exp021_candidate_fusion.query_label_memory import (
    Task,
    VariantSpec,
    build_track_lists,
    evaluate_predictions,
    history_music_text,
    history_user_text,
    load_base_hit_keys,
    load_metadata,
    load_tasks,
    metadata_values,
    parse_variants,
    profile_text,
    stable_float,
    write_tracks_json,
)


@dataclass(frozen=True)
class NeighborConfig:
    """query-neighbor source の retrieval / expansion 設定。"""

    top_neighbors: int
    artist_top_tracks: int
    tag_top_tracks: int
    tag_weight_scale: float


def task_document(task: Task, spec: VariantSpec, metadata: dict[str, dict[str, Any]]) -> str:
    """variant に対応する BM25 document/query text を作る。

    Args:
        task: train/dev/blind の session-turn task。
        spec: 使用する context view。
        metadata: track metadata。

    Returns:
        BM25 index / query に渡す複数行テキスト。
    """
    parts = [task.user_query]
    if spec.include_recent_user:
        parts.append(history_user_text(task.history))
    if spec.include_history_music_metadata:
        parts.append(history_music_text(task.history, metadata))
    if spec.include_profile:
        parts.append(profile_text(task.user_profile))
    return "\n".join(part for part in parts if part)


def build_label_expansions(
    train_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    *,
    spec: VariantSpec,
    config: NeighborConfig,
) -> dict[str, list[tuple[str, float]]]:
    """train label track ごとの track expansion を事前計算する。

    Args:
        train_tasks: train split の tasks。
        metadata: track metadata。
        spec: source weight 設定。
        config: expansion 上限。

    Returns:
        label track_id -> ``(candidate_track_id, relative_weight)`` の list。
    """
    tracks_by_artist, tracks_by_tag, _ = build_track_lists(metadata)
    label_ids = sorted({label for task in train_tasks for label in task.labels})
    expansions: dict[str, list[tuple[str, float]]] = {}
    for label in tqdm(label_ids, desc="build label expansions"):
        item = metadata.get(label)
        if item is None:
            continue
        scores: Counter[str] = Counter({label: spec.direct_weight})
        for artist in metadata_values(item, "artist_name"):
            for rank, track_id in enumerate(
                tracks_by_artist.get(artist, [])[: config.artist_top_tracks],
                start=1,
            ):
                scores[track_id] += spec.artist_weight / math.log2(rank + 2.0)
        for tag in metadata_values(item, "tag_list"):
            for rank, track_id in enumerate(
                tracks_by_tag.get(tag, [])[: config.tag_top_tracks],
                start=1,
            ):
                scores[track_id] += (
                    spec.tag_weight
                    * config.tag_weight_scale
                    / math.log2(rank + 2.0)
                )
        expansions[label] = list(scores.items())
    return expansions


def build_bm25(corpus: list[str]) -> bm25s.BM25:
    """train task corpus から BM25 index を作る。"""
    tokens = bm25s.tokenize(corpus, stopwords="english")
    model = bm25s.BM25()
    model.index(tokens)
    return model


def retrieve_neighbors(
    model: bm25s.BM25,
    queries: list[str],
    *,
    top_neighbors: int,
    n_threads: int,
    chunksize: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """dev/blind queries に対する train task 近傍を batch retrieval する。"""
    query_tokens = bm25s.tokenize(queries, stopwords="english")
    result = model.retrieve(
        query_tokens,
        k=top_neighbors,
        return_as="tuple",
        n_threads=n_threads,
        chunksize=chunksize,
    )
    indices: list[list[int]] = []
    scores: list[list[float]] = []
    for documents, score_values in zip(result.documents, result.scores):
        indices.append([int(item["id"] if isinstance(item, dict) else item) for item in documents])
        scores.append([float(value) for value in score_values])
    return indices, scores


def rank_from_neighbors(
    neighbor_indices: list[int],
    neighbor_scores: list[float],
    train_tasks: list[Task],
    label_expansions: dict[str, list[tuple[str, float]]],
    metadata: dict[str, dict[str, Any]],
    global_popular_tracks: list[str],
    *,
    topk: int,
) -> list[str]:
    """近傍 train task の label expansion を集約して ranked candidates を返す。"""
    scores: Counter[str] = Counter()
    for rank, (task_index, neighbor_score) in enumerate(
        zip(neighbor_indices, neighbor_scores),
        start=1,
    ):
        if neighbor_score <= 0.0:
            continue
        neighbor_weight = math.log1p(neighbor_score) / math.log2(rank + 2.0)
        for label in train_tasks[task_index].labels:
            for track_id, relative_weight in label_expansions.get(label, []):
                scores[track_id] += neighbor_weight * relative_weight

    ranked = sorted(
        scores,
        key=lambda track_id: (
            -scores[track_id],
            -stable_float(metadata.get(track_id, {}).get("popularity")),
            track_id,
        ),
    )
    seen = set(ranked)
    for track_id in global_popular_tracks:
        if len(ranked) >= topk:
            break
        if track_id in seen:
            continue
        ranked.append(track_id)
        seen.add(track_id)
    return ranked[:topk]


def run_variant(
    train_tasks: list[Task],
    target_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    spec: VariantSpec,
    *,
    config: NeighborConfig,
    topk: int,
    n_threads: int,
    chunksize: int,
    base_hit_keys: set[tuple[str, int]] | None,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """1 variant を 1 split に適用し、metrics と JSON artifact を保存する。"""
    train_docs = [task_document(task, spec, metadata) for task in tqdm(train_tasks, desc=f"docs train:{spec.name}")]
    target_docs = [task_document(task, spec, metadata) for task in tqdm(target_tasks, desc=f"docs {output_prefix}:{spec.name}")]
    model = build_bm25(train_docs)
    neighbor_indices, neighbor_scores = retrieve_neighbors(
        model,
        target_docs,
        top_neighbors=config.top_neighbors,
        n_threads=n_threads,
        chunksize=chunksize,
    )
    label_expansions = build_label_expansions(
        train_tasks,
        metadata,
        spec=spec,
        config=config,
    )
    _, _, global_popular_tracks = build_track_lists(metadata)

    predictions: dict[tuple[str, int], list[str]] = {}
    for task, indices, scores in tqdm(
        zip(target_tasks, neighbor_indices, neighbor_scores),
        total=len(target_tasks),
        desc=f"rank {output_prefix}:{spec.name}",
    ):
        predictions[(task.session_id, task.turn_number)] = rank_from_neighbors(
            indices,
            scores,
            train_tasks,
            label_expansions,
            metadata,
            global_popular_tracks,
            topk=topk,
        )

    metrics = evaluate_predictions(target_tasks, predictions, base_hit_keys=base_hit_keys)
    tracks_path = output_dir / f"{output_prefix}_tracks_exp021_qnm_{spec.name}.json"
    write_tracks_json(tracks_path, target_tasks, predictions, topk=topk)
    return {
        "metrics": metrics,
        "tracks_path": str(tracks_path),
    }


def run_split(
    train_tasks: list[Task],
    target_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    specs: list[VariantSpec],
    *,
    config: NeighborConfig,
    topk: int,
    n_threads: int,
    chunksize: int,
    base_hit_keys: set[tuple[str, int]] | None,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Any]:
    """複数 variant を 1 split に適用する。"""
    summary: dict[str, Any] = {}
    for spec in specs:
        summary[spec.name] = run_variant(
            train_tasks,
            target_tasks,
            metadata,
            spec,
            config=config,
            topk=topk,
            n_threads=n_threads,
            chunksize=chunksize,
            base_hit_keys=base_hit_keys,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    return summary


def summarize_overlap(
    target_tasks: list[Task],
    source_paths: dict[str, str],
    base_candidate_parquet: Path | None,
) -> dict[str, Any]:
    """source 間と base candidate の label hit を簡易比較する。"""
    base_hit_keys = load_base_hit_keys(base_candidate_parquet) if base_candidate_parquet else set()
    task_label_map = {
        (task.session_id, task.turn_number): set(task.labels)
        for task in target_tasks
        if task.labels
    }
    rows: list[dict[str, Any]] = []
    for name, path in source_paths.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hit_keys = set()
        for item in data:
            key = (str(item["session_id"]), int(item["turn_number"]))
            labels = task_label_map.get(key, set())
            if labels & set(item["predicted_track_ids"][:500]):
                hit_keys.add(key)
        rows.append(
            {
                "source": name,
                "hit@500": len(hit_keys),
                "base_miss_hit@500": len(hit_keys - base_hit_keys),
                "base_overlap_hit@500": len(hit_keys & base_hit_keys),
            }
        )
    return {"source_hit_rows": rows}


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
    parser.add_argument("--skip_dev", action="store_true")
    parser.add_argument("--variants", default="current,recent_profile,history_music")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--top_neighbors", type=int, default=160)
    parser.add_argument("--artist_top_tracks", type=int, default=80)
    parser.add_argument("--tag_top_tracks", type=int, default=24)
    parser.add_argument("--tag_weight_scale", type=float, default=0.35)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--limit_train_sessions", type=int)
    parser.add_argument("--limit_dev_sessions", type=int)
    parser.add_argument("--base_candidate_parquet", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory"))
    args = parser.parse_args()

    split_types = [item.strip() for item in args.track_split_types.split(",") if item.strip()]
    metadata = load_metadata(args.track_metadata_dataset_name, split_types)
    train_tasks = load_tasks(
        args.conversation_dataset_name,
        args.train_split,
        limit_sessions=args.limit_train_sessions,
    )
    dev_tasks = [] if args.skip_dev else load_tasks(
        args.conversation_dataset_name,
        args.dev_split,
        limit_sessions=args.limit_dev_sessions,
    )
    specs = parse_variants(args.variants)
    config = NeighborConfig(
        top_neighbors=args.top_neighbors,
        artist_top_tracks=args.artist_top_tracks,
        tag_top_tracks=args.tag_top_tracks,
        tag_weight_scale=args.tag_weight_scale,
    )
    base_hit_keys = load_base_hit_keys(args.base_candidate_parquet) if args.base_candidate_parquet else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "train_task_count": len(train_tasks),
        "dev_task_count": len(dev_tasks),
        "variants": [spec.__dict__ for spec in specs],
        "topk": args.topk,
        "top_neighbors": args.top_neighbors,
        "artist_top_tracks": args.artist_top_tracks,
        "tag_top_tracks": args.tag_top_tracks,
        "tag_weight_scale": args.tag_weight_scale,
        "dev": {},
    }
    if not args.skip_dev:
        report["dev"] = run_split(
            train_tasks,
            dev_tasks,
            metadata,
            specs,
            config=config,
            topk=args.topk,
            n_threads=args.n_threads,
            chunksize=args.chunksize,
            base_hit_keys=base_hit_keys,
            output_dir=args.output_dir,
            output_prefix="dev",
        )
        report["dev_overlap"] = summarize_overlap(
            dev_tasks,
            {name: value["tracks_path"] for name, value in report["dev"].items()},
            args.base_candidate_parquet,
        )

    if args.blind_dataset_name:
        blind_tasks = load_tasks(args.blind_dataset_name, args.blind_split)
        report["blind_task_count"] = len(blind_tasks)
        report["blind"] = run_split(
            train_tasks,
            blind_tasks,
            metadata,
            specs,
            config=config,
            topk=args.topk,
            n_threads=args.n_threads,
            chunksize=args.chunksize,
            base_hit_keys=None,
            output_dir=args.output_dir,
            output_prefix="blindA",
        )

    report_path = args.output_dir / "query_neighbor_memory_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "dev": report["dev"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
