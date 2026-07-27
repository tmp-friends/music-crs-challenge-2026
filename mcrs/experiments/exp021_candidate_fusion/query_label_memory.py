"""train query-label 記憶を使う候補生成 source を検証する。

catalog metadata だけを検索する既存 BM25 / exact source とは別に、train split の
「ユーザ発話 n-gram と正解 track」の対応を軽量な転置 index として保持し、dev / blind
のクエリをその記憶へ照合する。Dev/Blind の label や thought は使わない。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from mcrs.ltr.data_utils import normalize_conversations, parse_turn
from mcrs.retrieval_modules.metadata_exact import as_list, normalize_text


STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "give",
    "have",
    "i",
    "in",
    "is",
    "it",
    "like",
    "me",
    "more",
    "music",
    "of",
    "on",
    "or",
    "other",
    "play",
    "recommend",
    "song",
    "songs",
    "some",
    "that",
    "the",
    "this",
    "to",
    "track",
    "tracks",
    "what",
    "with",
    "you",
}

NGRAM_WEIGHTS = {1: 1.0, 2: 2.6, 3: 4.0}


@dataclass(frozen=True)
class Task:
    """session-turn 単位の候補生成対象。"""

    session_id: str
    user_id: str
    turn_number: int
    history: list[dict[str, Any]]
    user_query: str
    labels: tuple[str, ...]
    user_profile: dict[str, Any]


@dataclass(frozen=True)
class VariantSpec:
    """query-memory source の入力 view と scoring 設定。"""

    name: str
    include_recent_user: bool = False
    include_history_music_metadata: bool = False
    include_profile: bool = False
    direct_weight: float = 1.0
    artist_weight: float = 0.45
    tag_weight: float = 0.12
    profile_weight: float = 0.45
    recent_weight: float = 0.55
    history_music_weight: float = 0.35


@dataclass
class QueryMemoryIndex:
    """train query n-gram から label 側 track/artist/tag へ展開する index。"""

    task_count: int
    ngram_df: Counter[str]
    track_counts: dict[str, Counter[str]]
    artist_counts: dict[str, Counter[str]]
    tag_counts: dict[str, Counter[str]]
    artist_track_scores: dict[str, Counter[str]]
    tag_track_scores: dict[str, Counter[str]]
    metadata: dict[str, dict[str, Any]]
    tracks_by_artist: dict[str, list[str]]
    tracks_by_tag: dict[str, list[str]]
    global_popular_tracks: list[str]


def parse_variants(value: str) -> list[VariantSpec]:
    """CLI の variant 名リストを VariantSpec に変換する。

    Args:
        value: カンマ区切りの variant 名。

    Returns:
        検証対象 variant の定義リスト。

    Raises:
        ValueError: 未知の variant 名が含まれる場合。
    """
    registry = {
        "current": VariantSpec(name="current"),
        "recent": VariantSpec(name="recent", include_recent_user=True),
        "profile": VariantSpec(name="profile", include_profile=True),
        "recent_profile": VariantSpec(
            name="recent_profile",
            include_recent_user=True,
            include_profile=True,
        ),
        "history_music": VariantSpec(
            name="history_music",
            include_recent_user=True,
            include_history_music_metadata=True,
            include_profile=True,
        ),
    }
    names = [item.strip() for item in value.split(",") if item.strip()]
    specs: list[VariantSpec] = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown variant: {name}. Valid: {sorted(registry)}")
        specs.append(registry[name])
    return specs


def stable_float(value: Any) -> float:
    """metadata の popularity などを安定した float に変換する。"""
    if isinstance(value, list):
        value = value[0] if value else 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_metadata(dataset_name: str, split_types: list[str]) -> dict[str, dict[str, Any]]:
    """track metadata を track_id keyed dict として読み込む。"""
    dataset = load_dataset(dataset_name)
    merged = concatenate_datasets([dataset[split] for split in split_types])
    return {str(item["track_id"]): dict(item) for item in merged}


def metadata_values(metadata: dict[str, Any], field: str) -> list[str]:
    """metadata field の正規化済み値リストを返す。"""
    return [normalize_text(value) for value in as_list(metadata.get(field)) if normalize_text(value)]


def build_track_lists(metadata: dict[str, dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    """artist / tag ごとの popularity 順 track list と global fallback を作る。"""
    by_artist: dict[str, list[str]] = defaultdict(list)
    by_tag: dict[str, list[str]] = defaultdict(list)
    for track_id, item in metadata.items():
        for artist in metadata_values(item, "artist_name"):
            by_artist[artist].append(track_id)
        for tag in metadata_values(item, "tag_list"):
            by_tag[tag].append(track_id)

    def sort_key(track_id: str) -> tuple[float, str]:
        return (-stable_float(metadata[track_id].get("popularity")), track_id)

    by_artist = {key: sorted(set(values), key=sort_key) for key, values in by_artist.items()}
    by_tag = {key: sorted(set(values), key=sort_key) for key, values in by_tag.items()}
    global_tracks = sorted(metadata, key=sort_key)
    return by_artist, by_tag, global_tracks


def is_content_token(token: str) -> bool:
    """query-memory index に採用する token か判定する。"""
    if token in STOPWORDS:
        return False
    if len(token) < 3 and not any(ch.isdigit() for ch in token):
        return False
    return True


def content_tokens(text: str) -> list[str]:
    """正規化済み query から stopword を除いた token 列を返す。"""
    return [token for token in normalize_text(text).split() if is_content_token(token)]


def weighted_ngrams(text: str, *, weight: float = 1.0, max_n: int = 3) -> dict[str, float]:
    """query text から重み付き n-gram 辞書を作る。

    Args:
        text: 入力テキスト。
        weight: view 全体の重み。
        max_n: 最大 n-gram 長。

    Returns:
        n-gram -> weight の辞書。
    """
    tokens = content_tokens(text)
    output: dict[str, float] = {}
    for n in range(1, min(max_n, len(tokens)) + 1):
        base_weight = NGRAM_WEIGHTS.get(n, 1.0)
        for start in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[start : start + n])
            output[phrase] = max(output.get(phrase, 0.0), base_weight * weight)
    return output


def merge_weighted_ngrams(target: dict[str, float], source: dict[str, float]) -> None:
    """同じ n-gram は大きい view weight を採用して merge する。"""
    for key, value in source.items():
        target[key] = max(target.get(key, 0.0), value)


def history_user_text(history: list[dict[str, Any]], recent_turns: int = 2) -> str:
    """直近 user 発話だけを結合する。assistant thought は使わない。"""
    user_rows = [
        row
        for row in history
        if row.get("role") == "user" and row.get("turn_number") is not None
    ]
    user_rows.sort(key=lambda row: int(row.get("turn_number", 0)))
    return "\n".join(str(row.get("content", "")) for row in user_rows[-recent_turns:])


def history_music_text(
    history: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    recent_tracks: int = 2,
) -> str:
    """履歴 music track を metadata text に展開する。

    現在 turn より前に提示済みの track だけを使い、未来 turn や label は参照しない。
    """
    track_ids = [
        str(row.get("content", ""))
        for row in history
        if row.get("role") == "music"
    ][-recent_tracks:]
    parts: list[str] = []
    for track_id in track_ids:
        item = metadata.get(track_id)
        if not item:
            continue
        values = []
        for field in ("track_name", "artist_name", "album_name", "tag_list"):
            values.extend(as_list(item.get(field)))
        parts.append(" ".join(str(value) for value in values))
    return "\n".join(parts)


def profile_text(profile: dict[str, Any]) -> str:
    """候補生成に使う user profile text を作る。

    profile は dataset に付属する高レベル属性だけを使い、conversation thought は使わない。
    """
    fields = (
        "preferred_musical_culture",
        "preferred_language",
        "country_name",
        "age_group",
        "gender",
    )
    return " ".join(str(profile.get(field, "")) for field in fields)


def task_ngrams(task: Task, spec: VariantSpec, metadata: dict[str, dict[str, Any]]) -> dict[str, float]:
    """variant に応じた task query view から weighted n-gram を作る。"""
    result = weighted_ngrams(task.user_query, weight=1.0)
    if spec.include_recent_user:
        merge_weighted_ngrams(
            result,
            weighted_ngrams(history_user_text(task.history), weight=spec.recent_weight),
        )
    if spec.include_history_music_metadata:
        merge_weighted_ngrams(
            result,
            weighted_ngrams(
                history_music_text(task.history, metadata),
                weight=spec.history_music_weight,
            ),
        )
    if spec.include_profile:
        merge_weighted_ngrams(
            result,
            weighted_ngrams(profile_text(task.user_profile), weight=spec.profile_weight),
        )
    return result


def load_tasks(dataset_name: str, split: str, *, limit_sessions: int | None = None) -> list[Task]:
    """HF conversation dataset を session-turn tasks に展開する。"""
    dataset = load_dataset(dataset_name, split=split)
    tasks: list[Task] = []
    for index, item in enumerate(dataset):
        if limit_sessions is not None and index >= limit_sessions:
            break
        conversations = normalize_conversations(item["conversations"])
        target_turn_numbers = sorted(
            {
                int(row.get("turn_number", 0))
                for row in conversations
                if row.get("role") == "user"
            }
        )
        for turn_number in target_turn_numbers:
            history, user_query, labels = parse_turn(conversations, turn_number)
            if user_query is None:
                continue
            tasks.append(
                Task(
                    session_id=str(item["session_id"]),
                    user_id=str(item["user_id"]),
                    turn_number=int(turn_number),
                    history=history,
                    user_query=user_query,
                    labels=tuple(labels),
                    user_profile=dict(item.get("user_profile") or {}),
                )
            )
    return tasks


def build_index(
    train_tasks: list[Task],
    metadata: dict[str, dict[str, Any]],
    *,
    min_ngram_df: int,
    min_expansion_ngram_df: int,
    max_postings_per_ngram: int,
    max_expanded_tracks_per_ngram: int,
    artist_top_tracks: int,
    tag_top_tracks: int,
) -> QueryMemoryIndex:
    """train tasks から query-memory index を作る。

    Args:
        train_tasks: train split の session-turn tasks。
        metadata: track metadata。
        min_ngram_df: train 内でこの出現 task 数未満の n-gram を捨てる。
        min_expansion_ngram_df: artist/tag 展開に使う最小 n-gram DF。
        max_postings_per_ngram: 各 n-gram で保持する track/artist/tag posting 上限。
        max_expanded_tracks_per_ngram: artist/tag 展開後に保持する track posting 上限。
        artist_top_tracks: 各 artist から展開する popularity 上位 track 数。
        tag_top_tracks: 各 tag から展開する popularity 上位 track 数。

    Returns:
        QueryMemoryIndex。
    """
    ngram_df: Counter[str] = Counter()
    track_counts: dict[str, Counter[str]] = defaultdict(Counter)
    artist_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tag_counts: dict[str, Counter[str]] = defaultdict(Counter)

    base_spec = VariantSpec(name="train_index")
    for task in tqdm(train_tasks, desc="build query-memory index"):
        ngrams = set(task_ngrams(task, base_spec, metadata))
        if not ngrams:
            continue
        for ngram in ngrams:
            ngram_df[ngram] += 1
        for label in task.labels:
            item = metadata.get(label)
            if item is None:
                continue
            artists = metadata_values(item, "artist_name")
            tags = metadata_values(item, "tag_list")
            for ngram in ngrams:
                track_counts[ngram][label] += 1
                for artist in artists:
                    artist_counts[ngram][artist] += 1
                for tag in tags:
                    tag_counts[ngram][tag] += 1

    filtered_ngram_df = Counter(
        {ngram: count for ngram, count in ngram_df.items() if count >= min_ngram_df}
    )

    def filter_by_df(counter_map: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
        """低頻度 n-gram の posting を展開前に落とす。"""
        return {
            ngram: counter
            for ngram, counter in counter_map.items()
            if ngram in filtered_ngram_df
        }

    def filter_by_expansion_df(counter_map: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
        """artist/tag 展開用に一定以上の DF がある n-gram だけを残す。"""
        return {
            ngram: counter
            for ngram, counter in counter_map.items()
            if filtered_ngram_df.get(ngram, 0) >= min_expansion_ngram_df
        }

    def prune(counter_map: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
        pruned: dict[str, Counter[str]] = {}
        for key, counter in counter_map.items():
            pruned[key] = Counter(dict(counter.most_common(max_postings_per_ngram)))
        return pruned

    tracks_by_artist, tracks_by_tag, global_popular_tracks = build_track_lists(metadata)
    filtered_track_counts = filter_by_df(track_counts)
    filtered_artist_counts = filter_by_expansion_df(artist_counts)
    filtered_tag_counts = filter_by_expansion_df(tag_counts)
    pruned_artist_counts = prune(filtered_artist_counts)
    pruned_tag_counts = prune(filtered_tag_counts)

    def expand_artists(counter_map: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
        expanded: dict[str, Counter[str]] = {}
        for ngram, counter in tqdm(counter_map.items(), desc="expand artist postings"):
            scores: Counter[str] = Counter()
            for artist, count in counter.items():
                for rank, track_id in enumerate(tracks_by_artist.get(artist, [])[:artist_top_tracks], start=1):
                    scores[track_id] += float(count) / math.log2(rank + 2.0)
            expanded[ngram] = Counter(dict(scores.most_common(max_expanded_tracks_per_ngram)))
        return expanded

    def expand_tags(counter_map: dict[str, Counter[str]]) -> dict[str, Counter[str]]:
        expanded: dict[str, Counter[str]] = {}
        for ngram, counter in tqdm(counter_map.items(), desc="expand tag postings"):
            scores: Counter[str] = Counter()
            for tag, count in counter.items():
                for rank, track_id in enumerate(tracks_by_tag.get(tag, [])[:tag_top_tracks], start=1):
                    scores[track_id] += float(count) / math.log2(rank + 2.0)
            expanded[ngram] = Counter(dict(scores.most_common(max_expanded_tracks_per_ngram)))
        return expanded

    return QueryMemoryIndex(
        task_count=len(train_tasks),
        ngram_df=filtered_ngram_df,
        track_counts=prune(filtered_track_counts),
        artist_counts=pruned_artist_counts,
        tag_counts=pruned_tag_counts,
        artist_track_scores=expand_artists(pruned_artist_counts),
        tag_track_scores=expand_tags(pruned_tag_counts),
        metadata=metadata,
        tracks_by_artist=tracks_by_artist,
        tracks_by_tag=tracks_by_tag,
        global_popular_tracks=global_popular_tracks,
    )


def idf(index: QueryMemoryIndex, ngram: str) -> float:
    """query-memory n-gram の IDF 風重みを返す。"""
    return math.log((index.task_count + 1.0) / (index.ngram_df.get(ngram, 0) + 1.0)) + 1.0


def rank_task(
    task: Task,
    index: QueryMemoryIndex,
    spec: VariantSpec,
    *,
    topk: int,
) -> list[str]:
    """1 task に対する query-memory 候補を rank 順に返す。"""
    ngrams = task_ngrams(task, spec, index.metadata)
    scores: Counter[str] = Counter()
    for ngram, query_weight in ngrams.items():
        base = query_weight * idf(index, ngram)
        for track_id, count in index.track_counts.get(ngram, {}).items():
            scores[track_id] += spec.direct_weight * base * float(count)
        for track_id, value in index.artist_track_scores.get(ngram, {}).items():
            scores[track_id] += spec.artist_weight * base * float(value)
        for track_id, value in index.tag_track_scores.get(ngram, {}).items():
            scores[track_id] += spec.tag_weight * base * float(value)

    ranked = sorted(
        scores,
        key=lambda track_id: (
            -scores[track_id],
            -stable_float(index.metadata.get(track_id, {}).get("popularity")),
            track_id,
        ),
    )
    seen = set(ranked)
    for track_id in index.global_popular_tracks:
        if len(ranked) >= topk:
            break
        if track_id in seen:
            continue
        ranked.append(track_id)
        seen.add(track_id)
    return ranked[:topk]


def ndcg_at_20(predictions: list[str], labels: tuple[str, ...]) -> float:
    """binary relevance の nDCG@20 を計算する。"""
    label_set = set(labels)
    if not label_set:
        return 0.0
    dcg = 0.0
    for rank, track_id in enumerate(predictions[:20], start=1):
        if track_id in label_set:
            dcg += 1.0 / math.log2(rank + 1.0)
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, min(len(label_set), 20) + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_predictions(
    tasks: list[Task],
    predictions: dict[tuple[str, int], list[str]],
    *,
    base_hit_keys: set[tuple[str, int]] | None = None,
) -> dict[str, float | int]:
    """source prediction の recall / nDCG 指標を集計する。"""
    total = 0
    hit_at = {20: 0, 100: 0, 500: 0}
    ndcg_sum = 0.0
    recovered_base_miss = 0
    base_miss_total = 0
    for task in tasks:
        key = (task.session_id, task.turn_number)
        labels = set(task.labels)
        if not labels:
            continue
        total += 1
        pred = predictions[key]
        ndcg_sum += ndcg_at_20(pred, task.labels)
        for k in hit_at:
            hit_at[k] += int(bool(labels & set(pred[:k])))
        if base_hit_keys is not None and key not in base_hit_keys:
            base_miss_total += 1
            recovered_base_miss += int(bool(labels & set(pred[:500])))
    metrics: dict[str, float | int] = {
        "tasks_with_labels": total,
        "standalone_ndcg@20": ndcg_sum / total if total else 0.0,
    }
    for k, value in hit_at.items():
        metrics[f"hit@{k}"] = value
        metrics[f"recall@{k}"] = value / total if total else 0.0
    if base_hit_keys is not None:
        metrics["base_miss_tasks"] = base_miss_total
        metrics["base_miss_recovered@500"] = recovered_base_miss
        metrics["base_miss_recovered_rate@500"] = (
            recovered_base_miss / base_miss_total if base_miss_total else 0.0
        )
    return metrics


def load_base_hit_keys(path: Path) -> set[tuple[str, int]]:
    """既存 candidate parquet から正解候補入り済み group key を読む。"""
    df = pd.read_parquet(path, columns=["session_id", "turn_number", "label"])
    positive = df[df["label"] > 0][["session_id", "turn_number"]].drop_duplicates()
    return {(str(row.session_id), int(row.turn_number)) for row in positive.itertuples(index=False)}


def write_tracks_json(
    path: Path,
    tasks: list[Task],
    predictions: dict[tuple[str, int], list[str]],
    *,
    topk: int,
) -> None:
    """fusion source として使える prediction JSON を書き出す。"""
    records = []
    for task in tasks:
        records.append(
            {
                "session_id": task.session_id,
                "user_id": task.user_id,
                "turn_number": task.turn_number,
                "predicted_track_ids": predictions[(task.session_id, task.turn_number)][:topk],
                "predicted_response": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_split(
    tasks: list[Task],
    index: QueryMemoryIndex,
    specs: list[VariantSpec],
    *,
    topk: int,
    artist_top_tracks: int,
    tag_top_tracks: int,
    output_dir: Path,
    output_prefix: str,
    base_hit_keys: set[tuple[str, int]] | None,
) -> dict[str, Any]:
    """複数 variant を 1 split に適用し、metrics と tracks JSON を保存する。"""
    summary: dict[str, Any] = {}
    for spec in specs:
        predictions: dict[tuple[str, int], list[str]] = {}
        for task in tqdm(tasks, desc=f"rank {output_prefix}:{spec.name}"):
            predictions[(task.session_id, task.turn_number)] = rank_task(
                task,
                index,
                spec,
                topk=topk,
            )
        metrics = evaluate_predictions(tasks, predictions, base_hit_keys=base_hit_keys)
        tracks_path = output_dir / f"{output_prefix}_tracks_exp021_qlm_{spec.name}.json"
        write_tracks_json(tracks_path, tasks, predictions, topk=topk)
        summary[spec.name] = {
            "metrics": metrics,
            "tracks_path": str(tracks_path),
        }
    return summary


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
    parser.add_argument("--variants", default="current,recent,profile,recent_profile,history_music")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--artist_top_tracks", type=int, default=120)
    parser.add_argument("--tag_top_tracks", type=int, default=80)
    parser.add_argument("--min_ngram_df", type=int, default=2)
    parser.add_argument("--min_expansion_ngram_df", type=int, default=10)
    parser.add_argument("--max_postings_per_ngram", type=int, default=200)
    parser.add_argument("--max_expanded_tracks_per_ngram", type=int, default=300)
    parser.add_argument("--limit_train_sessions", type=int)
    parser.add_argument("--limit_dev_sessions", type=int)
    parser.add_argument("--base_candidate_parquet", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("mcrs/experiments/exp021_candidate_fusion/results/query_label_memory"))
    args = parser.parse_args()

    split_types = [item.strip() for item in args.track_split_types.split(",") if item.strip()]
    metadata = load_metadata(args.track_metadata_dataset_name, split_types)
    train_tasks = load_tasks(
        args.conversation_dataset_name,
        args.train_split,
        limit_sessions=args.limit_train_sessions,
    )
    dev_tasks = load_tasks(
        args.conversation_dataset_name,
        args.dev_split,
        limit_sessions=args.limit_dev_sessions,
    )
    specs = parse_variants(args.variants)
    index = build_index(
        train_tasks,
        metadata,
        min_ngram_df=args.min_ngram_df,
        min_expansion_ngram_df=args.min_expansion_ngram_df,
        max_postings_per_ngram=args.max_postings_per_ngram,
        max_expanded_tracks_per_ngram=args.max_expanded_tracks_per_ngram,
        artist_top_tracks=args.artist_top_tracks,
        tag_top_tracks=args.tag_top_tracks,
    )
    base_hit_keys = load_base_hit_keys(args.base_candidate_parquet) if args.base_candidate_parquet else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "train_task_count": len(train_tasks),
        "dev_task_count": len(dev_tasks),
        "ngram_count": len(index.ngram_df),
        "track_ngram_count": len(index.track_counts),
        "artist_expansion_ngram_count": len(index.artist_track_scores),
        "tag_expansion_ngram_count": len(index.tag_track_scores),
        "variants": [spec.__dict__ for spec in specs],
        "topk": args.topk,
        "artist_top_tracks": args.artist_top_tracks,
        "tag_top_tracks": args.tag_top_tracks,
        "min_ngram_df": args.min_ngram_df,
        "min_expansion_ngram_df": args.min_expansion_ngram_df,
        "max_postings_per_ngram": args.max_postings_per_ngram,
        "max_expanded_tracks_per_ngram": args.max_expanded_tracks_per_ngram,
        "dev": run_split(
            dev_tasks,
            index,
            specs,
            topk=args.topk,
            artist_top_tracks=args.artist_top_tracks,
            tag_top_tracks=args.tag_top_tracks,
            output_dir=args.output_dir,
            output_prefix="dev",
            base_hit_keys=base_hit_keys,
        ),
    }

    if args.blind_dataset_name:
        blind_tasks = load_tasks(args.blind_dataset_name, args.blind_split)
        report["blind_task_count"] = len(blind_tasks)
        report["blind"] = run_split(
            blind_tasks,
            index,
            specs,
            topk=args.topk,
            artist_top_tracks=args.artist_top_tracks,
            tag_top_tracks=args.tag_top_tracks,
            output_dir=args.output_dir,
            output_prefix="blindA",
            base_hit_keys=None,
        )

    report_path = args.output_dir / "query_label_memory_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "dev": report["dev"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
