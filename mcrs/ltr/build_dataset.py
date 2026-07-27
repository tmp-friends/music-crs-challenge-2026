"""LightGBM LTR 学習用 row dataset を構築する（共有ライブラリ）。

exp014 で実装した dataset builder（exp014 base パイプライン）を ``mcrs/ltr/`` へ切り出した
共有モジュール。本モジュールの ``main(args)`` は ``args`` のみを取り、feature builder /
candidate builder の injection hook はまだ持たない（その hook は exp015c fork 側にある。
追加方針は docs/experiment-guide.md §3 参照）。旧
``mcrs.experiments.exp014_lightgbm_ltr_reranker.build_dataset`` は re-export shim 経由で
このモジュールへ解決され、shim 側にも ``__main__`` 委譲があるため旧パスの CLI 実行も維持される。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from mcrs.db_item import MusicCatalogDB
from mcrs.db_item.track_embeddings import TrackEmbeddingDB
from mcrs.db_user.user_embeddings import UserEmbeddingDB
from mcrs.candidates.candidate_sources import (
    SOURCE_NAMES,
    WideCandidateBuilder,
)
from mcrs.ltr.data_utils import (
    config_list,
    prepare_tasks,
    validate_config,
)
from mcrs.ltr.features import (
    LGBMFeatureBuilder,
    PRIMARY_TEXT5_SOURCES,
)
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.metadata_exact import DEFAULT_METADATA_FIELDS, MetadataExactRetriever


def parse_sources(value: str | None) -> list[str]:
    """カンマ区切りの候補ソース名文字列を検証してリストに変換する。

    Args:
        value: カンマ区切りのソース名文字列。None または空文字列の場合は
            PRIMARY_TEXT5_SOURCES を返す。

    Returns:
        有効なソース名のリスト。

    Raises:
        ValueError: 未知のソース名が含まれる場合。
    """
    if value is None or not str(value).strip():
        return list(PRIMARY_TEXT5_SOURCES)
    sources = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(sources) - set(SOURCE_NAMES))
    if unknown:
        raise ValueError(f"Unknown candidate sources: {unknown}. Valid: {SOURCE_NAMES}")
    return sources


def load_config(path: str) -> Any:
    """OmegaConf で YAML / JSON 形式の候補生成 config を読み込む。

    Args:
        path: config ファイルのパス。

    Returns:
        OmegaConf の DictConfig オブジェクト。
    """
    return OmegaConf.load(Path(path))


def stable_seed(group_id: str, seed: int) -> int:
    """group_id と seed から再現可能な 32bit seed を生成する。

    Args:
        group_id: session-turn を表す group ID。
        seed: 実験全体の seed。

    Returns:
        random.Random に渡せる 32bit 整数 seed。
    """
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def binary_dcg(positive_count: int, k: int) -> float:
    """binary relevance の理想 DCG を正例数から計算する。

    Args:
        positive_count: 上位に並べられる正例数。
        k: 評価打ち切り位置。

    Returns:
        DCG 値。
    """
    return sum(1.0 / math.log2(rank + 2) for rank in range(min(positive_count, k)))


def oracle_ndcg_at_k(candidates: list[str], labels: list[str], k: int) -> float:
    """候補集合内で正例を理想順位に置いた場合の nDCG@k を計算する。

    Args:
        candidates: rank 順の候補 track_id。
        labels: 正例 track_id。
        k: 評価打ち切り位置。

    Returns:
        候補集合の oracle nDCG@k。正例がない task は 0.0。
    """
    label_set = set(labels)
    if not label_set:
        return 0.0
    candidate_positive_count = len(label_set & set(candidates))
    dcg = binary_dcg(candidate_positive_count, k)
    idcg = binary_dcg(len(label_set), k)
    return dcg / idcg if idcg > 0 else 0.0


def candidate_recall_at_k(candidates: list[str], labels: list[str], k: int) -> float:
    """候補上位 k 件に含まれる正例 recall を計算する。

    Args:
        candidates: rank 順の候補 track_id。
        labels: 正例 track_id。
        k: 評価打ち切り位置。

    Returns:
        正例 track_id の recall。正例がない task は 0.0。
    """
    label_set = set(labels)
    if not label_set:
        return 0.0
    hits = len(label_set & set(candidates[:k]))
    return hits / len(label_set)


def new_candidate_metric_state() -> dict[str, Any]:
    """candidate recall / oracle 集計用の初期 state を返す。

    Returns:
        build_split_rows 内で更新する集計辞書。
    """
    return {
        "task_count": 0,
        "tasks_with_labels": 0,
        "groups_with_positive": 0,
        "recall_sum_by_k": {20: 0.0, 100: 0.0, 500: 0.0, 1000: 0.0},
        "recall_all_sum": 0.0,
        "oracle_ndcg@20_sum": 0.0,
        "candidate_counts": [],
    }


def update_candidate_metric_state(
    state: dict[str, Any],
    *,
    candidates: list[str],
    labels: list[str],
) -> None:
    """1 task 分の candidate recall / oracle 指標を集計 state へ加算する。

    Args:
        state: new_candidate_metric_state が返した集計辞書。
        candidates: rank 順の候補 track_id。
        labels: 正例 track_id。
    """
    state["task_count"] += 1
    state["candidate_counts"].append(len(candidates))
    if not labels:
        return
    state["tasks_with_labels"] += 1
    label_set = set(labels)
    state["groups_with_positive"] += int(bool(label_set & set(candidates)))
    for k in state["recall_sum_by_k"]:
        state["recall_sum_by_k"][k] += candidate_recall_at_k(candidates, labels, k)
    state["recall_all_sum"] += candidate_recall_at_k(candidates, labels, len(candidates))
    state["oracle_ndcg@20_sum"] += oracle_ndcg_at_k(candidates, labels, 20)


def percentile(values: list[int], q: float) -> float:
    """整数リストの percentile を線形補間で返す。

    Args:
        values: 集計対象の整数値。
        q: 0.0 以上 1.0 以下の percentile。

    Returns:
        percentile 値。values が空の場合は 0.0。
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def finalize_candidate_metrics(state: dict[str, Any]) -> dict[str, float | int]:
    """candidate recall / oracle の平均値を summary 用 dict に変換する。

    Args:
        state: update_candidate_metric_state で更新した集計辞書。

    Returns:
        summary JSON に保存する metric dict。
    """
    denominator = int(state["tasks_with_labels"])
    metrics: dict[str, float | int] = {
        "candidate_metric_task_count": denominator,
        "candidate_hit_groups": int(state["groups_with_positive"]),
    }
    for k, value in state["recall_sum_by_k"].items():
        metrics[f"candidate_recall@{k}"] = value / denominator if denominator else 0.0
    metrics["candidate_recall@all"] = (
        state["recall_all_sum"] / denominator if denominator else 0.0
    )
    metrics["oracle_ndcg@20"] = (
        state["oracle_ndcg@20_sum"] / denominator if denominator else 0.0
    )
    counts = list(state.get("candidate_counts", []))
    metrics.update(
        {
            "candidate_count_mean": (sum(counts) / len(counts)) if counts else 0.0,
            "candidate_count_p50": percentile(counts, 0.50),
            "candidate_count_p95": percentile(counts, 0.95),
            "candidate_count_p99": percentile(counts, 0.99),
            "candidate_count_max": max(counts) if counts else 0,
        }
    )
    return metrics


def sample_union_candidates(
    union_candidates: list[str],
    labels: list[str],
    *,
    max_rows_per_group: int | None,
    head_negatives: int,
    seed: int,
    group_id: str,
) -> list[str]:
    """topK 候補 pool から正例全保持 + head/tail negative sampling で候補を絞る。

    top500 など広い候補生成の効果は保ちつつ、train split の特徴量構築と
    parquet 書き込み量を抑えるために使う。順位は元の union order を維持する。

    Args:
        union_candidates: RRF union 後の候補 track ID。順序は candidate rank。
        labels: group の正例 track ID。
        max_rows_per_group: 最終的に残す目標行数。None 以下の場合は sampling しない。
        head_negatives: 上位から必ず残す負例数。
        seed: tail negative sampling の seed。
        group_id: session-turn を表す group ID。

    Returns:
        sampling 後の候補 track ID。正例が多い場合は max_rows_per_group を超えても全保持する。
    """
    if max_rows_per_group is None or max_rows_per_group <= 0:
        return union_candidates
    if len(union_candidates) <= max_rows_per_group:
        return union_candidates

    label_set = set(labels)
    positives = [track_id for track_id in union_candidates if track_id in label_set]
    negatives = [track_id for track_id in union_candidates if track_id not in label_set]
    head_limit = max(0, min(head_negatives, max_rows_per_group - len(positives)))
    head = negatives[:head_limit]
    tail = negatives[head_limit:]
    remaining = max_rows_per_group - len(positives) - len(head)
    if remaining <= 0:
        tail_sample: list[str] = []
    elif len(tail) <= remaining:
        tail_sample = tail
    else:
        rng = random.Random(stable_seed(group_id, seed))
        tail_sample = rng.sample(tail, remaining)

    keep = set(positives)
    keep.update(head)
    keep.update(tail_sample)
    return [track_id for track_id in union_candidates if track_id in keep]


def init_builders(
    config: Any,
    *,
    enabled_sources: list[str],
    include_embedding_features: bool,
    metadata_embedding_name: str | None,
    recent_turn_count: int,
    bm25_n_threads: int,
    bm25_chunksize: int,
    parallel_bm25_sources: bool,
    bm25_source_workers: int,
) -> tuple[WideCandidateBuilder, LGBMFeatureBuilder, set[str]]:
    """Wide 候補生成・特徴量構築に必要な各種 DB と builder を初期化して返す。

    Args:
        config: 候補生成 config（OmegaConf DictConfig）。
        enabled_sources: 有効にする候補ソース名のリスト。
        include_embedding_features: audio / metadata embedding 特徴量を含めるか。
        metadata_embedding_name: metadata embedding の名前。None の場合はスキップ。
        recent_turn_count: 直近何ターンを "recent" クエリに用いるか。
        bm25_n_threads: bm25s retrieval に渡す thread 数。0 は sequential。
        bm25_chunksize: bm25s retrieval に渡す chunksize。
        parallel_bm25_sources: True の場合、独立した BM25 source を外側で並列実行する。
        bm25_source_workers: BM25 source 並列の worker 数。

    Returns:
        (WideCandidateBuilder, LGBMFeatureBuilder, catalog_track_ids の集合) のタプル。
    """
    validate_config(config)
    metadata_fields = config_list(config, "metadata_exact_fields", list(DEFAULT_METADATA_FIELDS))
    metadata_bm25_corpus_types = config_list(
        config, "metadata_bm25_corpus_types", list(DEFAULT_METADATA_FIELDS),
    )
    corpus_types = list(config.corpus_types)
    # 両 BM25 の corpus_types を統合して item_db に渡し、テキストインデックスを共有する
    item_db = MusicCatalogDB(
        config.item_db_name,
        list(config.track_split_types),
        sorted(set([*corpus_types, *metadata_bm25_corpus_types])),
    )
    baseline_bm25 = BM25_MODEL(
        config.item_db_name,
        list(config.track_split_types),
        corpus_types,
        config.cache_dir,
        n_threads=bm25_n_threads,
        chunksize=bm25_chunksize,
    )
    metadata_bm25 = BM25_MODEL(
        config.item_db_name,
        list(config.track_split_types),
        metadata_bm25_corpus_types,
        config.cache_dir,
        n_threads=bm25_n_threads,
        chunksize=bm25_chunksize,
    )
    exact_retriever = MetadataExactRetriever(
        dataset_name=config.item_db_name,
        split_types=list(config.track_split_types),
        metadata_fields=metadata_fields,
        cache_dir=config.cache_dir,
    )

    enabled_set = set(enabled_sources)
    # BERT retriever は重いため enabled_sources に "dense_text_recent" が含まれる場合のみロードする
    needs_bert = "dense_text_recent" in enabled_set
    bert_retriever = None
    if needs_bert and config.get("bert_model_name", None):
        from mcrs.retrieval_modules.bert import BERT_MODEL

        bert_retriever = BERT_MODEL(
            config.item_db_name,
            list(config.track_split_types),
            corpus_types,
            config.cache_dir,
            model_name=config.get("bert_model_name"),
            max_length=config.get("bert_max_length", 128),
        )

    # CF-BPR embedding DB は CF ソースまたは embedding 特徴量が有効な場合のみ初期化する
    track_embedding_db = None
    user_embedding_db = None
    if include_embedding_features or bool(enabled_set & {"user_cf_bpr", "history_cf_bpr_item2item"}):
        track_embedding_db = TrackEmbeddingDB(
            dataset_name=config.get(
                "track_embedding_dataset_name",
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            ),
            split_types=list(config.track_split_types),
            embedding_name=config.get("track_embedding_name", "cf-bpr"),
        )
        user_embedding_db = UserEmbeddingDB(
            dataset_name=config.get(
                "user_embedding_dataset_name",
                "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            ),
            split_types=config.get("user_embedding_split_types", None),
            embedding_name=config.get("user_embedding_name", "cf-bpr"),
        )

    # audio / metadata embedding DB は include_embedding_features が有効な場合のみ初期化する
    audio_embedding_db = None
    metadata_embedding_db = None
    if include_embedding_features:
        audio_embedding_db = TrackEmbeddingDB(
            dataset_name=config.get(
                "track_embedding_dataset_name",
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            ),
            split_types=list(config.track_split_types),
            embedding_name=config.get("audio_embedding_name", "audio-laion_clap"),
        )
        if metadata_embedding_name:
            metadata_embedding_db = TrackEmbeddingDB(
                dataset_name=config.get(
                    "track_embedding_dataset_name",
                    "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                ),
                split_types=list(config.track_split_types),
                embedding_name=metadata_embedding_name,
            )

    wide_builder = WideCandidateBuilder(
        item_db=item_db,
        baseline_bm25=baseline_bm25,
        metadata_bm25=metadata_bm25,
        exact_retriever=exact_retriever,
        bert_retriever=bert_retriever,
        track_embedding_db=track_embedding_db,
        user_embedding_db=user_embedding_db,
        recent_turn_count=recent_turn_count,
        enabled_sources=enabled_sources,
        parallel_bm25_sources=parallel_bm25_sources,
        bm25_source_workers=bm25_source_workers,
    )
    feature_builder = LGBMFeatureBuilder(
        item_db=item_db,
        track_embedding_db=track_embedding_db,
        user_embedding_db=user_embedding_db,
        audio_embedding_db=audio_embedding_db,
        metadata_embedding_db=metadata_embedding_db,
        source_weights=wide_builder.source_weights,
        rrf_k=wide_builder.rrf_k,
    )
    return wide_builder, feature_builder, set(exact_retriever.track_ids)


def build_split_rows(
    *,
    config: Any,
    wide_builder: WideCandidateBuilder,
    feature_builder: LGBMFeatureBuilder,
    catalog_track_ids: set[str],
    split: str,
    candidate_topk: int,
    limit: int | None,
    batch_size: int,
    progress_every: int,
    require_positive: bool,
    force_add_positive: bool,
    sample_max_rows_per_group: int | None = None,
    sample_head_negatives: int = 0,
    sample_seed: int = 42,
    task_shard_index: int = 0,
    task_shard_count: int = 1,
    output_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """1 split 分の全タスクをバッチ処理して LTR 学習用の特徴量行を生成する。

    タスクをバッチ単位で処理し、各タスクの候補集合に対して特徴量行を構築する。
    output_path を指定した場合は RowWriter でストリーミング書き込みし、
    大量データでの OOM を回避する。

    Args:
        config: 候補生成 config。
        wide_builder: 候補集合を構築する WideCandidateBuilder。
        feature_builder: 特徴量行を構築する LGBMFeatureBuilder。
        catalog_track_ids: catalog に存在する全 track_id の集合（未知 ID 検出用）。
        split: 処理する split 名（"train" / "test" など）。
        candidate_topk: 各タスクで保持する候補の上限数。0 以下の場合は
            builder 側の no-cap union を許可し、特徴量行も全件処理する。
        limit: 処理するタスク数の上限（デバッグ用）。None で全件処理。
        batch_size: 一度に retrieval するタスク数。
        progress_every: 進捗をログ出力する間隔（タスク数）。
        require_positive: True の場合、正例が候補に含まれないグループをスキップする。
        force_add_positive: True の場合、正例が候補に含まれない場合に強制末尾追加する。
        sample_max_rows_per_group: group ごとの特徴量行数を事前に絞る上限。
            None の場合は sampling しない。
        sample_head_negatives: sampling 時に rank 上位から必ず残す負例数。
        sample_seed: tail negative sampling の seed。
        task_shard_index: 処理する shard index（0-based）。
        task_shard_count: task shard 数。1 の場合は shard しない。
        output_path: ストリーミング書き込み先パス。None の場合は全行を返す。

    Returns:
        (行リスト, サマリー辞書) のタプル。output_path 指定時は行リストは空。
    """
    all_tasks = prepare_tasks(config=config, split=split, limit=limit)
    if task_shard_count < 1:
        raise ValueError(f"task_shard_count must be >= 1, got {task_shard_count}")
    if not 0 <= task_shard_index < task_shard_count:
        raise ValueError(
            f"task_shard_index must be in [0, {task_shard_count}), got {task_shard_index}"
        )
    tasks = all_tasks[task_shard_index::task_shard_count]
    rows: list[dict[str, Any]] = []
    writer = RowWriter(output_path) if output_path else None
    row_count = 0
    unknown_candidate_count = 0
    skipped_no_positive = 0
    forced_positive_count = 0
    emitted_group_count = 0
    candidate_metric_state = new_candidate_metric_state()
    start = time.time()
    candidate_limit = int(candidate_topk) if int(candidate_topk) > 0 else None

    try:
        for batch_start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_start: batch_start + batch_size]
            histories = [task["history"] for task in batch_tasks]
            user_queries = [task["user_query"] for task in batch_tasks]
            user_ids = [task["user_id"] for task in batch_tasks]
            batch_candidates, batch_query_views, _ = wide_builder.batch_candidates_by_source(
                histories=histories,
                user_queries=user_queries,
                user_ids=user_ids,
            )
            batch_rows: list[dict[str, Any]] = []

            for offset, task in enumerate(batch_tasks):
                cbs = batch_candidates[offset]
                labels = task["labels"]
                union_candidates = wide_builder.union(cbs, topk=candidate_limit or 0)
                unknown_candidate_count += sum(1 for track_id in union_candidates if track_id not in catalog_track_ids)
                update_candidate_metric_state(
                    candidate_metric_state,
                    candidates=union_candidates,
                    labels=labels,
                )

                has_positive = bool(set(union_candidates) & set(labels))
                if labels and not has_positive:
                    if force_add_positive:
                        # LTR 学習でグループに正例がないと勾配が立たないため、上限を削って末尾に強制追加する
                        missing_labels = [
                            label for label in labels
                            if label in catalog_track_ids and label not in union_candidates
                        ]
                        if missing_labels:
                            keep = (
                                max(0, candidate_limit - len(missing_labels))
                                if candidate_limit is not None
                                else len(union_candidates)
                            )
                            union_candidates = union_candidates[:keep] + missing_labels
                            forced_positive_count += len(missing_labels)
                    elif require_positive:
                        skipped_no_positive += 1
                        continue

                group_id = f"{task['session_id']}:{task['turn_number']}"
                union_candidates = sample_union_candidates(
                    union_candidates,
                    labels,
                    max_rows_per_group=sample_max_rows_per_group,
                    head_negatives=sample_head_negatives,
                    seed=sample_seed,
                    group_id=group_id,
                )
                group_rows = feature_builder.build_rows(
                    candidates_by_source=cbs,
                    union_candidates=union_candidates,
                    query_views=batch_query_views[offset],
                    user_id=task["user_id"],
                    session_memory=task["history"],
                    session_id=task["session_id"],
                    turn_number=task["turn_number"],
                    labels=labels,
                    topk=candidate_limit,
                )
                for row in group_rows:
                    row["group_id"] = group_id
                    row["split"] = split
                batch_rows.extend(group_rows)
                emitted_group_count += 1

            if writer:
                writer.write(batch_rows)
            else:
                rows.extend(batch_rows)
            row_count += len(batch_rows)

            processed = min(batch_start + len(batch_tasks), len(tasks))
            if processed % max(progress_every, 1) < len(batch_tasks) or processed == len(tasks):
                elapsed = time.time() - start
                print(
                    f"{split}: processed_tasks={processed}/{len(tasks)} rows={row_count} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
    finally:
        if writer:
            writer.close()

    summary = {
        "split": split,
        "total_tasks": len(all_tasks),
        "tasks": len(tasks),
        "task_shard_index": task_shard_index,
        "task_shard_count": task_shard_count,
        "rows": row_count,
        "groups_before_filter": len(tasks),
        "groups_after_filter": emitted_group_count,
        "groups_dropped_no_positive": skipped_no_positive,
        f"{split}_groups_before_filter": len(tasks),
        f"{split}_groups_after_filter": emitted_group_count,
        f"{split}_groups_dropped_no_positive": skipped_no_positive,
        "candidate_topk": candidate_topk,
        "unknown_candidate_count": unknown_candidate_count,
        "skipped_no_positive": skipped_no_positive,
        "forced_positive_count": forced_positive_count,
        "sample_max_rows_per_group": sample_max_rows_per_group,
        "sample_head_negatives": sample_head_negatives,
        "sample_seed": sample_seed,
    }
    summary.update(finalize_candidate_metrics(candidate_metric_state))
    return rows, summary


def write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    """行リストをファイル形式に応じて parquet / csv / jsonlines で書き出す。

    Args:
        path: 出力ファイルパス。拡張子で書き込み形式を決定する。
        rows: 書き出す行の辞書リスト。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class RowWriter:
    """バッチごとに parquet / csv / jsonlines へストリーミング書き込みするライタ。

    大規模データセットをメモリに全展開せずに書き込むために使用する。
    初回 write() 呼び出し時にスキーマを確定し、以降のバッチではカラム追加を禁止する。
    """

    def __init__(self, path: str):
        """出力先パスを受け取り、既存ファイルがあれば削除して初期化する。

        Args:
            path: 出力ファイルパス。
        """
        self.path = path
        self.columns: list[str] | None = None
        self.parquet_writer: pq.ParquetWriter | None = None
        self.csv_header_written = False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)

    def write(self, rows: list[dict[str, Any]]) -> None:
        """行リストをファイルに追記書き込みする。

        Args:
            rows: 書き込む行の辞書リスト。空リストの場合は何もしない。

        Raises:
            ValueError: 初回以降に新しいカラムが現れた場合（スキーマ変化を検知）。
        """
        if not rows:
            return
        df = pd.DataFrame(rows)
        if self.columns is None:
            self.columns = list(df.columns)
        else:
            # ストリーミング中のカラム追加は parquet row group スキーマの不整合を招くため禁止する
            new_columns = sorted(set(df.columns) - set(self.columns))
            if new_columns:
                raise ValueError(f"Streaming schema changed for {self.path}: new columns={new_columns}")
            # 後バッチで欠損するカラムは pd.NA で補完してスキーマを維持する
            for col in self.columns:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[self.columns]

        if self.path.endswith(".parquet"):
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self.parquet_writer is None:
                self.parquet_writer = pq.ParquetWriter(self.path, table.schema)
            self.parquet_writer.write_table(table)
        elif self.path.endswith(".csv"):
            df.to_csv(self.path, index=False, mode="a", header=not self.csv_header_written)
            self.csv_header_written = True
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """parquet ライタをフラッシュして閉じる。"""
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None


def main(args: argparse.Namespace) -> None:
    """train / dev 両 split の LTR データセットを構築してファイルに書き出す。

    Args:
        args: CLI 引数。candidate_config、出力先、candidate_topk、shard 設定、
            embedding feature の有無などを含む。
    """
    config = load_config(args.candidate_config)
    enabled_sources = parse_sources(args.enabled_sources)
    print(f"enabled_sources={enabled_sources}", flush=True)
    wide_builder, feature_builder, catalog_track_ids = init_builders(
        config,
        enabled_sources=enabled_sources,
        include_embedding_features=args.include_embedding_features,
        metadata_embedding_name=args.metadata_embedding_name,
        recent_turn_count=args.recent_turn_count,
        bm25_n_threads=args.bm25_n_threads,
        bm25_chunksize=args.bm25_chunksize,
        parallel_bm25_sources=args.parallel_bm25_sources,
        bm25_source_workers=args.bm25_source_workers,
    )

    train_rows, train_summary = build_split_rows(
        config=config,
        wide_builder=wide_builder,
        feature_builder=feature_builder,
        catalog_track_ids=catalog_track_ids,
        split=args.train_split,
        candidate_topk=args.candidate_topk,
        limit=args.train_limit,
        batch_size=args.retrieval_batch_size,
        progress_every=args.progress_every,
        require_positive=args.require_positive,
        force_add_positive=args.force_add_positive,
        sample_max_rows_per_group=args.train_sample_max_rows_per_group,
        sample_head_negatives=args.train_sample_head_negatives,
        sample_seed=args.train_sample_seed,
        task_shard_index=args.task_shard_index,
        task_shard_count=args.task_shard_count,
        output_path=args.train_output,
    )
    # dev split は評価用のためグループスキップや正例強制追加を行わない
    dev_rows, dev_summary = build_split_rows(
        config=config,
        wide_builder=wide_builder,
        feature_builder=feature_builder,
        catalog_track_ids=catalog_track_ids,
        split=args.dev_split,
        candidate_topk=args.candidate_topk,
        limit=args.dev_limit,
        batch_size=args.retrieval_batch_size,
        progress_every=args.progress_every,
        require_positive=False,
        force_add_positive=False,
        sample_max_rows_per_group=None,
        sample_head_negatives=0,
        sample_seed=args.train_sample_seed,
        task_shard_index=args.task_shard_index,
        task_shard_count=args.task_shard_count,
        output_path=args.dev_output,
    )

    if train_rows:
        write_rows(args.train_output, train_rows)
    if dev_rows:
        write_rows(args.dev_output, dev_rows)
    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_config": args.candidate_config,
                    "enabled_sources": enabled_sources,
                    "include_embedding_features": args.include_embedding_features,
                    "retrieval_batch_size": args.retrieval_batch_size,
                    "bm25_n_threads": args.bm25_n_threads,
                    "bm25_chunksize": args.bm25_chunksize,
                    "parallel_bm25_sources": args.parallel_bm25_sources,
                    "bm25_source_workers": args.bm25_source_workers,
                    "train": train_summary,
                    "dev": dev_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


def build_parser() -> argparse.ArgumentParser:
    """build_dataset CLI の argparse parser を構築する。

    旧 exp014 パスの re-export shim からも同じ parser を呼べるように関数化している
    （shim の ``__main__`` は runpy を使わず ``build_parser`` 経由で本体 CLI を実行する）。

    Returns:
        build_dataset の CLI 引数を定義した parser。
    """
    parser = argparse.ArgumentParser(description="Build exp014 LightGBM LTR datasets.")
    parser.add_argument("--candidate_config", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--dev_output", required=True)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--enabled_sources", default=",".join(PRIMARY_TEXT5_SOURCES))
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--dev_split", default="test")
    parser.add_argument("--candidate_topk", type=int, default=500)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)
    parser.add_argument("--retrieval_batch_size", type=int, default=256)
    parser.add_argument("--bm25_n_threads", type=int, default=0)
    parser.add_argument("--bm25_chunksize", type=int, default=50)
    parser.add_argument("--parallel_bm25_sources", action="store_true")
    parser.add_argument("--bm25_source_workers", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--recent_turn_count", type=int, default=2)
    parser.add_argument("--require_positive", action="store_true")
    parser.add_argument("--force_add_positive", action="store_true")
    parser.add_argument("--train_sample_max_rows_per_group", type=int, default=None)
    parser.add_argument("--train_sample_head_negatives", type=int, default=0)
    parser.add_argument("--train_sample_seed", type=int, default=42)
    parser.add_argument("--task_shard_index", type=int, default=0)
    parser.add_argument("--task_shard_count", type=int, default=1)
    parser.add_argument("--include_embedding_features", action="store_true")
    parser.add_argument("--metadata_embedding_name", default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
