"""exp015 の shard 分割された LightGBM LTR dataset artifact を結合する。"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sorted_glob(pattern: str) -> list[str]:
    """glob pattern に一致するファイルをソートして返す。

    Args:
        pattern: 入力ファイルを指定する glob pattern。

    Returns:
        ソート済みファイルパスのリスト。

    Raises:
        FileNotFoundError: pattern に一致するファイルが 1 件もない場合。
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return paths


def merge_parquet(inputs: list[str], output: str) -> None:
    """複数 parquet shard を row group 単位で 1 つの parquet に結合する。

    Args:
        inputs: 結合対象 parquet ファイルのパスリスト。
        output: 結合後 parquet の出力先。
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for input_path in inputs:
            parquet_file = pq.ParquetFile(input_path)
            for batch in parquet_file.iter_batches():
                table = pa.Table.from_batches([batch])
                if writer is None:
                    # 初回 shard の schema を採用し、以降の shard は同一 schema として追記する。
                    writer = pq.ParquetWriter(output, table.schema)
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def combine_split_summaries(summaries: list[dict[str, Any]], split: str) -> dict[str, Any]:
    """各 shard の split summary を合算する。

    Args:
        summaries: shard ごとの summary JSON 内容。
        split: ``train`` または ``dev``。

    Returns:
        合算済み split summary。
    """
    split_summaries = [summary[split] for summary in summaries]
    first = split_summaries[0]
    candidate_metric_count = sum(
        item.get("candidate_metric_task_count", item.get("tasks", 0))
        for item in split_summaries
    )
    combined = {
        "split": first["split"],
        "total_tasks": first.get("total_tasks", sum(item["tasks"] for item in split_summaries)),
        "tasks": sum(item["tasks"] for item in split_summaries),
        "task_shard_count": len(split_summaries),
        "rows": sum(item["rows"] for item in split_summaries),
        "groups_before_filter": sum(
            item.get("groups_before_filter", item.get("tasks", 0))
            for item in split_summaries
        ),
        "groups_after_filter": sum(
            item.get("groups_after_filter", item.get("tasks", 0) - item.get("skipped_no_positive", 0))
            for item in split_summaries
        ),
        "groups_dropped_no_positive": sum(
            item.get("groups_dropped_no_positive", item.get("skipped_no_positive", 0))
            for item in split_summaries
        ),
        "candidate_topk": first["candidate_topk"],
        "unknown_candidate_count": sum(item["unknown_candidate_count"] for item in split_summaries),
        "skipped_no_positive": sum(item["skipped_no_positive"] for item in split_summaries),
        "forced_positive_count": sum(item["forced_positive_count"] for item in split_summaries),
        "candidate_metric_task_count": candidate_metric_count,
        "candidate_hit_groups": sum(
            item.get("candidate_hit_groups", 0)
            for item in split_summaries
        ),
    }
    combined[f"{split}_groups_before_filter"] = combined["groups_before_filter"]
    combined[f"{split}_groups_after_filter"] = combined["groups_after_filter"]
    combined[f"{split}_groups_dropped_no_positive"] = combined["groups_dropped_no_positive"]

    for metric_name in [
        "candidate_recall@20",
        "candidate_recall@100",
        "candidate_recall@500",
        "oracle_ndcg@20",
    ]:
        weighted_sum = sum(
            float(item.get(metric_name, 0.0))
            * float(item.get("candidate_metric_task_count", item.get("tasks", 0)))
            for item in split_summaries
        )
        combined[metric_name] = (
            weighted_sum / float(candidate_metric_count)
            if candidate_metric_count else 0.0
        )
    return combined


def main(args: argparse.Namespace) -> None:
    """train/dev parquet と summary JSON の shard を結合する。

    Args:
        args: CLI 引数。train/dev/summary の glob pattern と結合後出力先を含む。
    """
    train_inputs = sorted_glob(args.train_glob)
    dev_inputs = sorted_glob(args.dev_glob)
    summary_inputs = sorted_glob(args.summary_glob)

    summaries = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in summary_inputs
    ]
    merge_parquet(train_inputs, args.train_output)
    merge_parquet(dev_inputs, args.dev_output)

    first = summaries[0]
    merged_summary = {
        "candidate_config": first["candidate_config"],
        "enabled_sources": first["enabled_sources"],
        "include_embedding_features": first["include_embedding_features"],
        "retrieval_batch_size": first["retrieval_batch_size"],
        "bm25_n_threads": first.get("bm25_n_threads", 0),
        "bm25_chunksize": first.get("bm25_chunksize", 50),
        "parallel_bm25_sources": first.get("parallel_bm25_sources", False),
        "bm25_source_workers": first.get("bm25_source_workers", 4),
        "task_shard_count": len(summaries),
        "train": combine_split_summaries(summaries, "train"),
        "dev": combine_split_summaries(summaries, "dev"),
        "shard_summaries": summaries,
    }

    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge exp015 sharded dataset artifacts.")
    parser.add_argument("--train_glob", required=True)
    parser.add_argument("--dev_glob", required=True)
    parser.add_argument("--summary_glob", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--dev_output", required=True)
    parser.add_argument("--summary_output", required=True)
    main(parser.parse_args())
