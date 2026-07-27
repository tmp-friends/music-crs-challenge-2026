"""prediction source rank feature で LightGBM reranker を学習する。

exp021 の continuity / QNM source は standalone でも強いが、RRF の手作業
探索だけでは source 間の非線形な相性を扱いにくい。この CLI は既存
prediction JSON を読み込み、primary と各 source における rank だけを特徴量化して
session-turn 単位の LightGBM ranker を学習する。

Dev full-fit は上限確認、session-fold OOF は汎化寄りの確認として併記する。
Blind A へ適用するときは Dev full-fit model を使い、`protect_top` で primary の
prefix を固定できる。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from mcrs.ltr.data_utils import prepare_tasks


Key = tuple[str, int]
PredictionMap = dict[Key, tuple[str, ...]]


@dataclass(frozen=True)
class SourceSpec:
    """rank source の名前と prediction JSON path。"""

    name: str
    path: Path


@dataclass
class FeatureBundle:
    """LightGBM へ渡す特徴量と row metadata。"""

    x: np.ndarray
    y: np.ndarray | None
    groups: list[int]
    row_keys: list[Key]
    row_tracks: list[str]
    candidate_positive_count: int


def parse_named_path(spec: str) -> SourceSpec:
    """`name=path` 形式の CLI 引数を SourceSpec に変換する。

    Args:
        spec: source 名と path を `=` で連結した文字列。

    Returns:
        SourceSpec。

    Raises:
        ValueError: `=` が無い場合。
    """
    if "=" not in spec:
        raise ValueError(f"source must be name=path: {spec}")
    name, path = spec.split("=", 1)
    return SourceSpec(name=name, path=Path(path))


def load_records(path: Path, *, topk: int, allow_short: bool = False) -> list[dict[str, Any]]:
    """prediction JSON を読み込み、track list を指定 prefix に切る。

    Args:
        path: prediction JSON path。
        topk: 読み込む推薦件数。
        allow_short: True の場合、行ごとの track 数が topk 未満（空含む）でも許容する。
            sparse な候補 source（例: cross-session exact-track。発火しない行は 0 件）を
            popularity padding なしで渡せるようにするための opt-in。既定 False で従来の
            「全行ちょうど topk」チェックを維持（既存の dense source の取り違え検知）。

    Returns:
        prediction record の list。
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"prediction root must be list: {path}")
    output: list[dict[str, Any]] = []
    for record in records:
        track_ids = tuple(str(track_id) for track_id in record["predicted_track_ids"][:topk])
        if not allow_short and len(track_ids) != topk:
            raise ValueError(
                f"{path} has only {len(track_ids)} track ids for "
                f"{record.get('session_id')} turn={record.get('turn_number')}"
            )
        copied = dict(record)
        copied["predicted_track_ids"] = list(track_ids)
        output.append(copied)
    return output


def records_to_map(records: list[dict[str, Any]]) -> PredictionMap:
    """prediction records を `(session_id, turn_number)` key の map へ変換する。"""
    predictions: PredictionMap = {}
    for record in records:
        key = (str(record["session_id"]), int(record["turn_number"]))
        if key in predictions:
            raise ValueError(f"duplicate prediction key: {key}")
        predictions[key] = tuple(str(track_id) for track_id in record["predicted_track_ids"])
    return predictions


def records_by_key(records: list[dict[str, Any]]) -> dict[Key, dict[str, Any]]:
    """prediction records を key -> record へ変換する。"""
    output: dict[Key, dict[str, Any]] = {}
    for record in records:
        key = (str(record["session_id"]), int(record["turn_number"]))
        output[key] = record
    return output


def ordered_keys(records: list[dict[str, Any]]) -> list[Key]:
    """records の出現順で key を返す。"""
    return [(str(record["session_id"]), int(record["turn_number"])) for record in records]


def load_dev_labels(dataset_name: str, split: str) -> tuple[list[Key], dict[Key, str]]:
    """TalkPlayData Dev split から session-turn key と正解 track を読む。

    Args:
        dataset_name: Hugging Face dataset 名。
        split: dataset split。

    Returns:
        key の評価順と key -> 正解 track_id。
    """
    config = type("Config", (), {"test_dataset_name": dataset_name})()
    tasks = prepare_tasks(config, split=split, limit=None)
    keys: list[Key] = []
    labels: dict[Key, str] = {}
    for task in tasks:
        if len(task["labels"]) != 1:
            raise ValueError(
                f"Dev task must have exactly one label: "
                f"{task['session_id']} turn={task['turn_number']}"
            )
        key = (str(task["session_id"]), int(task["turn_number"]))
        keys.append(key)
        labels[key] = str(task["labels"][0])
    return keys, labels


def feature_names(source_names: list[str]) -> list[str]:
    """source 数に応じた feature name を返す。"""
    names = [
        "primary_inv_rank",
        "primary_is_top1",
        "primary_in_top5",
        "primary_in_top10",
        "primary_in_top20",
    ]
    for source_name in source_names:
        names.extend(
            [
                f"{source_name}_inv_rank",
                f"{source_name}_in_top5",
                f"{source_name}_in_top20",
                f"{source_name}_rank_clipped",
            ]
        )
    names.extend(
        [
            "source_hit_rate",
            "source_best_inv_rank",
            "source_rrf_sum_k60",
            "turn_number",
            "is_early_turn",
            "is_followup_turn",
        ]
    )
    return names


def build_features(
    *,
    keys: list[Key],
    primary: PredictionMap,
    sources: dict[str, PredictionMap],
    source_names: list[str],
    labels: dict[Key, str] | None,
    source_topk: int,
) -> FeatureBundle:
    """rank source 群から LightGBM 用の行特徴量を作る。

    Args:
        keys: 出力 group の順序。
        primary: anchor prediction。
        sources: source 名 -> prediction map。
        source_names: feature 順を固定する source 名。
        labels: key -> 正解 track_id。Blind A では None。
        source_topk: 各 source から union に入れる prefix 長。

    Returns:
        FeatureBundle。
    """
    x_rows: list[list[float]] = []
    y_rows: list[int] | None = [] if labels is not None else None
    groups: list[int] = []
    row_keys: list[Key] = []
    row_tracks: list[str] = []
    candidate_positive_count = 0

    for key in keys:
        if key not in primary:
            raise ValueError(f"primary missing key: {key}")
        seen: set[str] = set()
        union: list[str] = []
        for track_id in primary[key]:
            if track_id not in seen:
                union.append(track_id)
                seen.add(track_id)
        for source_name in source_names:
            source = sources[source_name]
            if key not in source:
                raise ValueError(f"source {source_name} missing key: {key}")
            for track_id in source[key][:source_topk]:
                if track_id not in seen:
                    union.append(track_id)
                    seen.add(track_id)

        primary_rank = {track_id: rank + 1 for rank, track_id in enumerate(primary[key])}
        source_ranks = {
            source_name: {
                track_id: rank + 1
                for rank, track_id in enumerate(sources[source_name][key][:source_topk])
            }
            for source_name in source_names
        }
        label = labels.get(key) if labels is not None else None
        if label is not None and label in seen:
            candidate_positive_count += 1

        for track_id in union:
            rank = primary_rank.get(track_id, 9999)
            features = [
                1.0 / rank if rank < 9999 else 0.0,
                1.0 if rank == 1 else 0.0,
                1.0 if rank <= 5 else 0.0,
                1.0 if rank <= 10 else 0.0,
                1.0 if rank <= 20 else 0.0,
            ]
            hit_count = 0
            best_rank = 9999
            rrf_sum = 0.0
            for source_name in source_names:
                source_rank = source_ranks[source_name].get(track_id, 9999)
                if source_rank < 9999:
                    hit_count += 1
                    best_rank = min(best_rank, source_rank)
                    rrf_sum += 1.0 / (60.0 + source_rank)
                features.extend(
                    [
                        1.0 / source_rank if source_rank < 9999 else 0.0,
                        1.0 if source_rank <= 5 else 0.0,
                        1.0 if source_rank <= 20 else 0.0,
                        float(min(source_rank, source_topk + 1)),
                    ]
                )
            features.extend(
                [
                    hit_count / max(len(source_names), 1),
                    1.0 / best_rank if best_rank < 9999 else 0.0,
                    rrf_sum,
                    float(key[1]),
                    1.0 if key[1] <= 2 else 0.0,
                    1.0 if key[1] >= 3 else 0.0,
                ]
            )
            x_rows.append(features)
            if y_rows is not None:
                y_rows.append(1 if track_id == label else 0)
            row_keys.append(key)
            row_tracks.append(track_id)
        groups.append(len(union))

    return FeatureBundle(
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.int8) if y_rows is not None else None,
        groups=groups,
        row_keys=row_keys,
        row_tracks=row_tracks,
        candidate_positive_count=candidate_positive_count,
    )


def ndcg(predictions: PredictionMap, keys: list[Key], labels: dict[Key, str]) -> float:
    """prediction map の all-task nDCG@20 を計算する。"""
    total = 0.0
    for key in keys:
        label = labels[key]
        try:
            rank = predictions[key].index(label) + 1
        except ValueError:
            continue
        if rank <= 20:
            total += 1.0 / math.log2(rank + 1.0)
    return total / len(keys)


def ndcg_indices(
    predictions: PredictionMap,
    keys: list[Key],
    labels: dict[Key, str],
    indices: np.ndarray,
) -> float:
    """指定 index subset の all-task nDCG@20 を計算する。"""
    subset = [keys[int(index)] for index in indices.tolist()]
    return ndcg(predictions, subset, labels)


def score_to_predictions(
    bundle: FeatureBundle,
    keys: list[Key],
    primary: PredictionMap,
    *,
    scores: np.ndarray,
    protect_top: int,
    topk: int,
) -> PredictionMap:
    """row score を group ごとの ranked prediction に戻す。

    Args:
        bundle: build_features が返した row metadata。
        keys: 出力 group の順序。
        primary: prefix 保護と tie-break に使う primary prediction。
        scores: LightGBM score。
        protect_top: primary prefix を固定する件数。
        topk: 出力件数。

    Returns:
        key -> ranked track_id tuple。
    """
    predictions: PredictionMap = {}
    cursor = 0
    for key in keys:
        rows: list[tuple[str, float]] = []
        while cursor < len(bundle.row_keys) and bundle.row_keys[cursor] == key:
            rows.append((bundle.row_tracks[cursor], float(scores[cursor])))
            cursor += 1
        protected = list(primary[key][:protect_top])
        protected_set = set(protected)
        primary_rank = {track_id: rank for rank, track_id in enumerate(primary[key])}
        free_rows = [(track_id, score) for track_id, score in rows if track_id not in protected_set]
        free_rows.sort(key=lambda item: (-item[1], primary_rank.get(item[0], 9999), item[0]))
        predictions[key] = tuple((protected + [track_id for track_id, _ in free_rows])[:topk])
    return predictions


def change_stats(before: PredictionMap, after: PredictionMap, keys: list[Key]) -> dict[str, float | int]:
    """prediction 前後の変更量を集計する。"""
    changed = 0
    top1_changed = 0
    overlaps: list[int] = []
    for key in keys:
        if before[key] != after[key]:
            changed += 1
        if before[key][:1] != after[key][:1]:
            top1_changed += 1
        overlaps.append(len(set(before[key]) & set(after[key])))
    return {
        "changed": changed,
        "top1_changed": top1_changed,
        "avg_overlap20": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def build_records(
    predictions: PredictionMap,
    template_records: dict[Key, dict[str, Any]],
    keys: list[Key],
) -> list[dict[str, Any]]:
    """prediction map を response 空の tracks JSON records へ変換する。"""
    return [
        {
            "session_id": str(template_records[key]["session_id"]),
            "user_id": str(template_records[key].get("user_id", "")),
            "turn_number": int(template_records[key]["turn_number"]),
            "predicted_track_ids": list(predictions[key]),
            "predicted_response": "",
        }
        for key in keys
    ]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """records を pretty JSON で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_model(
    bundle: FeatureBundle,
    *,
    params: dict[str, Any],
    num_boost_round: int,
    feature_name: list[str],
) -> lgb.Booster:
    """FeatureBundle から LightGBM ranker を学習する。"""
    if bundle.y is None:
        raise ValueError("training bundle requires labels")
    dataset = lgb.Dataset(
        bundle.x,
        label=bundle.y,
        group=bundle.groups,
        feature_name=feature_name,
    )
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def split_indices_by_session(keys: list[Key], *, folds: int, seed: int) -> list[np.ndarray]:
    """session_id 単位で fold index を作る。"""
    sessions = sorted({session_id for session_id, _ in keys})
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    session_to_fold = {session_id: index % folds for index, session_id in enumerate(sessions)}
    buckets = [[] for _ in range(folds)]
    for index, (session_id, _) in enumerate(keys):
        buckets[session_to_fold[session_id]].append(index)
    return [np.asarray(bucket, dtype=np.int32) for bucket in buckets]


def run_oof(
    *,
    keys: list[Key],
    labels: dict[Key, str],
    primary: PredictionMap,
    sources: dict[str, PredictionMap],
    source_names: list[str],
    source_topk: int,
    protect_top: int,
    topk: int,
    params: dict[str, Any],
    feature_name: list[str],
    folds: int,
    seed: int,
    num_boost_round: int,
) -> tuple[PredictionMap, list[dict[str, Any]]]:
    """session-fold OOF prediction と fold metric を作る。"""
    fold_indices = split_indices_by_session(keys, folds=folds, seed=seed)
    all_indices = set(range(len(keys)))
    oof_predictions: PredictionMap = {}
    fold_rows: list[dict[str, Any]] = []
    for fold_index, valid_indices in enumerate(fold_indices):
        valid_set = set(int(index) for index in valid_indices.tolist())
        train_indices = sorted(all_indices - valid_set)
        train_keys = [keys[index] for index in train_indices]
        valid_keys = [keys[int(index)] for index in valid_indices.tolist()]
        train_bundle = build_features(
            keys=train_keys,
            primary=primary,
            sources=sources,
            source_names=source_names,
            labels=labels,
            source_topk=source_topk,
        )
        valid_bundle = build_features(
            keys=valid_keys,
            primary=primary,
            sources=sources,
            source_names=source_names,
            labels=labels,
            source_topk=source_topk,
        )
        model = train_model(
            train_bundle,
            params=params,
            num_boost_round=num_boost_round,
            feature_name=feature_name,
        )
        scores = model.predict(valid_bundle.x)
        fold_predictions = score_to_predictions(
            valid_bundle,
            valid_keys,
            primary,
            scores=scores,
            protect_top=protect_top,
            topk=topk,
        )
        oof_predictions.update(fold_predictions)
        fold_rows.append(
            {
                "fold": fold_index,
                "tasks": len(valid_keys),
                "base_ndcg": ndcg_indices(primary, keys, labels, valid_indices),
                "oof_ndcg": ndcg(fold_predictions, valid_keys, labels),
                "change_stats": change_stats(primary, fold_predictions, valid_keys),
            }
        )
    return oof_predictions, fold_rows


def main() -> None:
    """CLI entrypoint。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, help="name=dev_prediction.json")
    parser.add_argument("--source", action="append", required=True, help="name=dev_prediction.json")
    parser.add_argument("--blind_primary", help="name=blind_prediction.json")
    parser.add_argument("--blind_source", action="append", default=[], help="name=blind_prediction.json")
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--source_topk", type=int, default=100)
    parser.add_argument("--protect_top", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--num_boost_round", type=int, default=300)
    parser.add_argument("--oof_num_boost_round", type=int, default=160)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_data_in_leaf", type=int, default=80)
    parser.add_argument("--lambda_l2", type=float, default=5.0)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--output_dev_full", type=Path, required=True)
    parser.add_argument("--output_dev_oof", type=Path, required=True)
    parser.add_argument("--output_blind_tracks", type=Path)
    parser.add_argument("--model_output", type=Path)
    args = parser.parse_args()

    primary_spec = parse_named_path(args.primary)
    source_specs = [parse_named_path(spec) for spec in args.source]
    if len({source.name for source in source_specs}) != len(source_specs):
        raise ValueError("source names must be unique")
    source_names = [source.name for source in source_specs]
    names = feature_names(source_names)
    params: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "learning_rate": 0.03,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": args.lambda_l2,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": 8,
        "deterministic": True,
    }

    dev_keys, labels = load_dev_labels(args.dataset_name, args.split)
    primary_records = load_records(primary_spec.path, topk=args.topk)
    primary = records_to_map(primary_records)
    primary_records_by_key = records_by_key(primary_records)
    sources = {
        source.name: records_to_map(load_records(source.path, topk=args.source_topk))
        for source in source_specs
    }
    dev_bundle = build_features(
        keys=dev_keys,
        primary=primary,
        sources=sources,
        source_names=source_names,
        labels=labels,
        source_topk=args.source_topk,
    )
    full_model = train_model(
        dev_bundle,
        params=params,
        num_boost_round=args.num_boost_round,
        feature_name=names,
    )
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        full_model.save_model(str(args.model_output))
    full_scores = full_model.predict(dev_bundle.x)
    full_predictions = score_to_predictions(
        dev_bundle,
        dev_keys,
        primary,
        scores=full_scores,
        protect_top=args.protect_top,
        topk=args.topk,
    )
    oof_predictions, fold_rows = run_oof(
        keys=dev_keys,
        labels=labels,
        primary=primary,
        sources=sources,
        source_names=source_names,
        source_topk=args.source_topk,
        protect_top=args.protect_top,
        topk=args.topk,
        params=params,
        feature_name=names,
        folds=args.folds,
        seed=args.seed,
        num_boost_round=args.oof_num_boost_round,
    )

    write_records(args.output_dev_full, build_records(full_predictions, primary_records_by_key, dev_keys))
    write_records(args.output_dev_oof, build_records(oof_predictions, primary_records_by_key, dev_keys))

    base_ndcg = ndcg(primary, dev_keys, labels)
    full_ndcg = ndcg(full_predictions, dev_keys, labels)
    oof_ndcg = ndcg(oof_predictions, dev_keys, labels)
    report: dict[str, Any] = {
        "primary": primary_spec.name,
        "primary_path": str(primary_spec.path),
        "source_paths": {source.name: str(source.path) for source in source_specs},
        "source_topk": args.source_topk,
        "topk": args.topk,
        "protect_top": args.protect_top,
        "feature_names": names,
        "params": params,
        "num_boost_round": args.num_boost_round,
        "oof_num_boost_round": args.oof_num_boost_round,
        "candidate_rows": int(dev_bundle.x.shape[0]),
        "candidate_positive_count": int(dev_bundle.candidate_positive_count),
        "base_ndcg": base_ndcg,
        "full_fit_ndcg": full_ndcg,
        "full_fit_delta": full_ndcg - base_ndcg,
        "oof_ndcg": oof_ndcg,
        "oof_delta": oof_ndcg - base_ndcg,
        "full_change_stats_vs_primary": change_stats(primary, full_predictions, dev_keys),
        "oof_change_stats_vs_primary": change_stats(primary, oof_predictions, dev_keys),
        "folds": fold_rows,
        "output_dev_full": str(args.output_dev_full),
        "output_dev_oof": str(args.output_dev_oof),
        "model_output": str(args.model_output) if args.model_output else None,
    }

    if args.blind_primary:
        blind_primary_spec = parse_named_path(args.blind_primary)
        blind_source_specs = [parse_named_path(spec) for spec in args.blind_source]
        if {source.name for source in blind_source_specs} != set(source_names):
            raise ValueError("blind source names must match dev source names")
        blind_records = load_records(blind_primary_spec.path, topk=args.topk)
        blind_primary = records_to_map(blind_records)
        blind_keys = ordered_keys(blind_records)
        blind_records_by_key = records_by_key(blind_records)
        blind_sources = {
            source.name: records_to_map(load_records(source.path, topk=args.source_topk))
            for source in blind_source_specs
        }
        blind_bundle = build_features(
            keys=blind_keys,
            primary=blind_primary,
            sources=blind_sources,
            source_names=source_names,
            labels=None,
            source_topk=args.source_topk,
        )
        blind_scores = full_model.predict(blind_bundle.x)
        blind_predictions = score_to_predictions(
            blind_bundle,
            blind_keys,
            blind_primary,
            scores=blind_scores,
            protect_top=args.protect_top,
            topk=args.topk,
        )
        if args.output_blind_tracks is None:
            raise ValueError("--blind_primary を使う場合は --output_blind_tracks が必要です")
        write_records(
            args.output_blind_tracks,
            build_records(blind_predictions, blind_records_by_key, blind_keys),
        )
        report["blind_primary"] = blind_primary_spec.name
        report["blind_primary_path"] = str(blind_primary_spec.path)
        report["blind_source_paths"] = {source.name: str(source.path) for source in blind_source_specs}
        report["blind_change_stats_vs_primary"] = change_stats(
            blind_primary,
            blind_predictions,
            blind_keys,
        )
        report["output_blind_tracks"] = str(args.output_blind_tracks)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "base_ndcg": base_ndcg,
                "full_fit_ndcg": full_ndcg,
                "oof_ndcg": oof_ndcg,
                "output_report": str(args.output_report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
