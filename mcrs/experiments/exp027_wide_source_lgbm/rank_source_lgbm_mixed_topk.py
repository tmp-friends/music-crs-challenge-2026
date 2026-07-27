"""source ごとに topK を変えられる LightGBM rank-source reranker。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from mcrs.experiments.exp022_continuity_lgbm.rank_source_lgbm import (
    FeatureBundle,
    Key,
    PredictionMap,
    build_records,
    change_stats,
    load_dev_labels,
    load_records,
    ndcg,
    ndcg_indices,
    records_by_key,
    records_to_map,
    score_to_predictions,
    train_model,
    write_records,
)


@dataclass(frozen=True)
class MixedSourceSpec:
    """rank source の名前、使用 topK、prediction JSON path。"""

    name: str
    topk: int
    path: Path


def parse_mixed_source(spec: str, *, default_topk: int) -> MixedSourceSpec:
    """`name[:topk]=path` 形式の CLI 引数を解釈する。

    Args:
        spec: source 指定。topK 省略時は ``default_topk`` を使う。
        default_topk: topK 省略時の prefix 長。

    Returns:
        MixedSourceSpec。

    Raises:
        ValueError: `=` が無い、または topK が正でない場合。
    """
    if "=" not in spec:
        raise ValueError(f"source must be name[:topk]=path: {spec}")
    name_part, path = spec.split("=", 1)
    if ":" in name_part:
        name, topk_text = name_part.rsplit(":", 1)
        topk = int(topk_text)
    else:
        name = name_part
        topk = default_topk
    if not name:
        raise ValueError(f"source name must not be empty: {spec}")
    if topk <= 0:
        raise ValueError(f"source topk must be positive: {spec}")
    return MixedSourceSpec(name=name, topk=topk, path=Path(path))


def feature_names(source_specs: list[MixedSourceSpec]) -> list[str]:
    """source 数に応じた feature name を返す。

    Args:
        source_specs: feature 順を固定する source 指定。

    Returns:
        LightGBM feature name の list。
    """
    names = [
        "primary_inv_rank",
        "primary_is_top1",
        "primary_in_top5",
        "primary_in_top10",
        "primary_in_top20",
    ]
    for source in source_specs:
        names.extend(
            [
                f"{source.name}_inv_rank",
                f"{source.name}_in_top5",
                f"{source.name}_in_top20",
                f"{source.name}_rank_clipped",
                f"{source.name}_rank_pct",
            ]
        )
    names.extend(
        [
            "source_hit_rate",
            "source_hit_count",
            "source_best_inv_rank",
            "source_rrf_sum_k60",
            "source_rrf_sum_k200",
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
    source_specs: list[MixedSourceSpec],
    labels: dict[Key, str] | None,
    exclude_map: dict[Key, set[str]] | None = None,
) -> FeatureBundle:
    """source ごとの topK に従って LightGBM 用特徴量を作る。

    Args:
        keys: 出力 group の順序。
        primary: anchor prediction。
        sources: source 名 -> prediction map。
        source_specs: source 名、topK、path の指定。
        labels: key -> 正解 track_id。Blind A では None。
        exclude_map: key -> 候補 union から除外する track_id 集合（None で従来挙動）。
            within-session 既出 music 等の guaranteed-wrong を候補段で抜くために使う。
            正解 track は防御的に除外しない（除外対象は guaranteed-non-GT を想定）。

    Returns:
        LightGBM 入力と row metadata。
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
        label = labels.get(key) if labels is not None else None
        # 候補 union から外す track（within-session 既出等）。正解は防御的に保持。
        exclude = exclude_map.get(key, frozenset()) if exclude_map else frozenset()
        seen: set[str] = set()
        union: list[str] = []
        for track_id in primary[key]:
            if track_id not in seen and (track_id not in exclude or track_id == label):
                union.append(track_id)
                seen.add(track_id)
        for source in source_specs:
            source_predictions = sources[source.name]
            if key not in source_predictions:
                raise ValueError(f"source {source.name} missing key: {key}")
            for track_id in source_predictions[key][: source.topk]:
                if track_id not in seen and (track_id not in exclude or track_id == label):
                    union.append(track_id)
                    seen.add(track_id)

        # source_ranks は元の source 全体から計算（除外 track は union に無いので参照されない＝
        # 残存 track の rank feature は除外有無で不変）。
        primary_rank = {track_id: rank + 1 for rank, track_id in enumerate(primary[key])}
        source_ranks = {
            source.name: {
                track_id: rank + 1
                for rank, track_id in enumerate(sources[source.name][key][: source.topk])
            }
            for source in source_specs
        }
        if label is not None and label in seen:
            candidate_positive_count += 1

        for track_id in union:
            primary_track_rank = primary_rank.get(track_id, 9999)
            features = [
                1.0 / primary_track_rank if primary_track_rank < 9999 else 0.0,
                1.0 if primary_track_rank == 1 else 0.0,
                1.0 if primary_track_rank <= 5 else 0.0,
                1.0 if primary_track_rank <= 10 else 0.0,
                1.0 if primary_track_rank <= 20 else 0.0,
            ]
            hit_count = 0
            best_rank = 9999
            rrf_sum_k60 = 0.0
            rrf_sum_k200 = 0.0
            for source in source_specs:
                source_rank = source_ranks[source.name].get(track_id, 9999)
                if source_rank < 9999:
                    hit_count += 1
                    best_rank = min(best_rank, source_rank)
                    rrf_sum_k60 += 1.0 / (60.0 + source_rank)
                    rrf_sum_k200 += 1.0 / (200.0 + source_rank)
                clipped_rank = min(source_rank, source.topk + 1)
                rank_pct = clipped_rank / float(source.topk + 1)
                features.extend(
                    [
                        1.0 / source_rank if source_rank < 9999 else 0.0,
                        1.0 if source_rank <= 5 else 0.0,
                        1.0 if source_rank <= 20 else 0.0,
                        float(clipped_rank),
                        rank_pct,
                    ]
                )
            features.extend(
                [
                    hit_count / max(len(source_specs), 1),
                    float(hit_count),
                    1.0 / best_rank if best_rank < 9999 else 0.0,
                    rrf_sum_k60,
                    rrf_sum_k200,
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


def split_indices_by_session(keys: list[Key], *, folds: int, seed: int) -> list[np.ndarray]:
    """session_id 単位で fold index を作る。

    Args:
        keys: Dev task key の list。
        folds: fold 数。
        seed: session shuffle seed。

    Returns:
        fold ごとの row index 配列。
    """
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
    source_specs: list[MixedSourceSpec],
    protect_top: int,
    topk: int,
    params: dict[str, Any],
    feature_name: list[str],
    folds: int,
    seed: int,
    num_boost_round: int,
) -> tuple[PredictionMap, list[dict[str, Any]]]:
    """session-fold OOF prediction と fold metric を作る。

    Args:
        keys: Dev task key の list。
        labels: key -> 正解 track_id。
        primary: anchor prediction。
        sources: source 名 -> prediction map。
        source_specs: source 名、topK、path の指定。
        protect_top: primary prefix を固定する件数。
        topk: 出力推薦件数。
        params: LightGBM parameters。
        feature_name: feature name。
        folds: fold 数。
        seed: session split seed。
        num_boost_round: OOF fold model の boosting round。

    Returns:
        OOF prediction map と fold report rows。
    """
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
            source_specs=source_specs,
            labels=labels,
        )
        valid_bundle = build_features(
            keys=valid_keys,
            primary=primary,
            sources=sources,
            source_specs=source_specs,
            labels=labels,
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
    parser.add_argument("--source", action="append", required=True, help="name[:topk]=dev_prediction.json")
    parser.add_argument("--blind_primary", help="name=blind_prediction.json")
    parser.add_argument("--blind_source", action="append", default=[], help="name[:topk]=blind_prediction.json")
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--default_source_topk", type=int, default=100)
    parser.add_argument("--protect_top", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--num_boost_round", type=int, default=450)
    parser.add_argument("--oof_num_boost_round", type=int, default=240)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_data_in_leaf", type=int, default=120)
    parser.add_argument("--lambda_l2", type=float, default=12.0)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--output_dev_full", type=Path, required=True)
    parser.add_argument("--output_dev_oof", type=Path, required=True)
    parser.add_argument("--output_blind_tracks", type=Path)
    parser.add_argument("--model_output", type=Path)
    args = parser.parse_args()

    primary_name, primary_path_text = args.primary.split("=", 1)
    source_specs = [
        parse_mixed_source(spec, default_topk=args.default_source_topk)
        for spec in args.source
    ]
    if len({source.name for source in source_specs}) != len(source_specs):
        raise ValueError("source names must be unique")
    source_names = [source.name for source in source_specs]
    names = feature_names(source_specs)
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
    primary_records = load_records(Path(primary_path_text), topk=args.topk)
    primary = records_to_map(primary_records)
    primary_records_by_key = records_by_key(primary_records)
    sources = {
        source.name: records_to_map(load_records(source.path, topk=source.topk))
        for source in source_specs
    }
    dev_bundle = build_features(
        keys=dev_keys,
        primary=primary,
        sources=sources,
        source_specs=source_specs,
        labels=labels,
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
        source_specs=source_specs,
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
        "primary": primary_name,
        "primary_path": primary_path_text,
        "source_paths": {source.name: str(source.path) for source in source_specs},
        "source_topks": {source.name: source.topk for source in source_specs},
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
        blind_primary_name, blind_primary_path_text = args.blind_primary.split("=", 1)
        blind_source_specs = [
            parse_mixed_source(spec, default_topk=args.default_source_topk)
            for spec in args.blind_source
        ]
        if {source.name for source in blind_source_specs} != set(source_names):
            raise ValueError("blind source names must match dev source names")
        blind_topks = {source.name: source.topk for source in blind_source_specs}
        if blind_topks != {source.name: source.topk for source in source_specs}:
            raise ValueError("blind source topk values must match dev source topk values")
        blind_records = load_records(Path(blind_primary_path_text), topk=args.topk)
        blind_primary = records_to_map(blind_records)
        blind_keys = [(str(record["session_id"]), int(record["turn_number"])) for record in blind_records]
        blind_records_by_key = records_by_key(blind_records)
        blind_sources = {
            source.name: records_to_map(load_records(source.path, topk=source.topk))
            for source in blind_source_specs
        }
        blind_bundle = build_features(
            keys=blind_keys,
            primary=blind_primary,
            sources=blind_sources,
            source_specs=source_specs,
            labels=None,
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
        report["blind_primary"] = blind_primary_name
        report["blind_primary_path"] = blind_primary_path_text
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
