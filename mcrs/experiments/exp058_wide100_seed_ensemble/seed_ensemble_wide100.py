"""exp027 wide100_p0 を複数 seed の score-average で安定化する seed ensemble。

[[blind-a-ndcg-noise-floor]] のとおり Blind A 80 件は LightGBM 学習の非決定性だけで
ndcg が ±0.02 揺れる。単一 seed の wide100_p0 はその variance をそのまま LB に乗せて
しまうため、deterministic な複数 seed を学習し各候補行の生スコアを平均して再ランキング
する。top1 は固定せず(``protect_top=0``)、ensemble スコアで全体を並べ替えるので「top1 を
守る保守的 fusion」ではなく純粋な再ランキングになる。

スコア平均は全 seed が同一の候補行(同一 feature 順)を採点するため、RRF より情報損失が
少ない正攻法の rank ensemble。Dev OOF は floor 判定にのみ使い、selector には使わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mcrs.experiments.exp022_continuity_lgbm.rank_source_lgbm import (
    build_records,
    change_stats,
    load_dev_labels,
    load_records,
    ndcg,
    records_by_key,
    records_to_map,
    score_to_predictions,
    train_model,
    write_records,
)
from mcrs.experiments.exp027_wide_source_lgbm.rank_source_lgbm_mixed_topk import (
    build_features,
    feature_names,
    parse_mixed_source,
    split_indices_by_session,
)


def base_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """seed 以外を固定した LightGBM パラメータを返す。

    Args:
        args: CLI 引数(num_leaves / min_data_in_leaf / lambda_l2 を参照)。
        seed: この model 用の乱数 seed。

    Returns:
        lambdarank 用パラメータ辞書。
    """
    return {
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
        "seed": seed,
        "num_threads": args.num_threads,
        "deterministic": True,
    }


def train_seed_models(
    train_bundle: Any,
    *,
    args: argparse.Namespace,
    seeds: list[int],
    names: list[str],
    num_boost_round: int,
) -> list[Any]:
    """seed ごとに 1 model ずつ学習して list で返す。

    Args:
        train_bundle: 学習用 FeatureBundle。
        args: CLI 引数。
        seeds: seed list。
        names: feature name。
        num_boost_round: boosting round 数。

    Returns:
        booster の list。1 度学習すれば dev/blind 双方に予測を流用できるので、
        full-fit と Blind A 適用で同一 model を共有し二重学習を避ける。
    """
    return [
        train_model(
            train_bundle,
            params=base_params(args, seed),
            num_boost_round=num_boost_round,
            feature_name=names,
        )
        for seed in seeds
    ]


def averaged_scores(models: list[Any], x: np.ndarray) -> np.ndarray:
    """model 群の生スコアを平均して返す。

    Args:
        models: booster list。
        x: 採点する feature 行列。

    Returns:
        平均スコア配列。順位ではなくスコア平均なので候補全体を再ランキングできる。
    """
    acc = np.zeros(x.shape[0], dtype=np.float64)
    for model in models:
        acc += model.predict(x)
    return acc / float(len(models))


def run_oof_ensemble(
    *,
    keys: list[Any],
    labels: dict[Any, str],
    primary: Any,
    sources: dict[str, Any],
    source_specs: list[Any],
    args: argparse.Namespace,
    names: list[str],
    exclude_map: dict[Any, set[str]] | None = None,
    seeds: list[int],
) -> tuple[Any, list[dict[str, Any]]]:
    """session-fold OOF を seed ensemble で計算する。

    各 fold で全 seed を学習し valid スコアを平均する。floor 判定専用。

    Args:
        keys: Dev task key の list。
        labels: key -> 正解 track_id。
        primary: anchor prediction map。
        sources: source 名 -> prediction map。
        source_specs: source 指定。
        args: CLI 引数。
        names: feature name。
        seeds: ensemble seed list。

    Returns:
        OOF prediction map と fold ごとの report 行。
    """
    fold_indices = split_indices_by_session(keys, folds=args.folds, seed=args.split_seed)
    all_indices = set(range(len(keys)))
    oof_predictions: Any = {}
    fold_rows: list[dict[str, Any]] = []
    for fold_index, valid_indices in enumerate(fold_indices):
        valid_set = {int(index) for index in valid_indices.tolist()}
        train_indices = sorted(all_indices - valid_set)
        train_keys = [keys[index] for index in train_indices]
        valid_keys = [keys[int(index)] for index in valid_indices.tolist()]
        train_bundle = build_features(
            keys=train_keys, primary=primary, sources=sources,
            source_specs=source_specs, labels=labels, exclude_map=exclude_map,
        )
        valid_bundle = build_features(
            keys=valid_keys, primary=primary, sources=sources,
            source_specs=source_specs, labels=labels, exclude_map=exclude_map,
        )
        models = train_seed_models(
            train_bundle, args=args, seeds=seeds, names=names,
            num_boost_round=args.oof_num_boost_round,
        )
        scores = averaged_scores(models, valid_bundle.x)
        fold_predictions = score_to_predictions(
            valid_bundle, valid_keys, primary, scores=scores,
            protect_top=args.protect_top, topk=args.topk,
        )
        oof_predictions.update(fold_predictions)
        fold_rows.append({
            "fold": fold_index,
            "tasks": len(valid_keys),
            "oof_ndcg": ndcg(fold_predictions, valid_keys, labels),
        })
    return oof_predictions, fold_rows


def main() -> None:
    """CLI entrypoint。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--blind_primary", required=True)
    parser.add_argument("--blind_source", action="append", required=True)
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", default="20260616,20260617,20260618,20260619,20260620,20260621",
                        help="カンマ区切り seed list")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--default_source_topk", type=int, default=100)
    parser.add_argument("--protect_top", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=20260616)
    parser.add_argument("--num_boost_round", type=int, default=450)
    parser.add_argument("--oof_num_boost_round", type=int, default=240)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_data_in_leaf", type=int, default=80)
    parser.add_argument("--lambda_l2", type=float, default=8.0)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--skip_oof", action="store_true", help="floor 判定用 OOF を省略")
    parser.add_argument("--exclude_json", type=Path, default=None,
                        help="候補 union から外す track を per-key で指定する JSON（within-session 既出除外等。省略で従来挙動）")
    parser.add_argument("--allow_short_sources", action="store_true",
                        help="source 行の track 数が topk 未満（空含む）でも許容。sparse 候補 source を popularity padding 無しで渡すため。primary は常に strict")
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--output_dev_full", type=Path, required=True)
    parser.add_argument("--output_blind_tracks", type=Path, required=True)
    parser.add_argument("--output_dev_oof", type=Path)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds に最低 1 つの seed が必要です")
    primary_name, primary_path_text = args.primary.split("=", 1)
    source_specs = [parse_mixed_source(s, default_topk=args.default_source_topk) for s in args.source]
    source_names = [s.name for s in source_specs]
    names = feature_names(source_specs)

    # 候補 union から外す track の per-key 集合（within-session 既出除外等）。省略で None＝従来挙動。
    exclude_map: dict[Any, set[str]] | None = None
    if args.exclude_json is not None:
        exclude_records = json.loads(Path(args.exclude_json).read_text(encoding="utf-8"))
        exclude_map = {
            (str(r["session_id"]), int(r["turn_number"])): {str(t) for t in r.get("exclude_track_ids", [])}
            for r in exclude_records
        }

    dev_keys, labels = load_dev_labels(args.dataset_name, args.split)
    primary_records = load_records(Path(primary_path_text), topk=args.topk)
    primary = records_to_map(primary_records)
    primary_records_by_key = records_by_key(primary_records)
    sources = {
        s.name: records_to_map(load_records(s.path, topk=s.topk, allow_short=args.allow_short_sources))
        for s in source_specs
    }
    dev_bundle = build_features(
        keys=dev_keys, primary=primary, sources=sources,
        source_specs=source_specs, labels=labels, exclude_map=exclude_map,
    )

    # 全 Dev で seed ごとに 1 度だけ学習し、dev-full と Blind A の双方にこの model 群を流用する。
    full_models = train_seed_models(
        dev_bundle, args=args, seeds=seeds, names=names,
        num_boost_round=args.num_boost_round,
    )
    dev_scores = averaged_scores(full_models, dev_bundle.x)
    full_predictions = score_to_predictions(
        dev_bundle, dev_keys, primary, scores=dev_scores,
        protect_top=args.protect_top, topk=args.topk,
    )
    write_records(args.output_dev_full, build_records(full_predictions, primary_records_by_key, dev_keys))

    base_ndcg = ndcg(primary, dev_keys, labels)
    full_ndcg = ndcg(full_predictions, dev_keys, labels)

    oof_ndcg = None
    fold_rows: list[dict[str, Any]] = []
    if not args.skip_oof:
        oof_predictions, fold_rows = run_oof_ensemble(
            keys=dev_keys, labels=labels, primary=primary, sources=sources,
            source_specs=source_specs, args=args, names=names, seeds=seeds,
            exclude_map=exclude_map,
        )
        oof_ndcg = ndcg(oof_predictions, dev_keys, labels)
        if args.output_dev_oof:
            write_records(args.output_dev_oof, build_records(oof_predictions, primary_records_by_key, dev_keys))

    # Blind A: full-fit ensemble を適用。
    blind_primary_name, blind_primary_path_text = args.blind_primary.split("=", 1)
    blind_source_specs = [parse_mixed_source(s, default_topk=args.default_source_topk) for s in args.blind_source]
    if {s.name for s in blind_source_specs} != set(source_names):
        raise ValueError("blind source names must match dev source names")
    # dev と blind で source topk がずれると feature 整合が崩れるため一致を強制する。
    if {s.name: s.topk for s in blind_source_specs} != {s.name: s.topk for s in source_specs}:
        raise ValueError("blind source topk values must match dev source topk values")
    blind_records = load_records(Path(blind_primary_path_text), topk=args.topk)
    blind_primary = records_to_map(blind_records)
    blind_keys = [(str(r["session_id"]), int(r["turn_number"])) for r in blind_records]
    blind_records_by_key = records_by_key(blind_records)
    blind_sources = {
        s.name: records_to_map(load_records(s.path, topk=s.topk, allow_short=args.allow_short_sources))
        for s in blind_source_specs
    }
    blind_bundle = build_features(
        keys=blind_keys, primary=blind_primary, sources=blind_sources,
        source_specs=source_specs, labels=None, exclude_map=exclude_map,
    )
    # dev-full で学習済みの同一 model 群を Blind A 候補行へ適用しスコア平均する。
    blind_scores = averaged_scores(full_models, blind_bundle.x)
    blind_predictions = score_to_predictions(
        blind_bundle, blind_keys, blind_primary, scores=blind_scores,
        protect_top=args.protect_top, topk=args.topk,
    )
    write_records(args.output_blind_tracks, build_records(blind_predictions, blind_records_by_key, blind_keys))

    report = {
        "primary": primary_name,
        "primary_path": primary_path_text,
        "source_paths": {s.name: str(s.path) for s in source_specs},
        "seeds": seeds,
        "num_seeds": len(seeds),
        "params_template": base_params(args, seeds[0]),
        "num_boost_round": args.num_boost_round,
        "oof_num_boost_round": args.oof_num_boost_round,
        "protect_top": args.protect_top,
        "candidate_rows": int(dev_bundle.x.shape[0]),
        "candidate_positive_count": int(dev_bundle.candidate_positive_count),
        "base_ndcg": base_ndcg,
        "full_fit_ndcg": full_ndcg,
        "full_fit_delta": full_ndcg - base_ndcg,
        "oof_ndcg": oof_ndcg,
        "oof_delta": (oof_ndcg - base_ndcg) if oof_ndcg is not None else None,
        "folds": fold_rows,
        "blind_change_stats_vs_primary": change_stats(blind_primary, blind_predictions, blind_keys),
        "output_dev_full": str(args.output_dev_full),
        "output_blind_tracks": str(args.output_blind_tracks),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_ndcg": base_ndcg, "full_fit_ndcg": full_ndcg, "oof_ndcg": oof_ndcg,
        "num_seeds": len(seeds), "output_report": str(args.output_report),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
