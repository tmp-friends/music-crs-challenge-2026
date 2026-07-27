"""exp090-fix の候補・特徴量を固定したまま reranker を multi-family ensemble へ拡張する。

best nDCG モデル（exp090-fix = exp058 wide100 + cross_session source の 6-seed LightGBM
seed ensemble）への **model 層だけの addon**。候補プール・特徴量・exclude_map・source は
exp090-fix と byte-identical（同じ ``build_features`` / 同じ run script の source 群）に保ち、
LightGBM 単一 family の seed-average を **LightGBM + XGBoost + CatBoost の family ensemble**
へ置き換える。

設計（advisor 指針）:

- 3 family は同じ 78 次元 rank 特徴量を入力に取る。GBDT 3 種は同一関数族なので point-nDCG の
  上振れは期待薄で、狙いは主に **seed 非決定性に対する安定化（variance 低減）**。別関数族
  （TabM/TabPFN）は別途検討。
- blend は family ごとに group 内 z-score 正規化してから **family 単位で等重み平均**
  （6 LightGBM seed でも 1 票。``mcrs.rerank_modules.multi_family.blend_family_scores``）。
- 採否 gate は Blind A。Dev OOF は floor 判定専用で、**同一 run・同一 fold で LightGBM 単独
  OOF も再計算**して blend との差分を attributable にする。
- blend 重みは Dev OOF で tune しない（CV-LB 逆相関で overfit するため等重み固定）。

LightGBM family は exp058 ``seed_ensemble_wide100`` の ``base_params`` /
``train_seed_models`` / ``averaged_scores`` をそのまま再利用し、exp090-fix と同一の学習に
する。XGBoost / CatBoost wrapper は ``mcrs.rerank_modules.multi_family`` を使う。
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
    write_records,
)
from mcrs.experiments.exp027_wide_source_lgbm.rank_source_lgbm_mixed_topk import (
    build_features,
    feature_names,
    parse_mixed_source,
    split_indices_by_session,
)
from mcrs.experiments.exp058_wide100_seed_ensemble.seed_ensemble_wide100 import (
    averaged_scores,
    base_params,
    train_seed_models,
)
from mcrs.rerank_modules.multi_family import (
    blend_family_scores,
    catboost_params,
    fit_catboost_ranker,
    fit_xgb_ranker,
    groupwise_zscore,
    predict_catboost,
    predict_xgb,
    xgb_params,
)

# tabm family は torch / tabm 依存なので遅延 import（GBDT のみの run では読み込まない）。
ALL_FAMILIES = {"lgbm", "xgb", "catboost", "tabm"}


def parse_weights(text: str | None, families: list[str]) -> dict[str, float]:
    """``name=w,...`` 形式の blend 重み指定を辞書化する。省略時は等重み。

    Args:
        text: ``lgbm=1,xgb=1,catboost=1`` のような指定（省略可）。
        families: 有効 family 名。

    Returns:
        family 名 -> 重み。

    Raises:
        ValueError: 未知の family 名が含まれる場合。
    """
    if not text:
        return {name: 1.0 for name in families}
    weights: dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in families:
            raise ValueError(f"unknown family in weights: {name}")
        weights[name] = float(value)
    # 指定されなかった family は 0 でなく 1（部分指定でも残りを等重みに保つ）。
    for name in families:
        weights.setdefault(name, 1.0)
    return weights


def train_families(
    bundle: Any,
    *,
    families: list[str],
    args: argparse.Namespace,
    names: list[str],
    lgbm_seeds: list[int],
    xgb_seeds: list[int],
    catboost_seeds: list[int],
    lgbm_rounds: int,
    xgb_rounds: int,
    catboost_iters: int,
    tabm_seeds: list[int] | None = None,
    tabm_epochs: int | None = None,
) -> dict[str, list[Any]]:
    """指定 family ごとに（seed 数だけ）model を学習し family 名 -> model list を返す。

    1 度学習した model は dev-full / Blind A 双方の採点へ流用するため list で保持する。

    Args:
        bundle: 学習用 FeatureBundle。
        families: 学習する family 名（``lgbm`` / ``xgb`` / ``catboost``）。
        args: CLI 引数。
        names: feature 名。
        lgbm_seeds: LightGBM family の seed list。
        xgb_seeds: XGBoost family の seed list。
        catboost_seeds: CatBoost family の seed list。
        lgbm_rounds: LightGBM boosting round。
        xgb_rounds: XGBoost boosting round。
        catboost_iters: CatBoost iterations。

    Returns:
        family 名 -> 学習済み model の list。
    """
    models: dict[str, list[Any]] = {}
    if "lgbm" in families:
        # exp090-fix と同一の LightGBM 学習（base_params + seed ごとに 1 model）。
        models["lgbm"] = train_seed_models(
            bundle, args=args, seeds=lgbm_seeds, names=names, num_boost_round=lgbm_rounds,
        )
    if "xgb" in families:
        models["xgb"] = [
            fit_xgb_ranker(
                bundle.x, bundle.y, bundle.groups,
                params=xgb_params(seed=seed, num_threads=args.num_threads, device=args.xgb_device),
                num_boost_round=xgb_rounds, feature_name=names,
            )
            for seed in xgb_seeds
        ]
    if "catboost" in families:
        models["catboost"] = [
            fit_catboost_ranker(
                bundle.x, bundle.y, bundle.groups,
                params={**catboost_params(seed=seed, num_threads=args.num_threads,
                                          task_type=args.catboost_task_type),
                        "iterations": catboost_iters},
                feature_name=names,
            )
            for seed in catboost_seeds
        ]
    if "tabm" in families:
        # 遅延 import（torch / tabm）。TabM は内部 k-MLP ensemble なので seed 1 本でも
        # 十分多様だが、複数 seed 指定があれば family_scores 側で平均する。
        from mcrs.rerank_modules.tabm_ranker import TabMConfig, fit_tabm_ranker

        config = TabMConfig()
        if tabm_epochs is not None:
            config.epochs = tabm_epochs
        models["tabm"] = [
            fit_tabm_ranker(
                bundle.x, bundle.y, seed=seed, config=config,
                device=args.tabm_device,
            )
            for seed in (tabm_seeds or [20260616])
        ]
    return models


def family_scores(
    models: dict[str, list[Any]], x: np.ndarray, *, names: list[str],
) -> dict[str, np.ndarray]:
    """family ごとに seed 平均した生 score を返す（blend 前段）。

    Args:
        models: family 名 -> model list。
        x: 採点する特徴量行列。
        names: feature 名（XGBoost DMatrix の整合用）。

    Returns:
        family 名 -> seed 平均 score。
    """
    scores: dict[str, np.ndarray] = {}
    for family, family_models in models.items():
        if family == "lgbm":
            scores[family] = averaged_scores(family_models, x)
        elif family == "xgb":
            acc = np.zeros(x.shape[0], dtype=np.float64)
            for model in family_models:
                acc += predict_xgb(model, x, feature_name=names)
            scores[family] = acc / float(len(family_models))
        elif family == "catboost":
            acc = np.zeros(x.shape[0], dtype=np.float64)
            for model in family_models:
                acc += predict_catboost(model, x)
            scores[family] = acc / float(len(family_models))
        elif family == "tabm":
            from mcrs.rerank_modules.tabm_ranker import predict_tabm

            acc = np.zeros(x.shape[0], dtype=np.float64)
            for state in family_models:
                acc += predict_tabm(state, x)
            scores[family] = acc / float(len(family_models))
    return scores


def run_oof(
    *,
    keys: list[Any],
    labels: dict[Any, str],
    primary: Any,
    sources: dict[str, Any],
    source_specs: list[Any],
    args: argparse.Namespace,
    names: list[str],
    families: list[str],
    weights: dict[str, float],
    exclude_map: dict[Any, set[str]] | None,
    secondary_families: list[str] | None = None,
) -> dict[str, Any]:
    """session-fold OOF を family ごと + blend + LightGBM 単独で計算する。

    同一 fold split・同一 train bundle で全 family を学習し、family 別 OOF / blend OOF /
    LightGBM 単独 OOF を併記する（blend 効果を attributable にする）。floor 判定専用で
    OOF round は full-fit より軽くする。

    Args:
        keys: Dev task key の list。
        labels: key -> 正解 track_id。
        primary: anchor prediction map。
        sources: source 名 -> prediction map。
        source_specs: source 指定。
        args: CLI 引数。
        names: feature 名。
        families: 学習 family。
        weights: blend 重み。
        exclude_map: 候補 union 除外集合。

    Returns:
        ``{"oof_ndcg": {family/blend/lgbm_only: ndcg}, "folds": [...] }``。
    """
    fold_indices = split_indices_by_session(keys, folds=args.folds, seed=args.split_seed)
    all_indices = set(range(len(keys)))
    # family 別・blend・lgbm 単独の OOF prediction を貯める。
    oof_family: dict[str, dict[Any, Any]] = {name: {} for name in families}
    oof_blend: dict[Any, Any] = {}
    oof_secondary: dict[Any, Any] = {}
    oof_lgbm_only: dict[Any, Any] = {}
    # secondary blend は primary families の部分集合のみ（同一 train model を使い回す）。
    use_secondary = bool(secondary_families) and set(secondary_families) != set(families)
    fold_rows: list[dict[str, Any]] = []
    oof_lgbm_seeds = [int(s) for s in args.seeds.split(",")][: args.oof_lgbm_n]
    oof_xgb_seeds = [int(s) for s in args.seeds.split(",")][: args.oof_xgb_n]
    oof_catboost_seeds = [int(s) for s in args.seeds.split(",")][: args.oof_catboost_n]

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
        oof_tabm_seeds = [int(s) for s in args.seeds.split(",")][: args.oof_tabm_n]
        models = train_families(
            train_bundle, families=families, args=args, names=names,
            lgbm_seeds=oof_lgbm_seeds, xgb_seeds=oof_xgb_seeds, catboost_seeds=oof_catboost_seeds,
            lgbm_rounds=args.oof_num_boost_round, xgb_rounds=args.oof_xgb_rounds,
            catboost_iters=args.oof_catboost_iters,
            tabm_seeds=oof_tabm_seeds, tabm_epochs=args.oof_tabm_epochs,
        )
        fscores = family_scores(models, valid_bundle.x, names=names)
        # family 別 OOF
        for name in families:
            preds = score_to_predictions(
                valid_bundle, valid_keys, primary, scores=fscores[name],
                protect_top=args.protect_top, topk=args.topk,
            )
            oof_family[name].update(preds)
        # secondary blend OOF（primary families の部分集合。同一 model から）
        if use_secondary:
            sub_scores = {n: fscores[n] for n in secondary_families}
            sub_blended = blend_family_scores(sub_scores, valid_bundle.groups)
            oof_secondary.update(score_to_predictions(
                valid_bundle, valid_keys, primary, scores=sub_blended,
                protect_top=args.protect_top, topk=args.topk,
            ))
        # blend OOF
        blended = blend_family_scores(fscores, valid_bundle.groups, weights=weights)
        oof_blend.update(score_to_predictions(
            valid_bundle, valid_keys, primary, scores=blended,
            protect_top=args.protect_top, topk=args.topk,
        ))
        # LightGBM 単独 OOF（attribution baseline。lgbm が families に居れば z-score 後も
        # 単一 family なので順位不変だが、念のため raw score を使う）
        if "lgbm" in fscores:
            oof_lgbm_only.update(score_to_predictions(
                valid_bundle, valid_keys, primary, scores=fscores["lgbm"],
                protect_top=args.protect_top, topk=args.topk,
            ))
        fold_rows.append({
            "fold": fold_index, "tasks": len(valid_keys),
            "blend_ndcg": ndcg(oof_blend, valid_keys, labels),
        })

    oof_ndcg: dict[str, float] = {}
    for name in families:
        oof_ndcg[name] = ndcg(oof_family[name], keys, labels)
    oof_ndcg["blend"] = ndcg(oof_blend, keys, labels)
    if use_secondary:
        oof_ndcg["secondary_blend"] = ndcg(oof_secondary, keys, labels)
    if oof_lgbm_only:
        oof_ndcg["lgbm_only"] = ndcg(oof_lgbm_only, keys, labels)
    return {"oof_ndcg": oof_ndcg, "folds": fold_rows}


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser を構築する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--blind_primary", required=True)
    parser.add_argument("--blind_source", action="append", required=True)
    parser.add_argument("--dataset_name", default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", default="20260616,20260617,20260618,20260619,20260620,20260621",
                        help="LightGBM family の seed list（カンマ区切り）")
    parser.add_argument("--families", default="lgbm,xgb,catboost",
                        help="ensemble に入れる family（lgbm/xgb/catboost）")
    parser.add_argument("--blend_weights", default=None, help="family=weight,... 省略で等重み")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--default_source_topk", type=int, default=100)
    parser.add_argument("--protect_top", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=20260616)
    # LightGBM（exp090-fix と同一既定）
    parser.add_argument("--num_boost_round", type=int, default=450)
    parser.add_argument("--oof_num_boost_round", type=int, default=240)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_data_in_leaf", type=int, default=80)
    parser.add_argument("--lambda_l2", type=float, default=8.0)
    parser.add_argument("--num_threads", type=int, default=8)
    # XGBoost
    parser.add_argument("--xgb_n", type=int, default=2, help="XGBoost family の seed 数")
    parser.add_argument("--xgb_rounds", type=int, default=450)
    parser.add_argument("--xgb_device", default="cuda")
    parser.add_argument("--oof_xgb_n", type=int, default=1)
    parser.add_argument("--oof_xgb_rounds", type=int, default=300)
    # CatBoost
    parser.add_argument("--catboost_n", type=int, default=2, help="CatBoost family の seed 数")
    parser.add_argument("--catboost_iters", type=int, default=1000)
    parser.add_argument("--catboost_task_type", default="GPU")
    parser.add_argument("--oof_catboost_n", type=int, default=1)
    parser.add_argument("--oof_catboost_iters", type=int, default=700)
    # TabM（optional・別関数族 probe）
    parser.add_argument("--tabm_n", type=int, default=1, help="TabM family の seed 数")
    parser.add_argument("--tabm_epochs", type=int, default=6)
    parser.add_argument("--tabm_device", default="cuda")
    parser.add_argument("--oof_tabm_n", type=int, default=1)
    parser.add_argument("--oof_tabm_epochs", type=int, default=6)
    # OOF LightGBM seed 数（floor 用に軽く）
    parser.add_argument("--oof_lgbm_n", type=int, default=1)
    parser.add_argument("--skip_oof", action="store_true")
    parser.add_argument("--exclude_json", type=Path, default=None)
    parser.add_argument("--allow_short_sources", action="store_true")
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--output_dev_full", type=Path, required=True)
    parser.add_argument("--output_blind_tracks", type=Path, required=True)
    # secondary blend（同一 train model から families の部分集合で別 blend を出力。
    # 例: primary=lgbm,xgb,catboost,tabm に対し secondary=lgbm,xgb,catboost で trio 版も同時生成し
    # 二重学習を避ける）。
    parser.add_argument("--secondary_families", default=None, help="部分集合 family。省略で無効")
    parser.add_argument("--output_dev_full2", type=Path, default=None)
    parser.add_argument("--output_blind_tracks2", type=Path, default=None)
    return parser


def main() -> None:
    """CLI entrypoint。"""
    args = build_arg_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for name in families:
        if name not in ALL_FAMILIES:
            raise ValueError(f"unknown family: {name}")
    weights = parse_weights(args.blend_weights, families)
    secondary_families: list[str] | None = None
    if args.secondary_families:
        secondary_families = [f.strip() for f in args.secondary_families.split(",") if f.strip()]
        if not set(secondary_families) <= set(families):
            raise ValueError("secondary_families must be a subset of --families")
    all_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not all_seeds:
        raise ValueError("--seeds に最低 1 つの seed が必要")
    lgbm_seeds = all_seeds
    xgb_seeds = all_seeds[: args.xgb_n]
    catboost_seeds = all_seeds[: args.catboost_n]

    primary_name, primary_path_text = args.primary.split("=", 1)
    source_specs = [parse_mixed_source(s, default_topk=args.default_source_topk) for s in args.source]
    source_names = [s.name for s in source_specs]
    names = feature_names(source_specs)

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

    # 全 Dev で family ごとに学習し dev-full / Blind A 双方へ流用。
    tabm_seeds = all_seeds[: args.tabm_n]
    full_models = train_families(
        dev_bundle, families=families, args=args, names=names,
        lgbm_seeds=lgbm_seeds, xgb_seeds=xgb_seeds, catboost_seeds=catboost_seeds,
        lgbm_rounds=args.num_boost_round, xgb_rounds=args.xgb_rounds,
        catboost_iters=args.catboost_iters,
        tabm_seeds=tabm_seeds, tabm_epochs=args.tabm_epochs,
    )
    dev_fscores = family_scores(full_models, dev_bundle.x, names=names)
    dev_blend = blend_family_scores(dev_fscores, dev_bundle.groups, weights=weights)
    full_predictions = score_to_predictions(
        dev_bundle, dev_keys, primary, scores=dev_blend,
        protect_top=args.protect_top, topk=args.topk,
    )
    write_records(args.output_dev_full, build_records(full_predictions, primary_records_by_key, dev_keys))

    base_ndcg = ndcg(primary, dev_keys, labels)
    full_ndcg = ndcg(full_predictions, dev_keys, labels)
    # secondary blend（families の部分集合）の dev-full も同一 model から生成。
    full_secondary_ndcg = None
    secondary_full_predictions = None
    if secondary_families and set(secondary_families) != set(families):
        sub_dev = blend_family_scores({n: dev_fscores[n] for n in secondary_families}, dev_bundle.groups)
        secondary_full_predictions = score_to_predictions(
            dev_bundle, dev_keys, primary, scores=sub_dev,
            protect_top=args.protect_top, topk=args.topk,
        )
        full_secondary_ndcg = ndcg(secondary_full_predictions, dev_keys, labels)
        if args.output_dev_full2:
            write_records(args.output_dev_full2,
                          build_records(secondary_full_predictions, primary_records_by_key, dev_keys))
    # family 別 full-fit ndcg（standalone parity 確認用）
    full_family_ndcg: dict[str, float] = {}
    for name in families:
        preds = score_to_predictions(
            dev_bundle, dev_keys, primary, scores=dev_fscores[name],
            protect_top=args.protect_top, topk=args.topk,
        )
        full_family_ndcg[name] = ndcg(preds, dev_keys, labels)

    oof_block: dict[str, Any] = {}
    if not args.skip_oof:
        oof_block = run_oof(
            keys=dev_keys, labels=labels, primary=primary, sources=sources,
            source_specs=source_specs, args=args, names=names, families=families,
            weights=weights, exclude_map=exclude_map, secondary_families=secondary_families,
        )

    # Blind A: full-fit ensemble を適用。
    blind_primary_name, blind_primary_path_text = args.blind_primary.split("=", 1)
    blind_source_specs = [parse_mixed_source(s, default_topk=args.default_source_topk) for s in args.blind_source]
    if {s.name for s in blind_source_specs} != set(source_names):
        raise ValueError("blind source names must match dev source names")
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
    blind_fscores = family_scores(full_models, blind_bundle.x, names=names)
    blind_blend = blend_family_scores(blind_fscores, blind_bundle.groups, weights=weights)
    blind_predictions = score_to_predictions(
        blind_bundle, blind_keys, blind_primary, scores=blind_blend,
        protect_top=args.protect_top, topk=args.topk,
    )
    write_records(args.output_blind_tracks, build_records(blind_predictions, blind_records_by_key, blind_keys))

    # secondary blend の Blind A tracks も同一 model から生成（trio 版 zip 用）。
    secondary_blind_change = None
    if secondary_families and set(secondary_families) != set(families) and args.output_blind_tracks2:
        sub_blind = blend_family_scores({n: blind_fscores[n] for n in secondary_families}, blind_bundle.groups)
        sub_blind_predictions = score_to_predictions(
            blind_bundle, blind_keys, blind_primary, scores=sub_blind,
            protect_top=args.protect_top, topk=args.topk,
        )
        write_records(args.output_blind_tracks2,
                      build_records(sub_blind_predictions, blind_records_by_key, blind_keys))
        secondary_blind_change = change_stats(blind_primary, sub_blind_predictions, blind_keys)

    report = {
        "primary": primary_name,
        "primary_path": primary_path_text,
        "source_paths": {s.name: str(s.path) for s in source_specs},
        "families": families,
        "blend_weights": weights,
        "lgbm_seeds": lgbm_seeds,
        "xgb_seeds": xgb_seeds,
        "catboost_seeds": catboost_seeds,
        "tabm_seeds": tabm_seeds if "tabm" in families else [],
        "secondary_families": secondary_families,
        "num_boost_round": args.num_boost_round,
        "xgb_rounds": args.xgb_rounds,
        "catboost_iters": args.catboost_iters,
        "protect_top": args.protect_top,
        "candidate_rows": int(dev_bundle.x.shape[0]),
        "candidate_positive_count": int(dev_bundle.candidate_positive_count),
        "base_ndcg": base_ndcg,
        "full_fit_blend_ndcg": full_ndcg,
        "full_fit_blend_delta": full_ndcg - base_ndcg,
        "full_fit_secondary_blend_ndcg": full_secondary_ndcg,
        "full_fit_family_ndcg": full_family_ndcg,
        "oof": oof_block,
        "blind_change_stats_vs_primary": change_stats(blind_primary, blind_predictions, blind_keys),
        "secondary_blind_change_stats_vs_primary": secondary_blind_change,
        "output_dev_full": str(args.output_dev_full),
        "output_blind_tracks": str(args.output_blind_tracks),
        "output_blind_tracks2": str(args.output_blind_tracks2) if args.output_blind_tracks2 else None,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_ndcg": base_ndcg,
        "full_fit_blend_ndcg": full_ndcg,
        "full_fit_family_ndcg": full_family_ndcg,
        "oof_ndcg": oof_block.get("oof_ndcg") if oof_block else None,
        "families": families,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
