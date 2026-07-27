"""exp015 の LightGBM ranker モデルを学習する。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from mcrs.experiments.exp015_candidate_rules_ablation.features import feature_columns


def require_lightgbm():
    """lightgbm をインポートして返す。未インストール時は分かりやすいエラーで即終了する。"""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is not installed. Install it with `uv pip install lightgbm` "
            "or install the project dependency before training."
        ) from exc
    return lgb


def read_table(path: str) -> pd.DataFrame:
    """拡張子に応じて parquet / csv / jsonlines を読み込む。

    Args:
        path: 入力ファイルパス。

    Returns:
        読み込んだ DataFrame。
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def group_count(df: pd.DataFrame) -> int:
    """DataFrame 内の group_id 数を数える。

    Args:
        df: group_id 列を持つ DataFrame。

    Returns:
        ユニークな group_id 数。
    """
    return int(df["group_id"].nunique())


def filter_positive_train_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """学習用 DataFrame から正例を含まない group を除外する。

    LightGBM LambdaRank は group 内に正例がないと有効な順位ペアを作れないため、
    train split だけでこの guard を適用する。valid split は zero-fill 評価のため
    正例なし group を残す。

    Args:
        df: group_id と label 列を持つ学習用 DataFrame。

    Returns:
        (正例あり group のみを残した DataFrame, group count summary) のタプル。

    Raises:
        ValueError: 必須列がない、または全 group が除外された場合。
    """
    missing_columns = sorted({"group_id", "label"} - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for train group filter: {missing_columns}")

    labels = pd.to_numeric(df["label"], errors="coerce").fillna(0.0)
    group_label_sum = labels.groupby(df["group_id"], sort=False).transform("sum")
    before = group_count(df)
    filtered = df[group_label_sum > 0].copy()
    after = group_count(filtered) if not filtered.empty else 0
    dropped = before - after
    if after <= 0:
        raise ValueError(
            "All train groups were dropped by positive-group filter. "
            f"groups_before_filter={before}"
        )
    return filtered, {
        "train_groups_before_filter": before,
        "train_groups_after_filter": after,
        "train_groups_dropped_no_positive": dropped,
    }


def slim_training_frame(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """LightGBM 学習に必要な列だけを残し、特徴量を float32 に寄せる。

    Args:
        df: 候補生成 dataset の DataFrame。
        feature_cols: 使用する特徴量列名。None の場合は f_ prefix から検出する。

    Returns:
        (必要列だけに絞った DataFrame, 特徴量列名リスト) のタプル。

    Raises:
        ValueError: 特徴量列が見つからない、または必須列が欠けている場合。
    """
    cols = list(feature_cols or feature_columns(df))
    if not cols:
        raise ValueError("No feature columns found. Expected columns prefixed with 'f_'.")
    required = ["group_id", "label"]
    if "candidate_rank" in df.columns:
        required.append("candidate_rank")
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for LightGBM training: {missing_columns}")

    # track_id や source list などの非特徴量列は数百万行で重いため、学習前に落として OOM を避ける。
    slim = df[required + cols].copy()
    for col in cols:
        slim[col] = pd.to_numeric(slim[col], errors="coerce").astype(np.float32)
    slim["label"] = pd.to_numeric(slim["label"], errors="coerce").fillna(0.0).astype(np.float32)
    return slim, cols


def prepare_frame(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str], list[int]]:
    """DataFrame を group_id ソートし、特徴量列と LightGBM 用 group sizes を抽出する。

    Args:
        df: 学習または検証用の特徴量 DataFrame。
        feature_cols: 使用する特徴量列名のリスト。None の場合は
            f_ プレフィックス列を自動検出する。

    Returns:
        (ソート済み DataFrame, 特徴量列名リスト, グループサイズリスト) のタプル。

    Raises:
        ValueError: f_ プレフィックスの特徴量列が見つからない場合。
    """
    # LightGBM の group パラメータはサンプルが group_id で連続していることを前提とするためソートが必須
    sort_cols = ["group_id", "candidate_rank"] if "candidate_rank" in df.columns else ["group_id"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    cols = list(feature_cols or feature_columns(df))
    if not cols:
        raise ValueError("No feature columns found. Expected columns prefixed with 'f_'.")
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0.0).astype(np.float32)
    group_sizes = df.groupby("group_id", sort=False).size().astype(int).tolist()
    return df, cols, group_sizes


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """1 グループの nDCG@k を計算する。

    Args:
        labels: 正解ラベルの配列（0/1）。
        scores: モデルスコアの配列。
        k: 評価打ち切り位置。

    Returns:
        nDCG@k の値。正例なしグループは 0.0 を返す。
    """
    # 正例なし（labels.max() == 0）は IDCG=0 でゼロ除算になるため早期リターンする
    if labels.size == 0 or labels.max() <= 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")[:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum(ideal * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def grouped_metrics(df: pd.DataFrame, scores: np.ndarray, ks: list[int]) -> dict[str, float]:
    """全グループの平均 nDCG / recall を計算する。

    Args:
        df: group_id・label 列を持つ DataFrame（prepare_frame でソート済みであること）。
        scores: 各行のモデルスコア配列（df と同じ行順）。
        ks: 評価する上位件数のリスト（例: [1, 10, 20]）。

    Returns:
        各指標名 → 平均値のマップ（ndcg@k, recall@k, groups_with_positive を含む）。
    """
    metrics = {f"ndcg@{k}": 0.0 for k in ks}
    metrics.update({f"recall@{k}": 0.0 for k in ks})
    group_count = 0
    offset = 0
    for _, group in df.groupby("group_id", sort=False):
        n = len(group)
        labels = group["label"].to_numpy(dtype=float)
        group_scores = scores[offset: offset + n]
        offset += n
        positives = float(labels.sum())
        # 正例なしグループは IDCG が定義できないため集計対象から除外する
        if positives <= 0:
            continue
        order = np.argsort(-group_scores, kind="stable")
        for k in ks:
            top = order[:k]
            metrics[f"ndcg@{k}"] += ndcg_at_k(labels, group_scores, k)
            metrics[f"recall@{k}"] += float(labels[top].sum()) / positives
        group_count += 1
    if group_count:
        for key in list(metrics):
            metrics[key] /= group_count
    metrics["groups_with_positive"] = float(group_count)
    return metrics


def zero_fill_metrics(
    positive_group_metrics: dict[str, float],
    *,
    total_groups: int,
    ks: list[int],
) -> dict[str, float]:
    """正例あり group 条件付き metric を全 task zero-fill metric に変換する。

    Args:
        positive_group_metrics: grouped_metrics が返した条件付き metric。
        total_groups: 正例なし group も含む評価 group 数。
        ks: 評価する上位件数のリスト。

    Returns:
        all_tasks_ndcg@k / all_tasks_recall@k を持つ辞書。
    """
    positive_groups = float(positive_group_metrics.get("groups_with_positive", 0.0))
    metrics: dict[str, float] = {}
    for k in ks:
        scale = positive_groups / float(total_groups) if total_groups else 0.0
        metrics[f"all_tasks_ndcg@{k}"] = float(positive_group_metrics.get(f"ndcg@{k}", 0.0)) * scale
        metrics[f"all_tasks_recall@{k}"] = float(positive_group_metrics.get(f"recall@{k}", 0.0)) * scale
    return metrics


def write_feature_importance(path: str, feature_names: list[str], importances: list[float]) -> None:
    """特徴量重要度を gain 降順で CSV ファイルに書き出す。

    Args:
        path: 出力 CSV ファイルパス。
        feature_names: 特徴量名のリスト。
        importances: 対応する重要度スコアのリスト。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(rows)


def resolve_objective(objective: str, metric: str) -> tuple[str, str]:
    """実験用 objective 名を LightGBM パラメータへ変換する。

    Args:
        objective: CLI で指定された objective 名。`logloss` は binary ranker の
            実験名として受け取り、LightGBM には `binary` として渡す。
        metric: CLI で指定された metric 名。`auto` の場合は objective に応じて
            `ndcg` または `binary_logloss` を選ぶ。

    Returns:
        (LightGBM objective, LightGBM metric) のタプル。

    Raises:
        ValueError: 未対応の objective が指定された場合。
    """
    normalized_objective = objective.strip().lower()
    if normalized_objective in {"lambdarank", "rank_xendcg"}:
        lgbm_objective = normalized_objective
        default_metric = "ndcg"
    elif normalized_objective in {"logloss", "binary"}:
        lgbm_objective = "binary"
        default_metric = "binary_logloss"
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    normalized_metric = metric.strip().lower()
    lgbm_metric = default_metric if normalized_metric == "auto" else normalized_metric
    return lgbm_objective, lgbm_metric


def build_lightgbm_params(args: argparse.Namespace) -> dict[str, Any]:
    """CLI 引数から LightGBM の学習パラメータを組み立てる。

    Args:
        args: objective と LightGBM hyperparameter を含む CLI 引数。

    Returns:
        lgb.train に渡す params 辞書。
    """
    lgbm_objective, lgbm_metric = resolve_objective(args.objective, args.metric)
    params: dict[str, Any] = {
        "objective": lgbm_objective,
        "metric": lgbm_metric,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "verbosity": -1,
        "seed": args.seed,
    }
    if lgbm_objective in {"lambdarank", "rank_xendcg"}:
        params["eval_at"] = [20]
        # binary label (0/1) に合わせる; LightGBM のデフォルト gain テーブルは多値ラベル向け
        params["label_gain"] = [0, 1]
    return params


def main(args: argparse.Namespace) -> None:
    """LightGBM ranker モデルの学習・評価・モデル保存を実行する。

    Args:
        args: CLI 引数。学習/検証データ、出力先、LightGBM hyperparameter、
            objective、early stopping 設定を含む。
    """
    lgb = require_lightgbm()
    train_df_raw = read_table(args.train_data)
    valid_df_raw = read_table(args.valid_data)
    train_df_raw, cols = slim_training_frame(train_df_raw)
    valid_df_raw, _ = slim_training_frame(valid_df_raw, cols)
    train_df_filtered, train_filter_stats = filter_positive_train_groups(train_df_raw)
    del train_df_raw
    train_df, cols, train_group = prepare_frame(train_df_filtered)
    del train_df_filtered
    valid_df, _, valid_group = prepare_frame(valid_df_raw, cols)
    del valid_df_raw

    train_data = lgb.Dataset(
        train_df[cols],
        label=train_df["label"].astype(float),
        group=train_group,
        feature_name=cols,
        free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        valid_df[cols],
        label=valid_df["label"].astype(float),
        group=valid_group,
        # valid のビン境界を train に揃えて、分布シフトによる評価誤差を防ぐ
        reference=train_data,
        free_raw_data=True,
    )
    params = build_lightgbm_params(args)
    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=args.log_period),
    ]
    evals_result: dict[str, Any] = {}
    callbacks.append(lgb.record_evaluation(evals_result))
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    booster.save_model(args.model_output)

    valid_scores = booster.predict(valid_df[cols], num_iteration=booster.best_iteration)
    train_scores = booster.predict(train_df[cols], num_iteration=booster.best_iteration)
    metric_ks = [1, 10, 20]
    train_metrics = grouped_metrics(train_df, train_scores, metric_ks)
    valid_metrics = grouped_metrics(valid_df, valid_scores, metric_ks)
    valid_total_groups = group_count(valid_df)
    all_task_metrics = zero_fill_metrics(valid_metrics, total_groups=valid_total_groups, ks=metric_ks)
    metrics = {
        "model_output": args.model_output,
        "objective": args.objective,
        "lightgbm_objective": params["objective"],
        "lightgbm_metric": params["metric"],
        "best_iteration": int(booster.best_iteration or args.num_boost_round),
        "feature_count": len(cols),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        **train_filter_stats,
        "valid_groups_total": valid_total_groups,
        "valid_groups_with_positive": int(valid_metrics.get("groups_with_positive", 0.0)),
        "valid_ndcg@20_positive_groups": float(valid_metrics.get("ndcg@20", 0.0)),
        "all_tasks_ndcg@20": all_task_metrics["all_tasks_ndcg@20"],
        "all_tasks_recall@20": all_task_metrics["all_tasks_recall@20"],
        "train": train_metrics,
        "valid": valid_metrics,
        "valid_all_tasks": all_task_metrics,
        "lightgbm_eval": evals_result,
    }
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.feature_importance_output:
        write_feature_importance(
            args.feature_importance_output,
            cols,
            booster.feature_importance(importance_type="gain").tolist(),
        )
    print(json.dumps(metrics["valid"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train exp015 LightGBM ranker model.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--valid_data", required=True)
    parser.add_argument("--model_output", required=True)
    parser.add_argument("--metrics_output", required=True)
    parser.add_argument("--feature_importance_output", default=None)
    parser.add_argument(
        "--objective",
        choices=["lambdarank", "rank_xendcg", "logloss", "binary"],
        default="lambdarank",
    )
    parser.add_argument("--metric", default="auto")
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_data_in_leaf", type=int, default=30)
    parser.add_argument("--feature_fraction", type=float, default=0.85)
    parser.add_argument("--bagging_fraction", type=float, default=0.85)
    parser.add_argument("--bagging_freq", type=int, default=1)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--num_boost_round", type=int, default=2000)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_period", type=int, default=50)
    main(parser.parse_args())
