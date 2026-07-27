"""LightGBM モデルを候補 row に適用して prediction を作る（共有ライブラリ）。

exp014 由来。``mcrs/ltr/`` へ切り出した共有モジュールで、旧
``mcrs.experiments.exp014_lightgbm_ltr_reranker.apply_lgbm`` は re-export shim 経由で解決される
（shim 側に ``__main__`` 委譲があるため旧パスの CLI 実行も維持される）。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def require_lightgbm():
    """lightgbm をインポートして返す。未インストール時は分かりやすいエラーで即終了する。"""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is not installed. Install it with `uv pip install lightgbm`."
        ) from exc
    return lgb


def parquet_columns(path: str) -> list[str] | None:
    """parquet ファイルの schema から列名だけを取得する。

    Args:
        path: 入力ファイルパス。

    Returns:
        parquet の場合は列名リスト。それ以外では None。
    """
    if not path.endswith(".parquet"):
        return None
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_table(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """拡張子に応じて parquet / csv / jsonlines を読み込む。

    Args:
        path: 入力ファイルパス。
        columns: 読み込む列名。None の場合は全列を読む。

    Returns:
        読み込んだ DataFrame。
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    if path.endswith(".csv"):
        return pd.read_csv(path, usecols=columns)
    df = pd.read_json(path, lines=True)
    return df if columns is None else df[columns]


def read_candidate_table(path: str, feature_names: list[str]) -> pd.DataFrame:
    """予測に必要な metadata とモデル特徴量だけを読み込む。

    Args:
        path: candidate row dataset path。
        feature_names: LightGBM モデルが要求する特徴量列名。

    Returns:
        必要列だけを持つ DataFrame。
    """
    metadata_columns = [
        "group_id",
        "session_id",
        "user_id",
        "turn_number",
        "track_id",
        "candidate_rank",
        "label",
    ]
    available = parquet_columns(path)
    if available is None:
        df = read_table(path)
    else:
        selected = [col for col in [*metadata_columns, *feature_names] if col in available]
        df = read_table(path, columns=selected)
    if "candidate_rank" in df.columns:
        df["candidate_rank"] = pd.to_numeric(df["candidate_rank"], errors="coerce").fillna(0).astype(np.int32)
    if "turn_number" in df.columns:
        df["turn_number"] = pd.to_numeric(df["turn_number"], errors="coerce").fillna(0).astype(np.int32)
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0.0).astype(np.float32)
    for feature_name in feature_names:
        if feature_name not in df.columns:
            df[feature_name] = np.nan
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce").astype(np.float32)
    return df


def score_frame(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM モデルで各行をスコアリングして lgbm_score 列を付加する。

    Args:
        model_path: 保存済みモデルファイルのパス。
        df: 候補行 DataFrame。

    Returns:
        lgbm_score 列が追加された DataFrame。
    """
    lgb = require_lightgbm()
    booster = lgb.Booster(model_file=model_path)
    feature_names = list(booster.feature_name())
    # モデルが記録した特徴量リストを基準にし、存在しない列は NaN で補完する（学習・推論間の特徴量ズレに対処）
    for feature_name in feature_names:
        if feature_name not in df.columns:
            df[feature_name] = np.nan
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
    scores = booster.predict(
        df[feature_names].to_numpy(dtype=np.float32, copy=False),
        num_iteration=booster.best_iteration,
    )
    keep_columns = [
        col
        for col in ["group_id", "session_id", "user_id", "turn_number", "track_id", "candidate_rank", "label"]
        if col in df.columns
    ]
    scored = df[keep_columns].copy()
    scored["lgbm_score"] = np.asarray(scores, dtype=np.float32)
    return scored


def to_predictions(df: pd.DataFrame, topk: int, response_text: str) -> list[dict[str, object]]:
    """スコア付き DataFrame を提出形式の dict リストに変換する。

    group_id ごとに lgbm_score 降順（同スコアは candidate_rank 昇順でタイブレーク）で
    上位 topk の track_id を抽出する。

    Args:
        df: lgbm_score・group_id・session_id・user_id・turn_number・
            track_id・candidate_rank 列を持つ DataFrame。
        topk: 返す上位件数。
        response_text: 全予測に共通で埋める応答テキスト。

    Returns:
        提出形式 dict のリスト（session_id / user_id / turn_number /
        predicted_track_ids / predicted_response）。
    """
    predictions: list[dict[str, object]] = []
    # 各 group は数千件規模なので、全体 sort ではなく group 内 sort にしてメモリを抑える。
    for _, group in df.groupby("group_id", sort=False):
        ranked = group.sort_values(["lgbm_score", "candidate_rank"], ascending=[False, True], kind="stable")
        predictions.append(
            {
                "session_id": str(ranked.iloc[0]["session_id"]),
                "user_id": str(ranked.iloc[0]["user_id"]),
                "turn_number": int(ranked.iloc[0]["turn_number"]),
                "predicted_track_ids": ranked["track_id"].astype(str).head(topk).tolist(),
                "predicted_response": response_text,
            }
        )
    return predictions


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    """score 順に並んだ label 配列から nDCG@k を計算する。

    Args:
        labels: score 降順に並んだ 0/1 label。
        k: 評価打ち切り位置。

    Returns:
        nDCG@k。正例なしの場合は 0.0。
    """
    if labels.size == 0 or labels.max() <= 0:
        return 0.0
    gains = labels[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum(ideal * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def grouped_rank_metrics(df: pd.DataFrame, ks: list[int]) -> dict[str, Any]:
    """scored candidate DataFrame から group 単位の ranking metric を計算する。

    Args:
        df: group_id / label / lgbm_score / candidate_rank 列を持つ DataFrame。
        ks: 評価する cutoff。

    Returns:
        positive group 条件付き metric と all-task zero-fill metric。
    """
    if "label" not in df.columns:
        return {}
    metrics = {f"ndcg@{k}": 0.0 for k in ks}
    metrics.update({f"recall@{k}": 0.0 for k in ks})
    total_groups = 0
    positive_groups = 0
    for _, group in df.groupby("group_id", sort=False):
        total_groups += 1
        ranked = group.sort_values(["lgbm_score", "candidate_rank"], ascending=[False, True], kind="stable")
        labels = ranked["label"].to_numpy(dtype=np.float32, copy=False)
        positives = float(labels.sum())
        if positives <= 0:
            continue
        for k in ks:
            metrics[f"ndcg@{k}"] += ndcg_at_k(labels, k)
            metrics[f"recall@{k}"] += float(labels[:k].sum()) / positives
        positive_groups += 1
    if positive_groups:
        for key in list(metrics):
            metrics[key] /= positive_groups
    scale = positive_groups / float(total_groups) if total_groups else 0.0
    all_tasks = {}
    for k in ks:
        all_tasks[f"all_tasks_ndcg@{k}"] = metrics[f"ndcg@{k}"] * scale
        all_tasks[f"all_tasks_recall@{k}"] = metrics[f"recall@{k}"] * scale
    return {
        "valid_groups_total": total_groups,
        "valid_groups_with_positive": positive_groups,
        "valid_ndcg@20_positive_groups": metrics.get("ndcg@20", 0.0),
        "all_tasks_ndcg@20": all_tasks.get("all_tasks_ndcg@20", 0.0),
        "all_tasks_recall@20": all_tasks.get("all_tasks_recall@20", 0.0),
        "valid": {**metrics, "groups_with_positive": float(positive_groups)},
        "valid_all_tasks": all_tasks,
    }


def main(args: argparse.Namespace) -> None:
    """候補データを読み込み → スコアリング → 予測 JSON 書き出しまでを実行する。

    Args:
        args: CLI 引数。モデル、候補データ、prediction 出力先、score 出力先、
            topk、固定応答文を含む。
    """
    lgb = require_lightgbm()
    booster = lgb.Booster(model_file=args.model)
    feature_names = list(booster.feature_name())
    df = read_candidate_table(args.candidate_data, feature_names)
    scored = score_frame(args.model, df)
    os.makedirs(os.path.dirname(args.prediction_output), exist_ok=True)
    predictions = to_predictions(scored, args.topk, args.response_text)
    with open(args.prediction_output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    if args.score_output:
        os.makedirs(os.path.dirname(args.score_output), exist_ok=True)
        if args.score_output.endswith(".parquet"):
            scored.to_parquet(args.score_output, index=False)
        elif args.score_output.endswith(".csv"):
            scored.to_csv(args.score_output, index=False)
        else:
            scored.to_json(args.score_output, orient="records", lines=True, force_ascii=False)
    if args.metrics_output:
        metrics = grouped_rank_metrics(scored, ks=[1, 10, 20])
        metrics.update(
            {
                "model": args.model,
                "candidate_data": args.candidate_data,
                "prediction_output": args.prediction_output,
                "rows": int(len(scored)),
                "prediction_rows": int(len(predictions)),
                "topk": int(args.topk),
            }
        )
        os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"wrote_predictions={len(predictions)} path={args.prediction_output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """apply_lgbm CLI の argparse parser を構築する。

    旧 exp014 パスの re-export shim からも同じ parser を呼べるように関数化している
    （shim の ``__main__`` は runpy を使わず ``build_parser`` 経由で本体 CLI を実行する）。

    Returns:
        apply_lgbm の CLI 引数を定義した parser。
    """
    parser = argparse.ArgumentParser(description="Apply exp014 LightGBM model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate_data", required=True)
    parser.add_argument("--prediction_output", required=True)
    parser.add_argument("--score_output", default=None)
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--response_text",
        default="Here are some tracks that match your request.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
