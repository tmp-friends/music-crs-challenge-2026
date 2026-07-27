"""exp015 LightGBM モデルを候補 row に適用して prediction を作る。"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def require_lightgbm():
    """lightgbm をインポートして返す。未インストール時は分かりやすいエラーで即終了する。"""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is not installed. Install it with `uv pip install lightgbm`."
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
    df = df.copy()
    df["lgbm_score"] = booster.predict(df[feature_names], num_iteration=booster.best_iteration)
    return df


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
    # lgbm_score 降順・同スコアは candidate_rank 昇順でタイブレークする
    sort_cols = ["group_id", "lgbm_score", "candidate_rank"]
    ascending = [True, False, True]
    for _, group in df.sort_values(sort_cols, ascending=ascending, kind="stable").groupby("group_id", sort=False):
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


def main(args: argparse.Namespace) -> None:
    """候補データを読み込み → スコアリング → 予測 JSON 書き出しまでを実行する。

    Args:
        args: CLI 引数。モデル、候補データ、prediction 出力先、score 出力先、
            topk、固定応答文を含む。
    """
    df = read_table(args.candidate_data)
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
    print(f"wrote_predictions={len(predictions)} path={args.prediction_output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply exp015 LightGBM model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate_data", required=True)
    parser.add_argument("--prediction_output", required=True)
    parser.add_argument("--score_output", default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--response_text",
        default="Here are some tracks that match your request.",
    )
    main(parser.parse_args())
