"""LightGBM 以外の GBDT ranker（XGBoost / CatBoost）と family blend のヘルパ。

reranker の learning-to-rank 部分を「同一の候補・同一の特徴量行列」を入力に取る複数
モデル族（family）の ensemble へ拡張するための共有モジュール。exp090-fix の
``FeatureBundle``（``x`` 特徴量行列 / ``y`` binary relevance / ``groups`` group sizes）と
同じ row レイアウトを受け取り、各 family が row ごとの score を返す。

ここに置く関数は **experiment フォルダや LightGBM 固有の型に依存しない**（numpy 配列と
group size list だけで完結する）。LightGBM family は既存 ``train_model`` /
``averaged_scores`` をそのまま使い、本モジュールは XGBoost / CatBoost wrapper と
family 横断の score blend のみを提供する。

blend は family ごとに score スケールが非互換（LambdaRank score / YetiRank score など）
なので、**group 内 z-score 正規化してから family 単位で等重み平均**する。group 内の
順位だけが nDCG に効くので z-score（group 内 affine 変換）は単一 family の順序を保ちつつ、
family 間を比較可能なスケールに揃える。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def groups_to_offsets(groups: list[int]) -> np.ndarray:
    """group size list を各 group の開始 row offset 配列へ変換する。

    Args:
        groups: 各 task の候補数（row 数）。``sum(groups)`` が全 row 数。

    Returns:
        長さ ``len(groups)+1`` の累積 offset 配列。``offsets[i]:offsets[i+1]`` が
        group ``i`` の row 範囲。
    """
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    np.cumsum(np.asarray(groups, dtype=np.int64), out=offsets[1:])
    return offsets


def groupwise_zscore(scores: np.ndarray, groups: list[int]) -> np.ndarray:
    """score を group ごとに z-score 正規化する（平均 0・分散 1）。

    group 内の順序を保つ affine 変換なので単一 family の nDCG は不変。family 間 blend の
    前段でスケールを揃えるために使う。group の分散が 0（候補 1 件や同点）の場合は 0 を返し、
    blend で中立になるようにする。

    Args:
        scores: row ごとの生 score。
        groups: 各 group の row 数。row は group 順に連続している前提。

    Returns:
        group 内 z-score 正規化済みの score 配列（``scores`` と同形）。
    """
    out = np.zeros_like(scores, dtype=np.float64)
    offsets = groups_to_offsets(groups)
    for start, end in zip(offsets[:-1], offsets[1:]):
        if end - start <= 1:
            # 候補 1 件の group は順位が自明なので 0（中立）にする。
            continue
        segment = scores[start:end].astype(np.float64)
        std = segment.std()
        if std <= 1e-12:
            # 全候補同点の group も中立。
            continue
        out[start:end] = (segment - segment.mean()) / std
    return out


def blend_family_scores(
    family_scores: dict[str, np.ndarray],
    groups: list[int],
    *,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """family ごとの score を group 内 z-score 正規化して重み付き平均する。

    各 family は内部で（seed 平均などにより）1 本の score にまとめられている前提。
    LightGBM が 6 seed でも family としては 1 票になり、XGBoost / CatBoost 各 1 票と
    対等に混ざる（per-model でなく per-family の等重み。advisor 指摘の「6 LGBM 票が
    XGB/CatBoost を埋没させる」罠を避ける）。

    Args:
        family_scores: family 名 -> row score。全 family が同じ row レイアウト。
        groups: 各 group の row 数。
        weights: family 名 -> 重み。省略時は全 family 等重み。Dev OOF で tune せず
            等重みを既定にする（CV-LB 逆相関で blend 重み tune は overfit しやすい）。

    Returns:
        blend 済み score 配列。``score_to_predictions`` にそのまま渡せる。

    Raises:
        ValueError: family_scores が空の場合。
    """
    if not family_scores:
        raise ValueError("family_scores must not be empty")
    if weights is None:
        weights = {name: 1.0 for name in family_scores}
    total_weight = sum(weights[name] for name in family_scores)
    if total_weight <= 0:
        raise ValueError("sum of family weights must be positive")
    blended = np.zeros(next(iter(family_scores.values())).shape[0], dtype=np.float64)
    for name, scores in family_scores.items():
        blended += weights[name] * groupwise_zscore(scores, groups)
    return blended / total_weight


# --------------------------------------------------------------------------- #
# XGBoost LambdaRank family
# --------------------------------------------------------------------------- #
def xgb_params(*, seed: int, num_threads: int, device: str = "cuda") -> dict[str, Any]:
    """LightGBM 設定に概ね対応づけた XGBoost LambdaRank パラメータを返す。

    exp090-fix の LightGBM（num_leaves=31≈max_depth6 / lr0.03 / subsample0.9 /
    colsample0.9 / lambda_l2=8）に寄せた値。ranking 目的は ``rank:ndcg``。

    Args:
        seed: この model の乱数 seed。
        num_threads: CPU thread 数（GPU 時もデータ前処理に使う）。
        device: ``"cuda"`` で GPU、``"cpu"`` で CPU。

    Returns:
        ``xgboost.train`` 用 param 辞書。
    """
    return {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@20",
        "eta": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 8.0,
        "tree_method": "hist",
        "device": device,
        "seed": seed,
        "nthread": num_threads,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 20,
        "verbosity": 0,
    }


def fit_xgb_ranker(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    *,
    params: dict[str, Any],
    num_boost_round: int,
    feature_name: list[str] | None = None,
) -> Any:
    """XGBoost LambdaRank ranker を学習して Booster を返す。

    Args:
        x: 特徴量行列（n_rows × n_features）。
        y: binary relevance ラベル。
        groups: 各 group の row 数（row は group 順に連続）。
        params: ``xgb_params`` 等の param 辞書。
        num_boost_round: boosting round 数。
        feature_name: feature 名（任意。importance 解釈用）。

    Returns:
        学習済み ``xgboost.Booster``。
    """
    import xgboost as xgb

    dtrain = xgb.DMatrix(x, label=y, feature_names=feature_name)
    dtrain.set_group(groups)
    return xgb.train(params, dtrain, num_boost_round=num_boost_round)


def predict_xgb(model: Any, x: np.ndarray, *, feature_name: list[str] | None = None) -> np.ndarray:
    """XGBoost Booster で row score を予測する。

    Args:
        model: 学習済み Booster。
        x: 採点する特徴量行列。
        feature_name: 学習時と同じ feature 名（DMatrix の整合用）。

    Returns:
        row ごとの score 配列。
    """
    import xgboost as xgb

    return np.asarray(model.predict(xgb.DMatrix(x, feature_names=feature_name)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# CatBoost YetiRank family
# --------------------------------------------------------------------------- #
def _group_ids(groups: list[int]) -> np.ndarray:
    """group size list を row ごとの group_id 配列へ展開する。

    CatBoost の ``Pool`` は row ごとの group_id を要求し、同一 group は連続している必要が
    ある（build_features は key 順に連続生成するので満たす）。

    Args:
        groups: 各 group の row 数。

    Returns:
        row ごとの整数 group_id 配列。
    """
    return np.repeat(np.arange(len(groups), dtype=np.int64), np.asarray(groups, dtype=np.int64))


def catboost_params(*, seed: int, num_threads: int, task_type: str = "GPU") -> dict[str, Any]:
    """LightGBM 設定に概ね対応づけた CatBoost YetiRank パラメータを返す。

    YetiRank は listwise ranking loss で nDCG 最適化に向く。GPU(task_type='GPU') では
    bitwise 決定性は保証されないが、family ensemble で平均するため許容する（必要なら
    ``task_type='CPU'`` で決定的に学習できる）。

    Args:
        seed: 乱数 seed。
        num_threads: CPU thread 数。
        task_type: ``"GPU"`` か ``"CPU"``。

    Returns:
        ``CatBoostRanker`` 用 param 辞書。
    """
    params: dict[str, Any] = {
        "loss_function": "YetiRank",
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "task_type": task_type,
        "verbose": False,
        "allow_writing_files": False,
    }
    if task_type == "GPU":
        params["devices"] = "0"
    else:
        params["thread_count"] = num_threads
    return params


def fit_catboost_ranker(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    *,
    params: dict[str, Any],
    feature_name: list[str] | None = None,
) -> Any:
    """CatBoost YetiRank ranker を学習して返す。

    Args:
        x: 特徴量行列。
        y: binary relevance ラベル。
        groups: 各 group の row 数（row は group 順に連続）。
        params: ``catboost_params`` 等の param 辞書（``iterations`` を含む）。
        feature_name: feature 名（任意）。

    Returns:
        学習済み ``CatBoostRanker``。
    """
    import catboost as cb

    pool = cb.Pool(
        data=x,
        label=y.astype(np.float32),
        group_id=_group_ids(groups),
        feature_names=list(feature_name) if feature_name else None,
    )
    model = cb.CatBoostRanker(**params)
    model.fit(pool)
    return model


def predict_catboost(model: Any, x: np.ndarray) -> np.ndarray:
    """CatBoostRanker で row score を予測する。

    Args:
        model: 学習済み CatBoostRanker。
        x: 採点する特徴量行列。

    Returns:
        row ごとの score 配列。
    """
    return np.asarray(model.predict(x), dtype=np.float64)
