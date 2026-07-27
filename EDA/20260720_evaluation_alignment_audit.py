"""Music-CRS 論文向けに評価信号の整合性を再監査する。

本スクリプトは、論文 ``paper_evaluation_alignment_audit.md`` の根拠を強化するため、
次の二つを再現可能に集計する。

1. ``music[t-1]``、``goal_progress[t]``／user message ``t``、``music[t]`` の
   時系列に沿った artist-pivot 分析、turn直接標準化、session-clustered
   bootstrap、turn fixed effects logistic回帰。
2. Blind-A の採点済み response artifact に対する composite 分解、response 特徴の
   記述的相関、および同一 prediction 再提出時の judge 変動。

``goal_progress`` と future music row は分析専用の公開 train annotation である。
推薦・応答生成・validation・fallback には一切流用しない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
from datasets import load_dataset

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
METADATA_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
MOVES = "MOVES_TOWARD_GOAL"
DONT = "DOES_NOT_MOVE_TOWARD_GOAL"
REPO_ROOT = Path(__file__).resolve().parents[1]

# 既存 EDA と同じ定義。artist 明示を要求しないため recall 寄りの discovery slice である。
BROAD_PIVOT_RE = re.compile(
    r"(different artist|another artist|someone (?:else|new)|new artists?|"
    r"haven'?t heard|never heard|artist i (?:haven'?t|have not|don'?t know)|"
    r"by (?:a|an) (?:other|different|new)|other than|"
    r"something (?:else|new|different)|discover (?:new|different|some|something))",
    re.I,
)

# artist/band 等または人物を指す変更表現を対象とする、artist-specific 主分析。
EXPLICIT_ARTIST_PIVOT_RE = re.compile(
    r"(?:\b(?:different|another|new|other)\s+"
    r"(?:artist|artists|band|bands|singer|singers|musician|musicians)\b|"
    r"\b(?:artist|band|singer|musician)\s+i\s+"
    r"(?:haven'?t|have not|don'?t)\b|"
    r"\bsomeone\s+(?:else|new)\b|"
    r"\bnot\s+(?:the\s+)?same\s+(?:artist|band|singer|musician)\b|"
    r"\bby\s+(?:someone\s+(?:else|new)|(?:a|an)\s+(?:other|different|new)\s+"
    r"(?:artist|band|singer|musician))\b)",
    re.I,
)

# novelty 一般ではなく、明示的な変更要求を中心にした代替定義（他定義の部分集合ではない）。
CORE_CHANGE_PIVOT_RE = re.compile(
    r"(?:\bdifferent\s+(?:artist|artists|band|bands|singer|singers|musician|musicians)\b|"
    r"\banother\s+(?:artist|band|singer|musician)\b|"
    r"\bsomeone\s+else\b|"
    r"\bnot\s+(?:the\s+)?same\s+(?:artist|band|singer|musician)\b|"
    r"\bby\s+(?:someone\s+else|(?:a|an)\s+(?:other|different)\s+"
    r"(?:artist|band|singer|musician))\b)",
    re.I,
)

PIVOT_DEFINITIONS = {
    "broad_discovery": BROAD_PIVOT_RE,
    "artist_explicit": EXPLICIT_ARTIST_PIVOT_RE,
    "core_change": CORE_CHANGE_PIVOT_RE,
}
PIVOT_DISPLAY_NAMES = {
    "broad_discovery": "broad discovery/change cue",
    "artist_explicit": "artist-targeted (primary)",
    "core_change": "core change cue",
}
TURN_NUMBERS = tuple(range(2, 9))
NORMAL_975 = 1.959963984540054

RESPONSE_VARIANTS = {
    "submission_exp058_wide100_ens6_A.zip": "Initial response",
    "submission_exp063_v2_qwen4b_A.zip": "Qwen 4B, revised",
    "submission_exp063_v2_claude_A.zip": "Claude, short",
    "submission_exp063_v2_claude_long_A.zip": "Claude, grounded long",
    "submission_exp064_v2_gemini_pro_A.zip": "Gemini Pro, grounded",
}


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。

    Returns:
        出力先、bootstrap 回数、乱数 seed、session 上限を持つ Namespace。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="出力 root。既定は EDA/。",
    )
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=5000,
        help="session-clustered bootstrap の反復数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
        help="bootstrap の乱数 seed。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭 N session に限定する smoke 用。未指定時は full train。",
    )
    return parser.parse_args()


def normalize_artists(value: Any) -> frozenset[str]:
    """metadata の artist 表現を正規化する。

    Args:
        value: ``artist_name`` の文字列または文字列リスト。

    Returns:
        小文字・strip 済み artist 名の集合。
    """
    if value is None:
        return frozenset()
    if isinstance(value, list):
        return frozenset(str(item).strip().lower() for item in value if item)
    return frozenset([str(value).strip().lower()])


def load_artist_index() -> dict[str, frozenset[str]]:
    """公式 metadata から track_id→artist 集合を構築する。

    Returns:
        全 catalog track の artist lookup。
    """
    metadata = load_dataset(METADATA_NAME)["all_tracks"]
    return {
        row["track_id"]: normalize_artists(row.get("artist_name")) for row in metadata
    }


def build_turn_frame(
    track_to_artists: dict[str, frozenset[str]], limit: int | None
) -> pd.DataFrame:
    """train session を逐次監査用の turn DataFrame へ変換する。

    ``goal_progress[t]`` は、TalkPlayData 2 の生成順に従い、listener が
    ``music[t-1]`` を受け取った後に生成する feedback として扱う。

    Args:
        track_to_artists: track_id→artist 集合。
        limit: 先頭から読む session 数。``None`` は full train。

    Returns:
        pivot、現在・直前の artist 継続、progress label を持つ DataFrame。
    """
    dataset = load_dataset(DATASET_NAME, split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records: list[dict[str, Any]] = []
    for session in dataset:
        by_turn: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in session["conversations"]:
            by_turn[int(row["turn_number"])][row["role"]] = row
        progress_by_turn = {
            int(row["turn_number"]): row.get("goal_progress_assessment")
            for row in session["goal_progress_assessments"]
        }

        cumulative_artists: set[str] = set()
        previous_artists: frozenset[str] = frozenset()
        previous_same_cumulative: bool | None = None

        for turn_number in sorted(by_turn):
            turn = by_turn[turn_number]
            music = turn.get("music") or {}
            user = turn.get("user") or {}
            current_artists = track_to_artists.get(
                str(music.get("content") or ""), frozenset()
            )
            user_text = str(user.get("content") or "")
            current_same_cumulative = (
                bool(current_artists & cumulative_artists)
                if current_artists and cumulative_artists
                else None
            )
            current_same_adjacent = (
                bool(current_artists & previous_artists)
                if current_artists and previous_artists
                else None
            )
            progress = progress_by_turn.get(turn_number)
            moves = 1 if progress == MOVES else (0 if progress == DONT else None)

            record: dict[str, Any] = {
                "session_id": session["session_id"],
                "turn_number": turn_number,
                "moves": moves,
                "current_same_cumulative": current_same_cumulative,
                "current_same_adjacent": current_same_adjacent,
                "previous_same_cumulative": previous_same_cumulative,
            }
            for name, pattern in PIVOT_DEFINITIONS.items():
                record[f"pivot_{name}"] = bool(pattern.search(user_text))
            records.append(record)

            # 次の listener turn から見れば、この music が feedback 対象になる。
            previous_same_cumulative = current_same_cumulative
            previous_artists = current_artists
            cumulative_artists |= current_artists

    return pd.DataFrame(records)


def _put_grouped(
    matrix: np.ndarray,
    session_positions: dict[str, int],
    frame: pd.DataFrame,
    mask: pd.Series,
    column: int,
    value_column: str | None = None,
) -> None:
    """条件に合う turn の session 別十分統計を行列へ格納する。

    Args:
        matrix: session×統計量の出力行列。
        session_positions: session_id→行番号。
        frame: turn DataFrame。
        mask: 集計対象を示す bool Series。
        column: 書き込み先列。
        value_column: ``None`` なら件数、指定時は列の合計を格納する。
    """
    selected = frame.loc[mask]
    if value_column is None:
        grouped = selected.groupby("session_id").size()
    else:
        grouped = selected.groupby("session_id")[value_column].sum()
    positions = [session_positions[str(session_id)] for session_id in grouped.index]
    matrix[positions, column] = grouped.to_numpy(dtype=float)


def build_cluster_statistics(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, slice]]:
    """pivot 定義ごとの session-level 十分統計を構築する。

    Args:
        frame: ``build_turn_frame`` の出力。

    Returns:
        統計行列、session_id 一覧、pivot 定義ごとの列 slice。
    """
    session_ids = sorted(str(value) for value in frame["session_id"].unique())
    positions = {session_id: index for index, session_id in enumerate(session_ids)}
    columns_per_definition = 14
    matrix = np.zeros(
        (len(session_ids), columns_per_definition * len(PIVOT_DEFINITIONS)),
        dtype=float,
    )
    slices: dict[str, slice] = {}

    current_valid = frame["moves"].notna() & frame["current_same_cumulative"].notna()
    for definition_index, name in enumerate(PIVOT_DEFINITIONS):
        start = definition_index * columns_per_definition
        slices[name] = slice(start, start + columns_per_definition)
        pivot = frame[f"pivot_{name}"] & current_valid
        nonpivot = ~frame[f"pivot_{name}"] & current_valid
        lagged = pivot & frame["previous_same_cumulative"].notna()
        prior_same = lagged & frame["previous_same_cumulative"].eq(True)
        prior_different = lagged & frame["previous_same_cumulative"].eq(False)
        conflict = prior_same & frame["moves"].eq(0)

        masks_and_values = [
            (pivot, None),
            (pivot, "current_same_cumulative"),
            (pivot, "current_same_adjacent"),
            (nonpivot, None),
            (nonpivot, "current_same_cumulative"),
            (lagged, None),
            (lagged, "previous_same_cumulative"),
            (prior_same, None),
            (prior_same, "moves"),
            (prior_different, None),
            (prior_different, "moves"),
            (conflict, None),
            (conflict, "current_same_cumulative"),
            (conflict, "current_same_adjacent"),
        ]
        for offset, (mask, value_column) in enumerate(masks_and_values):
            _put_grouped(
                matrix,
                positions,
                frame,
                mask,
                start + offset,
                value_column,
            )
    return matrix, session_ids, slices


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """0 除算を NaN とする要素ごとの比率を返す。

    Args:
        numerator: 分子。
        denominator: 分母。

    Returns:
        ``numerator / denominator``。分母0は NaN。
    """
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def metrics_from_totals(totals: np.ndarray) -> dict[str, np.ndarray]:
    """14列の十分統計から論文用の率と差を計算する。

    Args:
        totals: ``(..., 14)`` 形状の集約統計。

    Returns:
        指標名→配列の辞書。
    """
    pivot_same = _safe_ratio(totals[..., 1], totals[..., 0])
    pivot_adjacent = _safe_ratio(totals[..., 2], totals[..., 0])
    nonpivot_same = _safe_ratio(totals[..., 4], totals[..., 3])
    prior_same_share = _safe_ratio(totals[..., 6], totals[..., 5])
    prior_same_moves = _safe_ratio(totals[..., 8], totals[..., 7])
    prior_different_moves = _safe_ratio(totals[..., 10], totals[..., 9])
    conflict_same = _safe_ratio(totals[..., 12], totals[..., 11])
    conflict_adjacent = _safe_ratio(totals[..., 13], totals[..., 11])
    return {
        "next_same_rate": pivot_same,
        "next_adjacent_rate": pivot_adjacent,
        "nonpivot_next_same_rate": nonpivot_same,
        "pivot_minus_nonpivot_same_gap": pivot_same - nonpivot_same,
        "prior_same_rate": prior_same_share,
        "prior_same_moves_rate": prior_same_moves,
        "prior_different_moves_rate": prior_different_moves,
        "prior_moves_gap": prior_different_moves - prior_same_moves,
        "conflict_next_same_rate": conflict_same,
        "conflict_next_adjacent_rate": conflict_adjacent,
    }


def bootstrap_pivot_analysis(
    matrix: np.ndarray,
    slices: dict[str, slice],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """session を再標本化して pivot 指標の95%区間を求める。

    Args:
        matrix: session-level 十分統計。
        slices: pivot 定義ごとの列 slice。
        replicates: bootstrap 反復数。
        seed: 乱数 seed。

    Returns:
        point estimate と95%区間の表、および bootstrap 標本。

    Raises:
        ValueError: 反復数が正でない場合。
    """
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    point_totals = matrix.sum(axis=0, keepdims=True)
    bootstrap_values: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in PIVOT_DEFINITIONS
    }
    rng = np.random.default_rng(seed)
    probabilities = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    batch_size = 100

    # turn 単位ではなく session を再標本化し、同一対話内 turn の相関を保つ。
    for batch_start in range(0, replicates, batch_size):
        current_batch = min(batch_size, replicates - batch_start)
        weights = rng.multinomial(
            matrix.shape[0], probabilities, size=current_batch
        )
        totals = weights @ matrix
        for name, column_slice in slices.items():
            metrics = metrics_from_totals(totals[:, column_slice])
            for metric_name, values in metrics.items():
                bootstrap_values[name][metric_name].append(values)

    rows: list[dict[str, Any]] = []
    flattened: dict[str, dict[str, np.ndarray]] = {}
    for name, column_slice in slices.items():
        totals = point_totals[:, column_slice]
        point_metrics = metrics_from_totals(totals)
        flattened[name] = {
            metric_name: np.concatenate(parts)
            for metric_name, parts in bootstrap_values[name].items()
        }
        row: dict[str, Any] = {
            "pivot_definition": name,
            "pivot_turns": int(totals[0, 0]),
            "lagged_pivot_turns": int(totals[0, 5]),
            "prior_same_turns": int(totals[0, 7]),
            "prior_different_turns": int(totals[0, 9]),
            "conflict_turns": int(totals[0, 11]),
        }
        for metric_name, point_value in point_metrics.items():
            samples = flattened[name][metric_name]
            low, high = np.nanquantile(samples, [0.025, 0.975])
            row[metric_name] = float(point_value[0])
            row[f"{metric_name}_ci_low"] = float(low)
            row[f"{metric_name}_ci_high"] = float(high)
        rows.append(row)
    return pd.DataFrame(rows), flattened


def pivot_by_turn(frame: pd.DataFrame) -> pd.DataFrame:
    """pivot 定義・turn 別に pivot／non-pivot の artist 継続率を集計する。

    Args:
        frame: turn DataFrame。

    Returns:
        pivot 定義×turn の件数、継続率、pivot-minus-non-pivot 差。
    """
    rows: list[dict[str, Any]] = []
    current_valid = frame["moves"].notna() & frame["current_same_cumulative"].notna()
    for name in PIVOT_DEFINITIONS:
        for turn_number in TURN_NUMBERS:
            selected = frame[
                current_valid & frame["turn_number"].eq(turn_number)
            ]
            pivot = selected[selected[f"pivot_{name}"]]
            nonpivot = selected[~selected[f"pivot_{name}"]]
            if pivot.empty or nonpivot.empty:
                continue
            pivot_same = float(pivot["current_same_cumulative"].astype(bool).mean())
            nonpivot_same = float(
                nonpivot["current_same_cumulative"].astype(bool).mean()
            )
            pivot_adjacent = float(
                pivot["current_same_adjacent"].astype(bool).mean()
            )
            nonpivot_adjacent = float(
                nonpivot["current_same_adjacent"].astype(bool).mean()
            )
            rows.append(
                {
                    "pivot_definition": name,
                    "pivot_display_name": PIVOT_DISPLAY_NAMES[name],
                    "turn_number": turn_number,
                    "pivot_turns": int(len(pivot)),
                    "nonpivot_turns": int(len(nonpivot)),
                    "pivot_next_same_rate": pivot_same,
                    "nonpivot_next_same_rate": nonpivot_same,
                    "same_risk_difference": pivot_same - nonpivot_same,
                    "pivot_next_adjacent_rate": pivot_adjacent,
                    "nonpivot_next_adjacent_rate": nonpivot_adjacent,
                    "adjacent_risk_difference": pivot_adjacent - nonpivot_adjacent,
                }
            )
    return pd.DataFrame(rows)


def build_turn_standardization_statistics(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, slice]]:
    """turn 標準化bootstrap用のsession別十分統計を構築する。

    detector、turn、pivot有無ごとに件数、累積artist反復数、直前artist反復数を
    session単位で保持する。これにより同一session内turnの依存を保ったまま
    cluster bootstrapできる。

    Args:
        frame: ``build_turn_frame`` の出力。

    Returns:
        session×十分統計行列、session_id一覧、detectorごとの列slice。
    """
    session_ids = sorted(str(value) for value in frame["session_id"].unique())
    positions = {session_id: index for index, session_id in enumerate(session_ids)}
    measures_per_group = 3
    groups_per_turn = 2
    columns_per_definition = (
        len(TURN_NUMBERS) * groups_per_turn * measures_per_group
    )
    matrix = np.zeros(
        (len(session_ids), columns_per_definition * len(PIVOT_DEFINITIONS)),
        dtype=float,
    )
    slices: dict[str, slice] = {}
    current_valid = frame["moves"].notna() & frame["current_same_cumulative"].notna()

    for definition_index, name in enumerate(PIVOT_DEFINITIONS):
        start = definition_index * columns_per_definition
        slices[name] = slice(start, start + columns_per_definition)
        for turn_index, turn_number in enumerate(TURN_NUMBERS):
            in_turn = current_valid & frame["turn_number"].eq(turn_number)
            for group_index, is_pivot in enumerate((True, False)):
                group_mask = in_turn & frame[f"pivot_{name}"].eq(is_pivot)
                offset = (
                    (turn_index * groups_per_turn + group_index)
                    * measures_per_group
                )
                for measure_index, value_column in enumerate(
                    (None, "current_same_cumulative", "current_same_adjacent")
                ):
                    _put_grouped(
                        matrix,
                        positions,
                        frame,
                        group_mask,
                        start + offset + measure_index,
                        value_column,
                    )
    return matrix, session_ids, slices


def standardized_metrics_from_totals(
    totals: np.ndarray,
    turn_weights: np.ndarray,
    common_turn_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """turn別十分統計から直接標準化率とrisk differenceを計算する。

    Args:
        totals: ``(..., turn, pivot_group, measure)`` 形状の十分統計。
            pivot_groupは0がpivot、1がnon-pivot、measureは件数、累積反復数、
            直前反復数の順。
        turn_weights: 標準母集団におけるturn分布。
        common_turn_mask: pivot／non-pivot双方に観測があるturn。

    Returns:
        累積artist反復と直前artist反復の標準化率・risk difference。
    """
    counts = totals[..., 0]
    cumulative_rates = _safe_ratio(totals[..., 1], counts)
    adjacent_rates = _safe_ratio(totals[..., 2], counts)
    selected_weights = turn_weights[common_turn_mask]

    def standardize(rates: np.ndarray, group_index: int) -> np.ndarray:
        """共通support上のturn別率を固定標準分布で平均する。"""
        selected = rates[..., common_turn_mask, group_index]
        complete = np.all(np.isfinite(selected), axis=-1)
        values = np.sum(selected * selected_weights, axis=-1)
        return np.where(complete, values, np.nan)

    pivot_same = standardize(cumulative_rates, 0)
    nonpivot_same = standardize(cumulative_rates, 1)
    pivot_adjacent = standardize(adjacent_rates, 0)
    nonpivot_adjacent = standardize(adjacent_rates, 1)
    return {
        "standardized_pivot_same_rate": pivot_same,
        "standardized_nonpivot_same_rate": nonpivot_same,
        "standardized_same_risk_difference": pivot_same - nonpivot_same,
        "standardized_pivot_adjacent_rate": pivot_adjacent,
        "standardized_nonpivot_adjacent_rate": nonpivot_adjacent,
        "standardized_adjacent_risk_difference": (
            pivot_adjacent - nonpivot_adjacent
        ),
    }


def bootstrap_turn_standardized_analysis(
    matrix: np.ndarray,
    slices: dict[str, slice],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """session cluster bootstrapでturn標準化risk differenceを推定する。

    標準分布には全eligible turnの観測分布を固定して用いる。bootstrap標本ごとに
    pivot／non-pivotのturn別率だけを再推定するため、後半turnにpivotが多いという
    構成差をrisk differenceから除く。

    Args:
        matrix: ``build_turn_standardization_statistics`` の十分統計行列。
        slices: detectorごとの列slice。
        replicates: bootstrap反復数。
        seed: 乱数seed。

    Returns:
        detector別の標準化率、risk difference、95% percentile区間。

    Raises:
        ValueError: bootstrap反復数が正でない、または共通turn supportがない場合。
    """
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    point_totals = matrix.sum(axis=0)
    rng = np.random.default_rng(seed)
    probabilities = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    bootstrap_values: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in PIVOT_DEFINITIONS
    }
    configurations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: dict[str, dict[str, Any]] = {}

    for name, column_slice in slices.items():
        totals = point_totals[column_slice].reshape(len(TURN_NUMBERS), 2, 3)
        counts = totals[..., 0]
        common_turn_mask = np.all(counts > 0, axis=1)
        if not np.any(common_turn_mask):
            raise ValueError(f"no common turn support for {name}")
        standard_counts = counts.sum(axis=1)
        turn_weights = np.zeros(len(TURN_NUMBERS), dtype=float)
        turn_weights[common_turn_mask] = (
            standard_counts[common_turn_mask]
            / standard_counts[common_turn_mask].sum()
        )
        configurations[name] = (turn_weights, common_turn_mask)
        point_metrics = standardized_metrics_from_totals(
            totals[np.newaxis, ...], turn_weights, common_turn_mask
        )
        rows[name] = {
            "pivot_definition": name,
            "pivot_display_name": PIVOT_DISPLAY_NAMES[name],
            "standard_population_n": int(standard_counts[common_turn_mask].sum()),
            "pivot_turns": int(counts[common_turn_mask, 0].sum()),
            "nonpivot_turns": int(counts[common_turn_mask, 1].sum()),
            "standardized_turns": ";".join(
                str(turn)
                for turn, included in zip(TURN_NUMBERS, common_turn_mask, strict=True)
                if included
            ),
        }
        for metric_name, value in point_metrics.items():
            rows[name][metric_name] = float(value[0])

    batch_size = 100
    for batch_start in range(0, replicates, batch_size):
        current_batch = min(batch_size, replicates - batch_start)
        weights = rng.multinomial(
            matrix.shape[0], probabilities, size=current_batch
        )
        bootstrap_totals = weights @ matrix
        for name, column_slice in slices.items():
            totals = bootstrap_totals[:, column_slice].reshape(
                current_batch, len(TURN_NUMBERS), 2, 3
            )
            turn_weights, common_turn_mask = configurations[name]
            metrics = standardized_metrics_from_totals(
                totals, turn_weights, common_turn_mask
            )
            for metric_name, values in metrics.items():
                bootstrap_values[name][metric_name].append(values)

    output_rows: list[dict[str, Any]] = []
    for name in PIVOT_DEFINITIONS:
        row = rows[name]
        for metric_name, parts in bootstrap_values[name].items():
            samples = np.concatenate(parts)
            low, high = np.nanquantile(samples, [0.025, 0.975])
            row[f"{metric_name}_ci_low"] = float(low)
            row[f"{metric_name}_ci_high"] = float(high)
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def _logistic_probabilities(linear_predictor: np.ndarray) -> np.ndarray:
    """overflowを避けてlogistic確率を計算する。"""
    clipped = np.clip(linear_predictor, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_cluster_robust_logistic(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """logistic回帰をIRLSで推定しsession-cluster robust covarianceを返す。

    外部統計packageへの依存を追加せず、通常のBernoulli scoreに対する
    cluster sandwich covarianceを計算する。有限標本補正には
    ``G/(G-1) * (N-1)/(N-K)`` を用いる。

    Args:
        design: 切片を含む ``N×K`` design matrix。
        outcome: 0/1 outcome。
        clusters: 各観測のcluster識別子。
        max_iterations: Newton更新の最大反復数。
        tolerance: 係数更新量の収束閾値。

    Returns:
        係数、cluster-robust covariance、実行反復数。

    Raises:
        RuntimeError: 最大反復までに収束しない場合。
        ValueError: cluster数や自由度がsandwich推定に不足する場合。
    """
    beta = np.zeros(design.shape[1], dtype=float)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        probabilities = _logistic_probabilities(design @ beta)
        variances = np.clip(probabilities * (1.0 - probabilities), 1e-12, None)
        information = design.T @ (design * variances[:, np.newaxis])
        score = design.T @ (outcome - probabilities)
        step = np.linalg.solve(information, score)
        beta += step
        if float(np.max(np.abs(step))) < tolerance:
            break
    else:
        raise RuntimeError("cluster-robust logistic regression did not converge")

    probabilities = _logistic_probabilities(design @ beta)
    variances = np.clip(probabilities * (1.0 - probabilities), 1e-12, None)
    information = design.T @ (design * variances[:, np.newaxis])
    bread = np.linalg.inv(information)
    unique_clusters, cluster_codes = np.unique(clusters, return_inverse=True)
    n_observations, n_parameters = design.shape
    n_clusters = len(unique_clusters)
    if n_clusters <= 1 or n_observations <= n_parameters:
        raise ValueError("insufficient clusters or residual degrees of freedom")

    observation_scores = design * (outcome - probabilities)[:, np.newaxis]
    cluster_scores = np.zeros((n_clusters, n_parameters), dtype=float)
    np.add.at(cluster_scores, cluster_codes, observation_scores)
    meat = cluster_scores.T @ cluster_scores
    correction = (
        n_clusters
        / (n_clusters - 1)
        * (n_observations - 1)
        / (n_observations - n_parameters)
    )
    covariance = correction * (bread @ meat @ bread)
    return beta, covariance, iterations


def turn_fixed_effect_logistic_table(frame: pd.DataFrame) -> pd.DataFrame:
    """pivot効果をturn fixed effects付きlogistic回帰で補助確認する。

    Args:
        frame: ``build_turn_frame`` の出力。

    Returns:
        detector・outcome別の係数、session-cluster robust SE、odds ratio。
    """
    rows: list[dict[str, Any]] = []
    outcome_columns = {
        "cumulative_repeat": "current_same_cumulative",
        "adjacent_repeat": "current_same_adjacent",
    }
    for name in PIVOT_DEFINITIONS:
        for outcome_name, outcome_column in outcome_columns.items():
            selected = frame[
                frame["moves"].notna() & frame[outcome_column].notna()
            ].copy()
            turn_dummies = [
                selected["turn_number"].eq(turn).to_numpy(dtype=float)
                for turn in TURN_NUMBERS[1:]
            ]
            design = np.column_stack(
                [
                    np.ones(len(selected), dtype=float),
                    selected[f"pivot_{name}"].to_numpy(dtype=float),
                    *turn_dummies,
                ]
            )
            outcome = selected[outcome_column].to_numpy(dtype=float)
            clusters = selected["session_id"].astype(str).to_numpy()
            beta, covariance, iterations = fit_cluster_robust_logistic(
                design, outcome, clusters
            )
            coefficient = float(beta[1])
            robust_se = float(np.sqrt(covariance[1, 1]))
            ci_low = coefficient - NORMAL_975 * robust_se
            ci_high = coefficient + NORMAL_975 * robust_se
            z_score = coefficient / robust_se
            p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
            rows.append(
                {
                    "pivot_definition": name,
                    "pivot_display_name": PIVOT_DISPLAY_NAMES[name],
                    "outcome": outcome_name,
                    "n_observations": int(len(selected)),
                    "n_sessions": int(selected["session_id"].nunique()),
                    "pivot_coefficient": coefficient,
                    "cluster_robust_se": robust_se,
                    "odds_ratio": float(np.exp(coefficient)),
                    "odds_ratio_ci_low": float(np.exp(ci_low)),
                    "odds_ratio_ci_high": float(np.exp(ci_high)),
                    "two_sided_p_value": p_value,
                    "iterations": iterations,
                }
            )
    return pd.DataFrame(rows)


def read_prediction(path: Path) -> list[dict[str, Any]]:
    """JSON または submission ZIP から prediction records を読む。

    Args:
        path: prediction JSON または ZIP。

    Returns:
        prediction record のリスト。

    Raises:
        ValueError: JSON root が list でない場合。
    """
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            payload = json.loads(archive.read("prediction.json"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"prediction root must be list: {path}")
    return payload


def resolve_artifact(raw_path: str) -> Path | None:
    """ledger の artifact 表記をローカルファイルへ解決する。

    Args:
        raw_path: ledger に記録された相対または絶対パス。

    Returns:
        読み取り可能な artifact path。見つからなければ ``None``。
    """
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        return path

    blind_root = REPO_ROOT / "exp/inference/blindset_A"
    basename = Path(raw_path).name
    candidates = [blind_root / basename]
    if not Path(basename).suffix:
        candidates.extend(
            [blind_root / f"{basename}.zip", blind_root / f"{basename}.json"]
        )
    if basename.startswith("submission_") and basename.endswith(".zip"):
        stem = basename.removeprefix("submission_").removesuffix(".zip")
        candidates.append(blind_root / f"{stem}.json")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def response_features(texts: list[str]) -> dict[str, float]:
    """response corpus の解釈可能な表層特徴を計算する。

    Args:
        texts: submission 内の応答文字列。

    Returns:
        corpus-level response feature。
    """
    tokenized = [re.findall(r"\b[\w']+\b", text) for text in texts]
    openings = {
        " ".join(tokens[:5]).lower() for tokens in tokenized if len(tokens) > 0
    }
    apology_pattern = re.compile(
        r"\b(?:sorry|apolog(?:y|ize|ise|etic)|regret|unfortunately)\b", re.I
    )
    return {
        "mean_words": float(np.mean([len(tokens) for tokens in tokenized])),
        "question_end_rate": float(np.mean([text.rstrip().endswith("?") for text in texts])),
        "exclamation_per_response": float(np.mean([text.count("!") for text in texts])),
        "apology_rate": float(
            np.mean([bool(apology_pattern.search(text)) for text in texts])
        ),
        "mean_sentences": float(
            np.mean([len(re.findall(r"[.!?]+(?:\s|$)", text)) for text in texts])
        ),
        "opening_unique_rate": float(len(openings) / len(texts)),
    }


def load_scored_response_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """ledger とローカル artifact を結合して response corpus 表を作る。

    Returns:
        読み取り可能な artifact 表、除外理由表、対象 ledger 行数のタプル。
    """
    ledger_path = REPO_ROOT / "mcrs/experiments/results_ledger.jsonl"
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    eligible_count = 0
    for ledger_index, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("split") != "blindA"
            or record.get("judge") is None
            or not record.get("artifact")
        ):
            continue
        eligible_count += 1
        artifact = resolve_artifact(str(record["artifact"]))
        if artifact is None:
            exclusions.append(
                {
                    "ledger_index": ledger_index,
                    "exp": record.get("exp"),
                    "artifact": record.get("artifact"),
                    "reason": "artifact_not_found",
                }
            )
            continue
        try:
            prediction = read_prediction(artifact)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            exclusions.append(
                {
                    "ledger_index": ledger_index,
                    "exp": record.get("exp"),
                    "artifact": record.get("artifact"),
                    "reason": f"read_error:{type(error).__name__}",
                }
            )
            continue
        if len(prediction) != 80:
            exclusions.append(
                {
                    "ledger_index": ledger_index,
                    "exp": record.get("exp"),
                    "artifact": record.get("artifact"),
                    "reason": f"record_count:{len(prediction)}",
                }
            )
            continue

        texts = [str(row.get("predicted_response") or "").strip() for row in prediction]
        canonical = json.dumps(
            prediction, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        ranking_payload = json.dumps(
            [
                {
                    "session_id": row.get("session_id"),
                    "user_id": row.get("user_id"),
                    "turn_number": row.get("turn_number"),
                    "predicted_track_ids": row.get("predicted_track_ids"),
                }
                for row in prediction
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        top1_payload = json.dumps(
            [
                (row.get("predicted_track_ids") or [None])[0]
                for row in prediction
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response_payload = "\n".join(texts)
        row = {
            "ledger_index": ledger_index,
            "exp": record.get("exp"),
            "variant": record.get("variant"),
            "artifact": artifact.relative_to(REPO_ROOT).as_posix(),
            "artifact_name": artifact.name,
            "ndcg": record.get("ndcg"),
            "catalog_div": record.get("catalog_div"),
            "lexical_div": record.get("lexical_div"),
            "judge": float(record["judge"]),
            "composite": record.get("composite"),
            "prediction_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "ranking_hash": hashlib.sha256(ranking_payload.encode("utf-8")).hexdigest(),
            "top1_hash": hashlib.sha256(top1_payload.encode("utf-8")).hexdigest(),
            "response_hash": hashlib.sha256(response_payload.encode("utf-8")).hexdigest(),
        }
        row.update(response_features(texts))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(exclusions), eligible_count


def response_correlation_table(artifacts: pd.DataFrame) -> pd.DataFrame:
    """同一 response・ranking の再採点を平均化して記述的相関を計算する。

    Args:
        artifacts: ``load_scored_response_artifacts`` の出力。

    Returns:
        Pearson／Spearman 相関と観測数の表。
    """
    feature_columns = [
        "mean_words",
        "question_end_rate",
        "exclamation_per_response",
        "apology_rate",
        "mean_sentences",
        "opening_unique_rate",
    ]
    grouped_rows: list[dict[str, float | int]] = []
    for _, group in artifacts.groupby(["response_hash", "ranking_hash"]):
        row = {feature: float(group.iloc[0][feature]) for feature in feature_columns}
        row["judge"] = float(group["judge"].mean())
        row["n_official_scores"] = int(len(group))
        grouped_rows.append(row)
    unique_pairs = pd.DataFrame(grouped_rows)

    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        x = unique_pairs[feature]
        y = unique_pairs["judge"]
        rows.append(
            {
                "feature": feature,
                "n_unique_response_ranking_pairs": int(len(unique_pairs)),
                "pearson_r": float(x.corr(y, method="pearson")),
                "spearman_r": float(x.rank().corr(y.rank(), method="pearson")),
                "feature_min": float(x.min()),
                "feature_max": float(x.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def exact_resubmission_table(artifacts: pd.DataFrame) -> pd.DataFrame:
    """canonical prediction が同一で judge が異なる再提出を抽出する。

    Args:
        artifacts: 採点済み artifact 表。

    Returns:
        同一 prediction group ごとの公式スコア変動。
    """
    latest_per_artifact = (
        artifacts.sort_values("ledger_index").drop_duplicates("artifact", keep="last")
    )
    rows: list[dict[str, Any]] = []
    for prediction_hash, group in latest_per_artifact.groupby("prediction_hash"):
        if len(group) < 2 or group["judge"].nunique() < 2:
            continue
        rows.append(
            {
                "prediction_hash": prediction_hash,
                "n_submissions": int(len(group)),
                "artifacts": ";".join(sorted(group["artifact_name"].tolist())),
                "judge_values": ";".join(f"{value:.2f}" for value in group["judge"]),
                "judge_min": float(group["judge"].min()),
                "judge_max": float(group["judge"].max()),
                "judge_range": float(group["judge"].max() - group["judge"].min()),
                "composite_values": ";".join(
                    f"{float(value):.4f}" for value in group["composite"]
                ),
                "composite_range": float(
                    group["composite"].max() - group["composite"].min()
                ),
                "ndcg_values": ";".join(f"{float(value):.4f}" for value in group["ndcg"]),
                "lexical_values": ";".join(
                    f"{float(value):.4f}" for value in group["lexical_div"]
                ),
            }
        )
    return pd.DataFrame(rows)


def response_decomposition_table(artifacts: pd.DataFrame) -> pd.DataFrame:
    """固定ランキング response variant の composite 寄与を分解する。

    Args:
        artifacts: 採点済み artifact 表。

    Returns:
        lexical／judge 寄与と baseline 差分の表。

    Raises:
        ValueError: 必要な公式 variant が ledger から見つからない場合。
    """
    rows: list[dict[str, Any]] = []
    ranking_hashes: set[str] = set()
    for artifact_name, label in RESPONSE_VARIANTS.items():
        selected = artifacts[artifacts["artifact_name"] == artifact_name]
        if selected.empty:
            raise ValueError(f"missing scored response variant: {artifact_name}")
        record = selected.sort_values("ledger_index").iloc[-1]
        ranking_hashes.add(str(record["ranking_hash"]))
        ndcg = float(record["ndcg"])
        catalog = float(record["catalog_div"])
        lexical = float(record["lexical_div"])
        judge = float(record["judge"])
        lexical_term = 0.10 * lexical
        judge_term = 0.30 * ((judge - 1.0) / 4.0)
        fixed_term = 0.50 * ndcg + 0.10 * catalog
        reconstructed = fixed_term + lexical_term + judge_term
        rows.append(
            {
                "variant": label,
                "artifact_name": artifact_name,
                "ndcg": ndcg,
                "catalog_div": catalog,
                "lexical_div": lexical,
                "judge": judge,
                "official_composite": float(record["composite"]),
                "fixed_ranking_term": fixed_term,
                "lexical_term": lexical_term,
                "judge_term": judge_term,
                "reconstructed_composite": reconstructed,
                "ranking_hash": str(record["ranking_hash"]),
            }
        )
    if len(ranking_hashes) != 1:
        raise ValueError(
            "response variants do not share an identical ordered top-20 ranking"
        )
    table = pd.DataFrame(rows)
    baseline = table.iloc[0]
    table["lexical_delta_vs_initial"] = table["lexical_term"] - baseline["lexical_term"]
    table["judge_delta_vs_initial"] = table["judge_term"] - baseline["judge_term"]
    table["response_delta_vs_initial"] = (
        table["lexical_delta_vs_initial"] + table["judge_delta_vs_initial"]
    )
    table["ndcg_equivalent_delta"] = table["response_delta_vs_initial"] / 0.50
    return table


def write_summary(
    path: Path,
    *,
    pivot_table: pd.DataFrame,
    turn_adjusted: pd.DataFrame,
    logistic_table: pd.DataFrame,
    correlations: pd.DataFrame,
    exact_resubmissions: pd.DataFrame,
    decomposition: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
    n_sessions: int,
    response_eligible_count: int,
    response_included_count: int,
    response_excluded_count: int,
) -> None:
    """追加分析の主要結果を Markdown に保存する。

    Args:
        path: 出力 Markdown。
        pivot_table: pivot 感度分析表。
        turn_adjusted: turn直接標準化表。
        logistic_table: turn fixed effects logistic回帰表。
        correlations: response feature 相関表。
        exact_resubmissions: 同一 prediction 再提出表。
        decomposition: composite 分解表。
        bootstrap_replicates: bootstrap 反復数。
        seed: 乱数 seed。
        n_sessions: 分析 session 数。
        response_eligible_count: ledger 上の対象行数。
        response_included_count: 読み取れた artifact 数。
        response_excluded_count: 除外理由を記録した artifact 数。
    """
    primary = pivot_table[
        pivot_table["pivot_definition"] == "artist_explicit"
    ].iloc[0]
    primary_adjusted = turn_adjusted[
        turn_adjusted["pivot_definition"] == "artist_explicit"
    ].iloc[0]
    primary_logistic = logistic_table[
        (logistic_table["pivot_definition"] == "artist_explicit")
        & (logistic_table["outcome"] == "cumulative_repeat")
    ].iloc[0]
    broad = pivot_table[pivot_table["pivot_definition"] == "broad_discovery"].iloc[0]
    broad_adjusted = turn_adjusted[
        turn_adjusted["pivot_definition"] == "broad_discovery"
    ].iloc[0]
    core = pivot_table[pivot_table["pivot_definition"] == "core_change"].iloc[0]
    core_adjusted = turn_adjusted[
        turn_adjusted["pivot_definition"] == "core_change"
    ].iloc[0]
    long_row = decomposition[
        decomposition["variant"] == "Claude, grounded long"
    ].iloc[0]
    judge_share = long_row["judge_delta_vs_initial"] / long_row["response_delta_vs_initial"]
    exact_note = "該当なし"
    if not exact_resubmissions.empty:
        exact = exact_resubmissions.iloc[0]
        exact_note = (
            f"judge {exact['judge_min']:.2f}--{exact['judge_max']:.2f} "
            f"(range {exact['judge_range']:.2f}), composite range "
            f"{exact['composite_range']:.4f}"
        )
    correlation_lines = [
        "| Feature | Pearson r | Spearman r |",
        "|---|---:|---:|",
    ]
    for _, row in correlations.iterrows():
        correlation_lines.append(
            f"| {row['feature']} | {row['pearson_r']:.3f} | {row['spearman_r']:.3f} |"
        )
    correlation_markdown = "\n".join(correlation_lines)

    text = f"""# Evaluation-alignment audit: additional analysis

Last updated: 2026-07-24

## Run configuration

- Train sessions: {n_sessions:,}
- Cluster bootstrap: {bootstrap_replicates:,} resamples of whole sessions, seed={seed}
- Dataset access: official TalkPlayData-Challenge train + official track metadata
- Response audit: {response_included_count}/{response_eligible_count} eligible local
  Blind-A artifacts joined to `results_ledger.jsonl`; {response_excluded_count}
  exclusions are recorded in a separate CSV.
- Leak policy: train-only progress/future music rows are used for retrospective
  analysis only and never enter inference, prompt, fallback, or validation.

## RQ1: sequential artist-pivot audit with turn adjustment

The correct temporal ordering is `music[t-1] -> listener feedback/message[t] -> music[t]`.
Therefore `goal_progress[t]` evaluates the previous recommendation, not the current relevant track.

- Primary artist-targeted pivots: n={int(primary['pivot_turns']):,}; raw next-target
  prior-session-artist rate {primary['next_same_rate']:.3f}. After direct standardization
  to the pooled turn-2--8 distribution, pivot/non-pivot rates are
  {primary_adjusted['standardized_pivot_same_rate']:.3f}/{primary_adjusted['standardized_nonpivot_same_rate']:.3f},
  a risk difference of {primary_adjusted['standardized_same_risk_difference']:.3f}
  [{primary_adjusted['standardized_same_risk_difference_ci_low']:.3f},
  {primary_adjusted['standardized_same_risk_difference_ci_high']:.3f}].
- The turn-fixed-effects logistic sensitivity check gives OR
  {primary_logistic['odds_ratio']:.3f} [{primary_logistic['odds_ratio_ci_low']:.3f},
  {primary_logistic['odds_ratio_ci_high']:.3f}] with session-cluster robust SE.
- For t>=3 primary pivots, prior same-artist recommendations receive MOVES at
  {primary['prior_same_moves_rate']:.3f}, versus {primary['prior_different_moves_rate']:.3f}
  after prior different-artist recommendations: gap {primary['prior_moves_gap']:.3f}
  [{primary['prior_moves_gap_ci_low']:.3f}, {primary['prior_moves_gap_ci_high']:.3f}].
- High-conflict primary slice (artist-targeted pivot after prior same-artist +
  DOES_NOT): n={int(primary['conflict_turns']):,}; the next target repeats the
  immediately prior artist in {primary['conflict_next_adjacent_rate']:.3f}
  [{primary['conflict_next_adjacent_rate_ci_low']:.3f},
  {primary['conflict_next_adjacent_rate_ci_high']:.3f}].
- The broad discovery/change cue remains a sensitivity slice: n={int(broad['pivot_turns']):,},
  raw prior-session-artist rate {broad['next_same_rate']:.3f}, turn-standardized
  risk difference {broad_adjusted['standardized_same_risk_difference']:.3f}, and
  high-conflict immediate-artist repeat rate {broad['conflict_next_adjacent_rate']:.3f}.
- An alternative core-change operationalization gives n={int(core['pivot_turns']):,}
  and raw next-target prior-session-artist rate {core['next_same_rate']:.3f};
  turn-standardized risk difference {core_adjusted['standardized_same_risk_difference']:.3f};
  and high-conflict immediate-artist repeat rate {core['conflict_next_adjacent_rate']:.3f}.
  Because the three regex definitions are alternative rather than nested filters,
  this is a sensitivity check, not a strict conservative subset comparison.

## RQ2: fixed-ranking response audit

- Initial -> grounded-long composite delta: {long_row['response_delta_vs_initial']:.4f}.
- Judge contribution: {long_row['judge_delta_vs_initial']:.4f} ({judge_share:.1%});
  lexical contribution: {long_row['lexical_delta_vs_initial']:.4f} ({1 - judge_share:.1%}).
- The response-only delta has the same composite weight as an nDCG increase of
  {long_row['ndcg_equivalent_delta']:.4f}, with ranking held fixed.
- Exact canonical prediction resubmission: {exact_note}. One pair does not estimate
  the full judge variance, but proves that a 0.10 judge difference need not reflect a content change.

Across {int(correlations['n_unique_response_ranking_pairs'].iloc[0])} unique scored
response--ranking pairs, the strongest Pearson correlations are:

{correlation_markdown}

These are post-hoc submission-level associations. In particular, signs for sentence count and
exclamation frequency are not stable relative to the earlier 13-variant slice, so they must not
be described as causal judge preferences.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """全追加分析を実行し、CSV と Markdown summary を保存する。"""
    args = parse_args()
    tables_dir = args.out_root / "tables"
    summary_dir = args.out_root / "summary"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] loading official metadata and train split ...")
    artist_index = load_artist_index()
    turn_frame = build_turn_frame(artist_index, args.limit)

    print("[2/8] building session-level sufficient statistics ...")
    cluster_matrix, session_ids, slices = build_cluster_statistics(turn_frame)
    standard_matrix, standard_session_ids, standard_slices = (
        build_turn_standardization_statistics(turn_frame)
    )
    if session_ids != standard_session_ids:
        raise ValueError("session ordering differs between bootstrap matrices")

    print("[3/8] running session-clustered bootstrap ...")
    pivot_table, _ = bootstrap_pivot_analysis(
        cluster_matrix,
        slices,
        args.bootstrap_replicates,
        args.seed,
    )
    turn_table = pivot_by_turn(turn_frame)
    turn_adjusted = bootstrap_turn_standardized_analysis(
        standard_matrix,
        standard_slices,
        args.bootstrap_replicates,
        args.seed,
    )

    print("[4/8] fitting turn-fixed-effects logistic checks ...")
    logistic_table = turn_fixed_effect_logistic_table(turn_frame)

    print("[5/8] auditing scored response artifacts ...")
    response_artifacts, response_exclusions, response_eligible_count = (
        load_scored_response_artifacts()
    )
    correlations = response_correlation_table(response_artifacts)
    exact_resubmissions = exact_resubmission_table(response_artifacts)
    decomposition = response_decomposition_table(response_artifacts)

    print("[6/8] writing RQ1 tables ...")
    pivot_table.to_csv(
        tables_dir / "evaluation_alignment_pivot_sensitivity.csv", index=False
    )
    turn_table.to_csv(
        tables_dir / "evaluation_alignment_pivot_by_turn.csv", index=False
    )
    turn_adjusted.to_csv(
        tables_dir / "evaluation_alignment_turn_adjusted.csv", index=False
    )
    logistic_table.to_csv(
        tables_dir / "evaluation_alignment_turn_logistic.csv", index=False
    )

    print("[7/8] writing RQ2 tables ...")
    correlations.to_csv(
        tables_dir / "evaluation_alignment_response_correlations.csv", index=False
    )
    exact_resubmissions.to_csv(
        tables_dir / "evaluation_alignment_exact_resubmissions.csv", index=False
    )
    decomposition.to_csv(
        tables_dir / "evaluation_alignment_response_decomposition.csv", index=False
    )
    response_exclusions.to_csv(
        tables_dir / "evaluation_alignment_response_exclusions.csv", index=False
    )

    print("[8/8] writing summary ...")
    write_summary(
        summary_dir / "20260720_evaluation_alignment_audit.md",
        pivot_table=pivot_table,
        turn_adjusted=turn_adjusted,
        logistic_table=logistic_table,
        correlations=correlations,
        exact_resubmissions=exact_resubmissions,
        decomposition=decomposition,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        n_sessions=len(session_ids),
        response_eligible_count=response_eligible_count,
        response_included_count=len(response_artifacts),
        response_excluded_count=len(response_exclusions),
    )
    print("done ->", summary_dir / "20260720_evaluation_alignment_audit.md")


if __name__ == "__main__":
    main()
