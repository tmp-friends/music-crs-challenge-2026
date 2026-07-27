#!/usr/bin/env python3
"""Blind A の公開済み過去 turn label 混入仮説をローカル診断する。

このスクリプトは Codabench に提出するための artifact を作らない。
ZIP 内のファイル名も意図的に ``prediction.json`` ではなく
``DO_NOT_SUBMIT_prediction.json`` にして、誤提出を避ける。
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_FINAL_PREDICTIONS = "exp/inference/blindset_A/exp015_B_ll_t500_A.json"
DEFAULT_OUTPUT_DIR = "exp/audit/blind_a_past_turn_label"


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。

    Returns:
        argparse で解析した引数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Blind A の公開済み過去 turn label を混ぜた場合の naive scorer 仮説を診断する。"
        )
    )
    parser.add_argument(
        "--final-predictions",
        default=DEFAULT_FINAL_PREDICTIONS,
        help=(
            "未知 final turn 80 件に使う通常 prediction JSON。"
            f" default: {DEFAULT_FINAL_PREDICTIONS}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"診断 artifact の出力先。default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--assumed-final-ndcg",
        type=float,
        default=None,
        help="既知 LB などから仮置きする final 80 件の nDCG@20。",
    )
    parser.add_argument(
        "--target-ndcg",
        type=float,
        default=0.75,
        help="逆算対象の mixed nDCG@20。default: 0.75",
    )
    return parser.parse_args()


def normalize_conversations(conversations: Any) -> list[dict[str, Any]]:
    """datasets の conversation 表現を list-of-dict にそろえる。

    Args:
        conversations: Hugging Face datasets が返す dict-of-list または list-of-dict。

    Returns:
        conversation row のリスト。
    """
    if isinstance(conversations, dict):
        keys = list(conversations.keys())
        if not keys:
            return []
        length = len(conversations[keys[0]])
        return [{key: conversations[key][index] for key in keys} for index in range(length)]
    return [dict(row) for row in conversations]


def load_final_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """通常の 80-row Blind A prediction を session-turn key で読み込む。

    Args:
        path: prediction JSON のパス。

    Returns:
        ``(session_id, turn_number)`` から prediction row への mapping。

    Raises:
        FileNotFoundError: 指定パスが存在しない場合。
        ValueError: 同じ session-turn が重複している場合。
    """
    if not path.exists():
        raise FileNotFoundError(f"final prediction file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in data:
        key = (str(row["session_id"]), int(row["turn_number"]))
        if key in predictions:
            raise ValueError(f"duplicate final prediction key: {key}")
        predictions[key] = row
    return predictions


def load_fallback_track_ids(topk: int = 20) -> list[str]:
    """final prediction がない場合に使う人気 track fallback を取得する。

    Args:
        topk: 返す track 数。

    Returns:
        popularity 降順の track_id リスト。
    """
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    rows = [
        (str(item["track_id"]), float(item["popularity"] or 0.0))
        for item in tracks
    ]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [track_id for track_id, _ in rows[:topk]]


def visible_label_rows(
    *,
    final_predictions: dict[tuple[str, int], dict[str, Any]],
    fallback_track_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Blind A の全 user turn から 290-row probe を構築する。

    公開済み label 付き turn は label track を top1 に置く。未知 final turn は
    通常の 80-row prediction を流用する。これは scorer 仕様の検証用であり、
    正式提出に使う artifact ではない。

    Args:
        final_predictions: 未知 final turn に使う通常 prediction。
        fallback_track_ids: final prediction 欠損時の fallback track。

    Returns:
        (probe rows, summary) のタプル。
    """
    blind = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test")
    rows: list[dict[str, Any]] = []
    turn_counts: Counter[int] = Counter()
    visible_turn_counts: Counter[int] = Counter()
    hidden_turn_counts: Counter[int] = Counter()
    missing_final_prediction_keys: list[str] = []

    for item in blind:
        conversations = normalize_conversations(item["conversations"])
        user_turn_numbers = sorted(
            {
                int(row.get("turn_number", 0))
                for row in conversations
                if row.get("role") == "user"
            }
        )
        for turn_number in user_turn_numbers:
            current_turn = [
                row
                for row in conversations
                if int(row.get("turn_number", 0)) == turn_number
            ]
            labels = [
                str(row.get("content", ""))
                for row in current_turn
                if row.get("role") == "music"
            ]
            key = (str(item["session_id"]), int(turn_number))
            turn_counts[turn_number] += 1

            if labels:
                visible_turn_counts[turn_number] += 1
                predicted_track_ids = list(labels)
                predicted_track_ids.extend(
                    track_id
                    for track_id in fallback_track_ids
                    if track_id not in predicted_track_ids
                )
                rows.append(
                    {
                        "session_id": str(item["session_id"]),
                        "user_id": str(item["user_id"]),
                        "turn_number": int(turn_number),
                        "predicted_track_ids": predicted_track_ids[:20],
                        "predicted_response": (
                            "DO_NOT_SUBMIT diagnostic row for visible historical turn."
                        ),
                    }
                )
                continue

            hidden_turn_counts[turn_number] += 1
            final_row = final_predictions.get(key)
            if final_row is None:
                missing_final_prediction_keys.append(f"{key[0]}:{key[1]}")
                predicted_track_ids = fallback_track_ids[:20]
                predicted_response = "DO_NOT_SUBMIT fallback diagnostic row."
            else:
                predicted_track_ids = [str(track_id) for track_id in final_row["predicted_track_ids"][:20]]
                predicted_response = str(final_row.get("predicted_response", ""))
            rows.append(
                {
                    "session_id": str(item["session_id"]),
                    "user_id": str(item["user_id"]),
                    "turn_number": int(turn_number),
                    "predicted_track_ids": predicted_track_ids,
                    "predicted_response": predicted_response,
                }
            )

    visible_rows = sum(visible_turn_counts.values())
    hidden_rows = sum(hidden_turn_counts.values())
    total_rows = len(rows)
    summary = {
        "dataset": "talkpl-ai/TalkPlayData-Challenge-Blind-A:test",
        "probe_rows": total_rows,
        "visible_label_rows": visible_rows,
        "hidden_final_rows": hidden_rows,
        "turn_counts": dict(sorted(turn_counts.items())),
        "visible_turn_counts": dict(sorted(visible_turn_counts.items())),
        "hidden_turn_counts": dict(sorted(hidden_turn_counts.items())),
        "visible_only_micro_ndcg_if_hidden_zero": (
            visible_rows / total_rows if total_rows else 0.0
        ),
        "missing_final_prediction_keys": missing_final_prediction_keys,
    }
    return rows, summary


def add_score_hypotheses(
    summary: dict[str, Any],
    *,
    assumed_final_ndcg: float | None,
    target_ndcg: float,
) -> None:
    """naive submitted-row scorer を仮定した概算値を summary に追加する。

    Args:
        summary: 更新対象 summary。
        assumed_final_ndcg: final 80 件の仮置き nDCG。None の場合は混合値を計算しない。
        target_ndcg: 逆算対象の mixed nDCG。
    """
    visible_rows = int(summary["visible_label_rows"])
    hidden_rows = int(summary["hidden_final_rows"])
    total_rows = int(summary["probe_rows"])
    if total_rows <= 0 or hidden_rows <= 0:
        return

    required_hidden = (float(target_ndcg) * total_rows - visible_rows) / hidden_rows
    summary["target_ndcg"] = float(target_ndcg)
    summary["required_hidden_final_ndcg_for_target"] = required_hidden
    if assumed_final_ndcg is not None:
        mixed = (visible_rows + hidden_rows * float(assumed_final_ndcg)) / total_rows
        summary["assumed_final_ndcg"] = float(assumed_final_ndcg)
        summary["mixed_ndcg_with_assumed_final"] = mixed
        summary["mixed_ndcg_floor_if_hidden_zero"] = visible_rows / total_rows


def write_artifacts(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """診断 JSON と誤提出防止 ZIP を書き出す。

    Args:
        rows: 290-row probe prediction。
        summary: 診断 summary。
        output_dir: 出力先ディレクトリ。

    Returns:
        書き出した artifact path の辞書。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "visible_past_turn_probe_DO_NOT_SUBMIT.json"
    summary_path = output_dir / "visible_past_turn_probe_summary.json"
    zip_path = output_dir / "visible_past_turn_probe_DO_NOT_SUBMIT.zip"

    prediction_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Codabench の通常 submission root は prediction.json なので、意図的に別名にする。
        zf.write(prediction_path, arcname="DO_NOT_SUBMIT_prediction.json")
        zf.write(summary_path, arcname="manifest_DO_NOT_SUBMIT.json")
    return {
        "prediction_json": str(prediction_path),
        "summary_json": str(summary_path),
        "diagnostic_zip": str(zip_path),
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """probe row の最低限の形式を検証する。

    Args:
        rows: 検証対象の prediction row。

    Raises:
        ValueError: 必須 field、top20 件数、session-turn 重複に問題がある場合。
    """
    required = {
        "session_id",
        "user_id",
        "turn_number",
        "predicted_track_ids",
        "predicted_response",
    }
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        key = (str(row["session_id"]), int(row["turn_number"]))
        if key in seen:
            raise ValueError(f"duplicate session-turn row: {key}")
        seen.add(key)
        predicted_track_ids = row["predicted_track_ids"]
        if not isinstance(predicted_track_ids, list) or len(predicted_track_ids) != 20:
            raise ValueError(f"row {index} must have exactly 20 predicted_track_ids")


def main() -> int:
    """診断 artifact を生成する。"""
    args = parse_args()
    final_predictions = load_final_predictions(Path(args.final_predictions))
    fallback_track_ids = load_fallback_track_ids(topk=20)
    rows, summary = visible_label_rows(
        final_predictions=final_predictions,
        fallback_track_ids=fallback_track_ids,
    )
    add_score_hypotheses(
        summary,
        assumed_final_ndcg=args.assumed_final_ndcg,
        target_ndcg=args.target_ndcg,
    )
    validate_rows(rows)
    paths = write_artifacts(rows=rows, summary=summary, output_dir=Path(args.output_dir))

    print(json.dumps({"paths": paths, "summary": summary}, ensure_ascii=False, indent=2))
    print(
        "\nNOTE: diagnostic_zip intentionally does not contain root prediction.json; "
        "do not submit this artifact to Codabench."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
