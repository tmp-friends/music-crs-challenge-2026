"""Artist-pivot detector の一人用人手監査を準備・集計する。

公開 train split から detector-positive / detector-negative を層化抽出し、
判定時に detector 出力や正解楽曲を見せない annotation CSV を作成する。
annotation 完了後は、別 manifest と annotation_id で結合し、層化抽出率で
重み付けした precision / recall / F1 と bootstrap 95%区間を計算する。

本スクリプトは論文用の retrospective analysis 専用である。current relevant track、
goal progress、future turn は annotation form へ出力せず、推論・prompt・fallback・
validation にも一切流用しない。
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
METADATA_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
MOVES = "MOVES_TOWARD_GOAL"
DONT = "DOES_NOT_MOVE_TOWARD_GOAL"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "EDA/annotation/pivot_intent_annotation_200.csv"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "EDA/annotation/pivot_intent_annotation_200_manifest.csv"
)
DEFAULT_METRICS = REPO_ROOT / "EDA/tables/pivot_intent_manual_metrics.csv"
DEFAULT_ERRORS = REPO_ROOT / "EDA/annotation/pivot_intent_annotation_errors.csv"
DEFAULT_SUMMARY = REPO_ROOT / "EDA/summary/20260723_pivot_intent_manual_audit.md"
GUIDELINE_VERSION = "2026-07-23"

VALID_LABELS = {"ARTIST_PIVOT", "NOT_ARTIST_PIVOT", "UNCERTAIN"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
EXPECTED_POPULATION_N = 106_393
EXPECTED_DETECTOR_COUNTS = {
    "pivot_broad_discovery": 10_982,
    "pivot_artist_explicit": 15_126,
    "pivot_core_change": 7_147,
}
EXPECTED_STRATUM_COUNTS = {
    "broad_positive_high_conflict": 4_899,
    "broad_positive_other": 6_083,
    "broad_negative": 95_411,
}

# 2026-07-20 evaluation-alignment audit と同じ三つの operationalization。
BROAD_PIVOT_RE = re.compile(
    r"(different artist|another artist|someone (?:else|new)|new artists?|"
    r"haven'?t heard|never heard|artist i (?:haven'?t|have not|don'?t know)|"
    r"by (?:a|an) (?:other|different|new)|other than|"
    r"something (?:else|new|different)|discover (?:new|different|some|something))",
    re.I,
)
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
CORE_CHANGE_PIVOT_RE = re.compile(
    r"(?:\bdifferent\s+(?:artist|artists|band|bands|singer|singers|musician|musicians)\b|"
    r"\banother\s+(?:artist|band|singer|musician)\b|"
    r"\bsomeone\s+else\b|"
    r"\bnot\s+(?:the\s+)?same\s+(?:artist|band|singer|musician)\b|"
    r"\bby\s+(?:someone\s+else|(?:a|an)\s+(?:other|different)\s+"
    r"(?:artist|band|singer|musician))\b)",
    re.I,
)
DETECTORS = {
    "broad_discovery": BROAD_PIVOT_RE,
    "artist_explicit": EXPLICIT_ARTIST_PIVOT_RE,
    "core_change": CORE_CHANGE_PIVOT_RE,
}


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。

    Returns:
        subcommand と入出力設定を持つ Namespace。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="blind annotation CSV と非公開 manifest を生成する。"
    )
    prepare.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="人が編集する blind annotation CSV。",
    )
    prepare.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="完了まで見ない detector／sampling manifest。",
    )
    prepare.add_argument(
        "--positive_high_conflict",
        type=int,
        default=50,
        help="broad positive かつ high-conflict の抽出件数。",
    )
    prepare.add_argument(
        "--positive_other",
        type=int,
        default=50,
        help="high-conflict 以外の broad positive 抽出件数。",
    )
    prepare.add_argument(
        "--negative",
        type=int,
        default=100,
        help="broad negative の抽出件数。",
    )
    prepare.add_argument("--seed", type=int, default=20260723)
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="既存 annotation を上書きする。通常は指定しない。",
    )

    status = subparsers.add_parser(
        "status", help="annotation CSV の進捗と入力値を検証する。"
    )
    status.add_argument(
        "--annotations", type=Path, default=DEFAULT_ANNOTATIONS
    )
    status.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    summarize = subparsers.add_parser(
        "summarize", help="完了済み annotation を manifest と結合して集計する。"
    )
    summarize.add_argument(
        "--annotations", type=Path, default=DEFAULT_ANNOTATIONS
    )
    summarize.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    summarize.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    summarize.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    summarize.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    summarize.add_argument("--bootstrap_replicates", type=int, default=5000)
    summarize.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    """表示用テキストの過剰な空白を正規化する。

    Args:
        value: 元の文字列または文字列化可能な値。

    Returns:
        改行と連続空白を一つの空白へまとめた文字列。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_sha256(value: Any) -> str:
    """文字列の改変検知に使う SHA-256 digest を返す。

    Args:
        value: hash 対象の値。

    Returns:
        UTF-8 文字列の SHA-256 hex digest。
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def detector_rule_sha256() -> str:
    """detector 名・regex・flag をまとめた rule fingerprint を返す。

    Returns:
        三つの detector 定義を識別する SHA-256 hex digest。
    """
    serialized = "\n".join(
        f"{name}\0{pattern.pattern}\0{pattern.flags}"
        for name, pattern in DETECTORS.items()
    )
    return text_sha256(serialized)


def join_metadata_names(value: Any) -> str:
    """metadata の文字列または文字列リストを表示用に結合する。

    Args:
        value: track_name / artist_name の値。

    Returns:
        空要素を除き `` / `` で結合した表示文字列。
    """
    if isinstance(value, list):
        return " / ".join(normalize_text(item) for item in value if item)
    return normalize_text(value)


def normalize_artists(value: Any) -> frozenset[str]:
    """artist 表現を比較用の小文字集合へ正規化する。

    Args:
        value: artist_name の文字列または文字列リスト。

    Returns:
        strip・小文字化した artist 名の集合。
    """
    if value is None:
        return frozenset()
    if isinstance(value, list):
        return frozenset(str(item).strip().lower() for item in value if item)
    return frozenset([str(value).strip().lower()])


def load_track_indexes() -> tuple[
    dict[str, frozenset[str]], dict[str, tuple[str, str]], str
]:
    """公式 metadata から artist 比較・context 表示用 index を作る。

    Returns:
        track_id→artist集合、track_id→(title, artist表示)、dataset fingerprint。
    """
    metadata = load_dataset(METADATA_NAME)["all_tracks"]
    artist_index: dict[str, frozenset[str]] = {}
    display_index: dict[str, tuple[str, str]] = {}
    for row in metadata:
        track_id = str(row["track_id"])
        artist_index[track_id] = normalize_artists(row.get("artist_name"))
        display_index[track_id] = (
            join_metadata_names(row.get("track_name")) or "(unknown title)",
            join_metadata_names(row.get("artist_name")) or "(unknown artist)",
        )
    return artist_index, display_index, str(metadata._fingerprint)


def format_context_before_current(
    by_turn: dict[int, dict[str, dict[str, Any]]],
    current_turn: int,
    display_index: dict[str, tuple[str, str]],
) -> str:
    """current user message より前に観測された対話だけを整形する。

    Args:
        by_turn: turn_number→role→conversation row。
        current_turn: annotation 対象 turn。
        display_index: track_id→(title, artist表示)。

    Returns:
        prior user / recommendation / assistant を時系列に並べた文字列。
    """
    lines: list[str] = []
    for turn_number in sorted(turn for turn in by_turn if turn < current_turn):
        turn = by_turn[turn_number]
        user_text = normalize_text((turn.get("user") or {}).get("content"))
        if user_text:
            lines.append(f"Turn {turn_number} User: {user_text}")

        music = turn.get("music") or {}
        track_id = str(music.get("content") or "")
        title, artist = display_index.get(
            track_id, ("(metadata unavailable)", "(metadata unavailable)")
        )
        lines.append(
            f"Turn {turn_number} Recommended: {title} — {artist}"
        )

        assistant_text = normalize_text(
            (turn.get("assistant") or {}).get("content")
        )
        if assistant_text:
            lines.append(f"Turn {turn_number} Assistant: {assistant_text}")
    return "\n".join(lines)


def build_candidate_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    """論文の RQ1 と同じ対象母集団から annotation 候補を作る。

    Returns:
        detector flag、high-conflict flag、blind context を持つ turn 表と
        dataset provenance。
    """
    artist_index, display_index, metadata_fingerprint = load_track_indexes()
    dataset = load_dataset(DATASET_NAME, split="train")
    records: list[dict[str, Any]] = []

    for session in dataset:
        by_turn: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in session["conversations"]:
            by_turn[int(row["turn_number"])][str(row["role"])] = row
        progress_by_turn = {
            int(row["turn_number"]): row.get("goal_progress_assessment")
            for row in session["goal_progress_assessments"]
        }

        cumulative_artists: set[str] = set()
        previous_same_cumulative: bool | None = None
        for turn_number in sorted(by_turn):
            turn = by_turn[turn_number]
            music = turn.get("music") or {}
            track_id = str(music.get("content") or "")
            current_artists = artist_index.get(track_id, frozenset())
            current_same_cumulative = (
                bool(current_artists & cumulative_artists)
                if current_artists and cumulative_artists
                else None
            )
            progress = progress_by_turn.get(turn_number)
            moves = 1 if progress == MOVES else (0 if progress == DONT else None)
            raw_user_text = str(
                (turn.get("user") or {}).get("content") or ""
            )
            user_text = normalize_text(raw_user_text)

            # RQ1 の率と同じ母集団に限定するが、current track と progress は form に出さない。
            if moves is not None and current_same_cumulative is not None:
                detector_flags = {
                    # 既存 audit と同じ raw text に適用し、表示時だけ空白を整える。
                    name: bool(pattern.search(raw_user_text))
                    for name, pattern in DETECTORS.items()
                }
                records.append(
                    {
                        "session_id": str(session["session_id"]),
                        "turn_number": int(turn_number),
                        "context_before_current": format_context_before_current(
                            by_turn, turn_number, display_index
                        ),
                        "current_user_message": user_text,
                        "pivot_broad_discovery": detector_flags[
                            "broad_discovery"
                        ],
                        "pivot_artist_explicit": detector_flags[
                            "artist_explicit"
                        ],
                        "pivot_core_change": detector_flags["core_change"],
                        "moves": moves,
                        "previous_same_cumulative": previous_same_cumulative,
                        "current_same_cumulative": current_same_cumulative,
                        "high_conflict": bool(
                            detector_flags["broad_discovery"]
                            and previous_same_cumulative is True
                            and moves == 0
                        ),
                    }
                )

            # 次 turn の feedback 対象としてのみ保持する。
            previous_same_cumulative = current_same_cumulative
            cumulative_artists |= current_artists

    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_POPULATION_N:
        raise ValueError(
            "unexpected eligible population: "
            f"{len(frame):,} != {EXPECTED_POPULATION_N:,}"
        )
    actual_detector_counts = {
        column: int(frame[column].sum())
        for column in EXPECTED_DETECTOR_COUNTS
    }
    if actual_detector_counts != EXPECTED_DETECTOR_COUNTS:
        raise ValueError(
            "detector counts differ from evaluation-alignment audit: "
            f"{actual_detector_counts} != {EXPECTED_DETECTOR_COUNTS}"
        )
    provenance = {
        "conversation_dataset_name": DATASET_NAME,
        "conversation_dataset_fingerprint": str(dataset._fingerprint),
        "metadata_dataset_name": METADATA_NAME,
        "metadata_dataset_fingerprint": metadata_fingerprint,
        "detector_rule_sha256": detector_rule_sha256(),
        "guideline_version": GUIDELINE_VERSION,
    }
    return frame, provenance


def assign_sample_strata(frame: pd.DataFrame) -> pd.Series:
    """broad detector と high-conflict 条件から抽出層を割り当てる。

    Args:
        frame: annotation 候補表。

    Returns:
        各行の sampling stratum 名。
    """
    broad = frame["pivot_broad_discovery"].astype(bool)
    high_conflict = frame["high_conflict"].astype(bool)
    return pd.Series(
        np.select(
            [
                broad & high_conflict,
                broad & ~high_conflict,
            ],
            [
                "broad_positive_high_conflict",
                "broad_positive_other",
            ],
            default="broad_negative",
        ),
        index=frame.index,
    )


def prepare_annotation_files(args: argparse.Namespace) -> None:
    """層化抽出した blind form と detector manifest を保存する。

    Args:
        args: prepare subcommand の引数。

    Raises:
        FileExistsError: 既存 annotation を ``--overwrite`` なしで上書きする場合。
        ValueError: 指定抽出数を満たす候補がない場合。
    """
    outputs = [args.annotations, args.manifest]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing annotation files: {joined}"
        )

    frame, provenance = build_candidate_frame()
    frame["sample_stratum"] = assign_sample_strata(frame)
    requested = {
        "broad_positive_high_conflict": args.positive_high_conflict,
        "broad_positive_other": args.positive_other,
        "broad_negative": args.negative,
    }

    rng = np.random.default_rng(args.seed)
    selected_parts: list[pd.DataFrame] = []
    population_counts = frame["sample_stratum"].value_counts().to_dict()
    if population_counts != EXPECTED_STRATUM_COUNTS:
        raise ValueError(
            "sampling strata differ from evaluation-alignment audit: "
            f"{population_counts} != {EXPECTED_STRATUM_COUNTS}"
        )
    for stratum, sample_n in requested.items():
        candidates = frame[frame["sample_stratum"] == stratum]
        if sample_n <= 0 or sample_n > len(candidates):
            raise ValueError(
                f"invalid sample size for {stratum}: {sample_n} "
                f"(population={len(candidates)})"
            )
        positions = rng.choice(len(candidates), size=sample_n, replace=False)
        selected = candidates.iloc[positions].copy()
        selected["population_n"] = len(candidates)
        selected["sample_n"] = sample_n
        selected["sample_weight"] = len(candidates) / sample_n
        selected_parts.append(selected)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(
        drop=True
    )
    selected.insert(
        0,
        "annotation_id",
        [f"PIVOT-{index:04d}" for index in range(1, len(selected) + 1)],
    )

    annotation_columns = [
        "annotation_id",
        "session_id",
        "turn_number",
        "context_before_current",
        "current_user_message",
    ]
    annotations = selected[annotation_columns].copy()
    annotations["label"] = ""
    annotations["confidence"] = ""
    annotations["notes"] = ""
    selected["context_before_current_sha256"] = selected[
        "context_before_current"
    ].map(text_sha256)
    selected["current_user_message_sha256"] = selected[
        "current_user_message"
    ].map(text_sha256)
    selected["sample_seed"] = int(args.seed)
    for key, value in provenance.items():
        selected[key] = value

    manifest_columns = [
        "annotation_id",
        "session_id",
        "turn_number",
        "sample_stratum",
        "population_n",
        "sample_n",
        "sample_weight",
        "pivot_broad_discovery",
        "pivot_artist_explicit",
        "pivot_core_change",
        "moves",
        "previous_same_cumulative",
        "current_same_cumulative",
        "high_conflict",
        "context_before_current_sha256",
        "current_user_message_sha256",
        "sample_seed",
        "conversation_dataset_name",
        "conversation_dataset_fingerprint",
        "metadata_dataset_name",
        "metadata_dataset_fingerprint",
        "detector_rule_sha256",
        "guideline_version",
    ]
    manifest = selected[manifest_columns].copy()

    args.annotations.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(args.annotations, index=False)
    manifest.to_csv(args.manifest, index=False)

    print(f"annotations: {args.annotations} ({len(annotations)} rows)")
    print(f"manifest: {args.manifest}")
    for stratum in requested:
        print(
            f"  {stratum}: sample={requested[stratum]}, "
            f"population={population_counts[stratum]:,}"
        )
    print("Do not inspect the manifest until all labels are complete.")


def normalize_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """annotation の label / confidence / notes を検証用に正規化する。

    Args:
        frame: 人が編集した annotation 表。

    Returns:
        対象列を strip・大文字化したコピー。

    Raises:
        ValueError: 必須列欠損、ID重複、不正な label / confidence がある場合。
    """
    required = {
        "annotation_id",
        "session_id",
        "turn_number",
        "context_before_current",
        "current_user_message",
        "label",
        "confidence",
        "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing annotation columns: {missing}")
    if frame["annotation_id"].duplicated().any():
        duplicates = frame.loc[
            frame["annotation_id"].duplicated(), "annotation_id"
        ].tolist()
        raise ValueError(f"duplicate annotation_id: {duplicates[:5]}")

    normalized = frame.copy()
    for column in ["label", "confidence"]:
        normalized[column] = (
            normalized[column].fillna("").astype(str).str.strip().str.upper()
        )
    normalized["notes"] = normalized["notes"].fillna("").astype(str).str.strip()

    invalid_labels = sorted(
        set(normalized["label"]) - VALID_LABELS - {""}
    )
    if invalid_labels:
        raise ValueError(f"invalid labels: {invalid_labels}")
    invalid_confidence = sorted(
        set(normalized["confidence"]) - VALID_CONFIDENCE - {""}
    )
    if invalid_confidence:
        raise ValueError(f"invalid confidence values: {invalid_confidence}")
    missing_confidence = normalized[
        normalized["label"].ne("") & normalized["confidence"].eq("")
    ]
    if not missing_confidence.empty:
        ids = missing_confidence["annotation_id"].head(5).tolist()
        raise ValueError(f"labeled rows missing confidence: {ids}")
    return normalized


def load_manifest(path: Path) -> pd.DataFrame:
    """manifest を読み込み、必須列と ID 一意性を検証する。

    Args:
        path: detector／sampling manifest。

    Returns:
        ID列を文字列として読み込んだ manifest。

    Raises:
        ValueError: 必須列欠損または annotation_id 重複がある場合。
    """
    manifest = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={"annotation_id": "string", "session_id": "string"},
    )
    required = {
        "annotation_id",
        "session_id",
        "turn_number",
        "context_before_current_sha256",
        "current_user_message_sha256",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"missing manifest columns: {missing}")
    if manifest["annotation_id"].duplicated().any():
        raise ValueError("manifest contains duplicate annotation_id")
    return manifest


def validate_annotation_integrity(
    annotations: pd.DataFrame, manifest: pd.DataFrame
) -> None:
    """form の行集合と非入力列が生成時から変わっていないことを検証する。

    Args:
        annotations: 正規化済み annotation form。
        manifest: 生成時の ID と hash を持つ manifest。

    Raises:
        ValueError: ID、session-turn、context、message が変わった場合。
    """
    annotation_ids = set(annotations["annotation_id"])
    manifest_ids = set(manifest["annotation_id"])
    if annotation_ids != manifest_ids:
        raise ValueError(
            "annotation/manifest ID mismatch: "
            f"annotation_only={len(annotation_ids - manifest_ids)}, "
            f"manifest_only={len(manifest_ids - annotation_ids)}"
        )

    form = annotations[
        [
            "annotation_id",
            "session_id",
            "turn_number",
            "context_before_current",
            "current_user_message",
        ]
    ].copy()
    form["context_before_current_sha256"] = form[
        "context_before_current"
    ].map(text_sha256)
    form["current_user_message_sha256"] = form[
        "current_user_message"
    ].map(text_sha256)
    expected = manifest[
        [
            "annotation_id",
            "session_id",
            "turn_number",
            "context_before_current_sha256",
            "current_user_message_sha256",
        ]
    ]
    checked = form.merge(
        expected,
        on="annotation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_form", "_manifest"),
    )
    identity_mismatch = (
        checked["session_id_form"].ne(checked["session_id_manifest"])
        | checked["turn_number_form"].ne(checked["turn_number_manifest"])
    )
    if identity_mismatch.any():
        ids = checked.loc[identity_mismatch, "annotation_id"].head(5).tolist()
        raise ValueError(f"session_id or turn_number changed: {ids}")
    for column in [
        "context_before_current_sha256",
        "current_user_message_sha256",
    ]:
        mismatch = checked[f"{column}_form"].ne(
            checked[f"{column}_manifest"]
        )
        if mismatch.any():
            ids = checked.loc[mismatch, "annotation_id"].head(5).tolist()
            raise ValueError(f"protected annotation text changed: {ids}")


def print_annotation_status(path: Path, manifest_path: Path) -> pd.DataFrame:
    """annotation 進捗を表示する。

    Args:
        path: 人が編集する annotation CSV。
        manifest_path: ID と非入力列の改変検知に使う manifest。

    Returns:
        正規化・検証済み annotation 表。
    """
    annotations = normalize_annotation_columns(
        pd.read_csv(
            path,
            keep_default_na=False,
            dtype={"annotation_id": "string", "session_id": "string"},
        )
    )
    validate_annotation_integrity(annotations, load_manifest(manifest_path))
    total = len(annotations)
    labeled = int(annotations["label"].ne("").sum())
    print(f"progress: {labeled}/{total} ({labeled / total:.1%})")
    counts = (
        annotations["label"]
        .replace("", "UNLABELED")
        .value_counts()
        .reindex(
            ["ARTIST_PIVOT", "NOT_ARTIST_PIVOT", "UNCERTAIN", "UNLABELED"],
            fill_value=0,
        )
    )
    for label, count in counts.items():
        print(f"  {label}: {int(count)}")
    return annotations


def as_bool(series: pd.Series) -> pd.Series:
    """CSV 由来の bool 列を安全に bool Series へ変換する。

    Args:
        series: bool または文字列表現を含む列。

    Returns:
        True / False の Series。

    Raises:
        ValueError: True / False 以外が含まれる場合。
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"true", "false"})
    if invalid:
        raise ValueError(f"invalid bool values: {invalid}")
    return normalized.eq("true")


def weighted_metrics(
    frame: pd.DataFrame, detector_column: str
) -> dict[str, float]:
    """一つの detector の層化重み付き confusion metrics を計算する。

    ``UNCERTAIN`` は無理に二値へ変換せず、metrics の分母から除外する。

    Args:
        frame: annotation と manifest の結合表。
        detector_column: detector flag 列名。

    Returns:
        weighted TP / FP / FN / TN、precision / recall / F1。
    """
    certain = frame[frame["label"].ne("UNCERTAIN")].copy()
    detected = as_bool(certain[detector_column])
    truth = certain["label"].eq("ARTIST_PIVOT")
    weight = certain["sample_weight"].astype(float)

    tp = float(weight[detected & truth].sum())
    fp = float(weight[detected & ~truth].sum())
    fn = float(weight[~detected & truth].sum())
    tn = float(weight[~detected & ~truth].sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
        else np.nan
    )
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall == 0:
        f1 = 0.0
    return {
        "unweighted_evaluable_n": int(len(certain)),
        "weighted_evaluable_n": float(weight.sum()),
        "weighted_tp": tp,
        "weighted_fp": fp,
        "weighted_fn": fn,
        "weighted_tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def bootstrap_detector_metrics(
    frame: pd.DataFrame,
    detector_column: str,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """全層共通の session cluster bootstrap で95%区間を求める。

    Args:
        frame: annotation と manifest の結合表。
        detector_column: detector flag 列名。
        replicates: bootstrap 反復数。
        seed: 乱数 seed。

    Returns:
        precision / recall / F1 → (2.5%, 97.5%) の辞書。

    Raises:
        ValueError: 反復数が正でない場合。
    """
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rng = np.random.default_rng(seed)
    session_codes, session_ids = pd.factorize(frame["session_id"], sort=False)
    population_by_stratum = (
        frame.groupby("sample_stratum")["population_n"].first().to_dict()
    )
    samples: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for _ in range(replicates):
        # 多項分布の cluster multiplicity を同一 session の全行へ共通適用する。
        for _attempt in range(100):
            cluster_multiplicity = rng.multinomial(
                len(session_ids),
                np.full(len(session_ids), 1.0 / len(session_ids)),
            )
            row_multiplicity = cluster_multiplicity[session_codes]
            represented = {
                stratum: int(
                    row_multiplicity[
                        frame["sample_stratum"].eq(stratum).to_numpy()
                    ].sum()
                )
                for stratum in population_by_stratum
            }
            if all(count > 0 for count in represented.values()):
                break
        else:
            raise RuntimeError(
                "failed to represent every sampling stratum in bootstrap"
            )

        resampled = frame.copy()
        for stratum, population_n in population_by_stratum.items():
            stratum_mask = resampled["sample_stratum"].eq(stratum)
            # 各層の総 weight を母集団数へ再正規化する。
            resampled.loc[stratum_mask, "sample_weight"] = (
                float(population_n)
                * row_multiplicity[stratum_mask.to_numpy()]
                / represented[stratum]
            )
        metrics = weighted_metrics(
            resampled, detector_column
        )
        for metric_name in samples:
            samples[metric_name].append(metrics[metric_name])
    return {
        metric_name: tuple(
            float(value)
            for value in np.nanquantile(values, [0.025, 0.975])
        )
        for metric_name, values in samples.items()
    }


def load_completed_annotations(
    annotations_path: Path, manifest_path: Path
) -> pd.DataFrame:
    """完了済み annotation と manifest を一対一で結合する。

    Args:
        annotations_path: 人が編集した annotation CSV。
        manifest_path: detector／sampling manifest。

    Returns:
        annotation と detector 情報の結合表。

    Raises:
        ValueError: 未入力行、ID不一致、session-turn 不一致がある場合。
    """
    annotations = print_annotation_status(annotations_path, manifest_path)
    unlabeled = annotations[annotations["label"].eq("")]
    if not unlabeled.empty:
        ids = unlabeled["annotation_id"].head(10).tolist()
        raise ValueError(
            f"annotation is incomplete: {len(unlabeled)} unlabeled rows; "
            f"first ids={ids}"
        )

    manifest = load_manifest(manifest_path)

    merged = annotations.merge(
        manifest,
        on=["annotation_id", "session_id", "turn_number"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(annotations):
        raise ValueError("session_id or turn_number changed in annotation CSV")
    return merged


def build_error_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """各 detector の false positive / false negative 行を作る。

    Args:
        frame: annotation と manifest の結合表。

    Returns:
        error_type を付与した annotation 行。
    """
    rows: list[dict[str, Any]] = []
    certain = frame[frame["label"].ne("UNCERTAIN")]
    truth = certain["label"].eq("ARTIST_PIVOT")
    for detector_name in DETECTORS:
        detector_column = f"pivot_{detector_name}"
        detected = as_bool(certain[detector_column])
        error_masks = {
            "FALSE_POSITIVE": detected & ~truth,
            "FALSE_NEGATIVE": ~detected & truth,
        }
        for error_type, mask in error_masks.items():
            for _, row in certain.loc[mask].iterrows():
                rows.append(
                    {
                        "detector": detector_name,
                        "error_type": error_type,
                        "annotation_id": row["annotation_id"],
                        "session_id": row["session_id"],
                        "turn_number": int(row["turn_number"]),
                        "current_user_message": row["current_user_message"],
                        "label": row["label"],
                        "confidence": row["confidence"],
                        "notes": row["notes"],
                    }
                )
    columns = [
        "detector",
        "error_type",
        "annotation_id",
        "session_id",
        "turn_number",
        "current_user_message",
        "label",
        "confidence",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_manual_audit_summary(
    path: Path,
    metrics: pd.DataFrame,
    annotations: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> None:
    """一人用 manual audit の Markdown summary を保存する。

    Args:
        path: 出力 Markdown。
        metrics: detector 別 metrics 表。
        annotations: annotation と manifest の結合表。
        bootstrap_replicates: bootstrap 反復数。
        seed: bootstrap seed。
    """
    label_counts = annotations["label"].value_counts()
    total_weight = float(annotations["sample_weight"].sum())
    uncertain_weight = float(
        annotations.loc[
            annotations["label"].eq("UNCERTAIN"), "sample_weight"
        ].sum()
    )
    uncertain_rate = uncertain_weight / total_weight
    binary_coverage = 1.0 - uncertain_rate
    sampled_sessions = int(annotations["session_id"].nunique())
    duplicate_session_rows = len(annotations) - sampled_sessions
    metric_lines = [
        "| Detector | Eval. n | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        metric_lines.append(
            f"| {row['detector']} | {int(row['unweighted_evaluable_n'])} | "
            f"{row['precision']:.3f} "
            f"[{row['precision_ci_low']:.3f}, {row['precision_ci_high']:.3f}] | "
            f"{row['recall']:.3f} "
            f"[{row['recall_ci_low']:.3f}, {row['recall_ci_high']:.3f}] | "
            f"{row['f1']:.3f} "
            f"[{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] |"
        )
    metric_markdown = "\n".join(metric_lines)

    text = f"""# Single-annotator artist-pivot detector audit

Last updated: 2026-07-23

## Scope

- Annotators: 1
- Sample: {len(annotations)} stratified train turns
- Labels: ARTIST_PIVOT / NOT_ARTIST_PIVOT / UNCERTAIN
- Sessions represented: {sampled_sessions} ({duplicate_session_rows} additional rows
  share a session)
- Bootstrap: {bootstrap_replicates:,} common session-cluster resamples across
  strata with within-stratum weight normalization, seed={seed}
- Weighting: population turns / sampled turns within each sampling stratum
- Agreement: not applicable (single-annotator audit)

## Unweighted sample label counts

- ARTIST_PIVOT: {int(label_counts.get('ARTIST_PIVOT', 0))}
- NOT_ARTIST_PIVOT: {int(label_counts.get('NOT_ARTIST_PIVOT', 0))}
- UNCERTAIN: {int(label_counts.get('UNCERTAIN', 0))}

These raw counts describe the deliberately stratified sample and are not prevalence
estimates. The population-weighted uncertain rate is {uncertain_rate:.3%}; binary
metric coverage is {binary_coverage:.3%}.

## Detector metrics

{metric_markdown}

`UNCERTAIN` rows are excluded from binary metrics. `Eval. n` is the unweighted
number of classifiable sampled rows. Confidence intervals use session clusters
within each sampling stratum; they quantify sampling uncertainty but do not include
annotator uncertainty. Because there is only one annotator, the result must be
described as a manual audit rather than inter-annotator validation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_annotations(args: argparse.Namespace) -> None:
    """完了済み annotation の metrics・error analysis・summary を保存する。

    Args:
        args: summarize subcommand の引数。
    """
    merged = load_completed_annotations(args.annotations, args.manifest)
    metric_rows: list[dict[str, Any]] = []
    for detector_index, detector_name in enumerate(DETECTORS):
        detector_column = f"pivot_{detector_name}"
        point = weighted_metrics(merged, detector_column)
        intervals = bootstrap_detector_metrics(
            merged,
            detector_column,
            args.bootstrap_replicates,
            args.seed + detector_index,
        )
        row: dict[str, Any] = {"detector": detector_name, **point}
        for metric_name, (low, high) in intervals.items():
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    errors = build_error_analysis(merged)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    errors.to_csv(args.errors, index=False)
    write_manual_audit_summary(
        args.summary,
        metrics,
        merged,
        args.bootstrap_replicates,
        args.seed,
    )
    print(f"metrics: {args.metrics}")
    print(f"errors: {args.errors}")
    print(f"summary: {args.summary}")


def main() -> None:
    """選択された annotation workflow subcommand を実行する。"""
    args = parse_args()
    if args.command == "prepare":
        prepare_annotation_files(args)
    elif args.command == "status":
        print_annotation_status(args.annotations, args.manifest)
    elif args.command == "summarize":
        summarize_annotations(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
