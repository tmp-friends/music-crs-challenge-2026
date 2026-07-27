"""EDA for relistening tendency in TalkPlayData challenge conversations.

The script intentionally writes only under EDA/. It does not touch
mcrs/experiments/ or inference configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont


SPLIT_LABELS = {
    "train": "train",
    "test": "dev_test",
}
TOPKS = (20, 50, 100)
REPEAT_TYPES = ("exact", "artist", "album", "tag")
SCOPES = ("session", "user", "combined")


@dataclass(frozen=True)
class TrackMeta:
    track_id: str
    artists: tuple[str, ...]
    albums: tuple[str, ...]
    tags: tuple[str, ...]
    popularity: float


@dataclass(frozen=True)
class PriorState:
    tracks: tuple[str, ...]
    artists: tuple[str, ...]
    albums: tuple[str, ...]
    tags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversation-dataset",
        default="input/challenge-dataset",
        help="Local load_from_disk path for TalkPlayData conversation dataset.",
    )
    parser.add_argument(
        "--track-metadata-dataset",
        default="input/track-metadata",
        help="Local load_from_disk path for track metadata dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="EDA",
        help="Output directory. Tables and figures are written below this path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional session limit per split for smoke runs.",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated dataset splits to analyze.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG figure generation.",
    )
    return parser.parse_args()


def listify(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(item) for item in value if item is not None and str(item))


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def normalize_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope: {scope}")
    return scope


def load_track_metadata(path: str) -> dict[str, TrackMeta]:
    dataset = load_from_disk(path)["all_tracks"]
    metadata: dict[str, TrackMeta] = {}
    for row in dataset:
        track_id = str(row["track_id"])
        metadata[track_id] = TrackMeta(
            track_id=track_id,
            artists=listify(row.get("artist_name")),
            albums=listify(row.get("album_name")),
            tags=listify(row.get("tag_list")),
            popularity=safe_float(row.get("popularity")),
        )
    return metadata


def build_metadata_indexes(
    metadata: dict[str, TrackMeta],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    artist_to_tracks: dict[str, list[str]] = defaultdict(list)
    album_to_tracks: dict[str, list[str]] = defaultdict(list)
    for track_id, meta in metadata.items():
        for artist in meta.artists:
            artist_to_tracks[artist].append(track_id)
        for album in meta.albums:
            album_to_tracks[album].append(track_id)

    def sort_key(track_id: str) -> tuple[float, str]:
        return (-metadata[track_id].popularity, track_id)

    for index in (artist_to_tracks, album_to_tracks):
        for key, track_ids in index.items():
            unique_track_ids = list(dict.fromkeys(track_ids))
            unique_track_ids.sort(key=sort_key)
            index[key] = unique_track_ids
    return dict(artist_to_tracks), dict(album_to_tracks)


def music_events(session: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for row in session["conversations"]:
        if row.get("role") != "music":
            continue
        events.append(
            {
                "track_id": str(row["content"]),
                "turn_number": int(row["turn_number"]),
            }
        )
    events.sort(key=lambda item: item["turn_number"])
    return events


def prior_from_track_ids(
    track_ids: Iterable[str], metadata: dict[str, TrackMeta]
) -> PriorState:
    tracks = ordered_unique(track_ids)
    artists: list[str] = []
    albums: list[str] = []
    tags: list[str] = []
    for track_id in tracks:
        meta = metadata.get(track_id)
        if meta is None:
            continue
        artists.extend(meta.artists)
        albums.extend(meta.albums)
        tags.extend(meta.tags)
    return PriorState(
        tracks=tracks,
        artists=ordered_unique(artists),
        albums=ordered_unique(albums),
        tags=ordered_unique(tags),
    )


def combine_priors(a: PriorState, b: PriorState) -> PriorState:
    return PriorState(
        tracks=ordered_unique((*a.tracks, *b.tracks)),
        artists=ordered_unique((*a.artists, *b.artists)),
        albums=ordered_unique((*a.albums, *b.albums)),
        tags=ordered_unique((*a.tags, *b.tags)),
    )


def has_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    return bool(set(left) & set(right))


def tag_stats(target_tags: tuple[str, ...], prior_tags: tuple[str, ...]) -> tuple[int, float]:
    target_set = set(target_tags)
    prior_set = set(prior_tags)
    overlap = len(target_set & prior_set)
    union = len(target_set | prior_set)
    return overlap, overlap / union if union else 0.0


def candidates_from_tracks(prior: PriorState, max_k: int) -> list[str]:
    return list(prior.tracks[:max_k])


def candidates_from_index(
    keys: tuple[str, ...],
    index: dict[str, list[str]],
    max_k: int,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for key in keys:
        for track_id in index.get(key, []):
            if track_id in seen:
                continue
            seen.add(track_id)
            candidates.append(track_id)
            if len(candidates) >= max_k:
                return candidates
    return candidates


def actionability_hits(
    target_track_id: str,
    prior: PriorState,
    artist_to_tracks: dict[str, list[str]],
    album_to_tracks: dict[str, list[str]],
) -> dict[tuple[str, int], int]:
    max_k = max(TOPKS)
    candidate_sets = {
        "prior_track": candidates_from_tracks(prior, max_k),
        "prior_artist": candidates_from_index(prior.artists, artist_to_tracks, max_k),
        "prior_album": candidates_from_index(prior.albums, album_to_tracks, max_k),
    }
    hits = {}
    for source, candidates in candidate_sets.items():
        for topk in TOPKS:
            hits[(source, topk)] = int(target_track_id in set(candidates[:topk]))
    return hits


def row_date_key(session: dict[str, Any]) -> tuple[str, str]:
    return (str(session.get("session_date") or ""), str(session["session_id"]))


def build_event_rows(
    split_name: str,
    sessions: list[dict[str, Any]],
    metadata: dict[str, TrackMeta],
    artist_to_tracks: dict[str, list[str]],
    album_to_tracks: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    split_label = SPLIT_LABELS.get(split_name, split_name)

    sessions_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        sessions_by_user[str(session["user_id"])].append(session)

    for user_sessions in sessions_by_user.values():
        user_sessions.sort(key=row_date_key)
        user_prior_track_ids: list[str] = []
        for session in user_sessions:
            session_prior_track_ids: list[str] = []
            events = music_events(session)
            goal = session.get("conversation_goal") or {}
            profile = session.get("user_profile") or {}
            user_prior = prior_from_track_ids(reversed(user_prior_track_ids), metadata)

            for event in events:
                target_track_id = event["track_id"]
                target_meta = metadata.get(target_track_id)
                if target_meta is None:
                    missing_rows.append(
                        {
                            "split": split_label,
                            "session_id": session["session_id"],
                            "user_id": session["user_id"],
                            "turn_number": event["turn_number"],
                            "track_id": target_track_id,
                        }
                    )
                    continue

                session_prior = prior_from_track_ids(
                    reversed(session_prior_track_ids), metadata
                )
                priors = {
                    "session": session_prior,
                    "user": user_prior,
                    "combined": combine_priors(session_prior, user_prior),
                }

                base = {
                    "split": split_label,
                    "session_id": session["session_id"],
                    "user_id": session["user_id"],
                    "session_date": session.get("session_date"),
                    "turn_number": event["turn_number"],
                    "track_id": target_track_id,
                    "goal_category": goal.get("category") or "",
                    "goal_specificity": goal.get("specificity") or "",
                    "user_split": profile.get("user_split") or "",
                }
                for scope, prior in priors.items():
                    overlap_count, jaccard = tag_stats(target_meta.tags, prior.tags)
                    row = dict(base)
                    row.update(
                        {
                            "scope": scope,
                            "prior_track_count": len(prior.tracks),
                            "prior_artist_count": len(prior.artists),
                            "prior_album_count": len(prior.albums),
                            "prior_tag_count": len(prior.tags),
                            "exact_repeat": int(target_track_id in set(prior.tracks)),
                            "artist_repeat": int(
                                has_overlap(target_meta.artists, prior.artists)
                            ),
                            "album_repeat": int(
                                has_overlap(target_meta.albums, prior.albums)
                            ),
                            "tag_repeat": int(overlap_count > 0),
                            "tag_overlap_count": overlap_count,
                            "tag_jaccard": jaccard,
                        }
                    )
                    event_rows.append(row)

                    hits = actionability_hits(
                        target_track_id,
                        prior,
                        artist_to_tracks,
                        album_to_tracks,
                    )
                    for (source, topk), hit in hits.items():
                        action_rows.append(
                            {
                                "split": split_label,
                                "scope": scope,
                                "source": source,
                                "topk": topk,
                                "active": int(bool(prior.tracks)),
                                "hit": hit,
                            }
                        )

                session_prior_track_ids.append(target_track_id)
            user_prior_track_ids.extend(event["track_id"] for event in events)

    return event_rows, action_rows, missing_rows


def rate_rows(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        event_count = len(group)
        for repeat_type in REPEAT_TYPES:
            col = f"{repeat_type}_repeat"
            row = dict(base)
            row.update(
                {
                    "repeat_type": repeat_type,
                    "events": event_count,
                    "eligible_events": int((group["prior_track_count"] > 0).sum()),
                    "hits": int(group[col].sum()),
                    "rate": float(group[col].mean()) if event_count else 0.0,
                    "eligible_rate": (
                        float(group.loc[group["prior_track_count"] > 0, col].mean())
                        if (group["prior_track_count"] > 0).any()
                        else 0.0
                    ),
                    "avg_tag_jaccard": float(group["tag_jaccard"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_actionability(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["split", "scope", "source", "topk"], dropna=False)
    rows = []
    for keys, group in grouped:
        split, scope, source, topk = keys
        rows.append(
            {
                "split": split,
                "scope": scope,
                "source": source,
                "topk": int(topk),
                "events": len(group),
                "active_events": int(group["active"].sum()),
                "active_rate": float(group["active"].mean()) if len(group) else 0.0,
                "hits": int(group["hit"].sum()),
                "recall": float(group["hit"].mean()) if len(group) else 0.0,
                "eligible_recall": (
                    float(group.loc[group["active"] > 0, "hit"].mean())
                    if (group["active"] > 0).any()
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "scope", "source", "topk"])


def history_size_table(event_df: pd.DataFrame) -> pd.DataFrame:
    df = event_df[event_df["scope"] == "user"].copy()
    bins = [-1, 0, 8, 24, 64, 10**9]
    labels = ["0", "1-8", "9-24", "25-64", "65+"]
    df["user_prior_track_bin"] = pd.cut(
        df["prior_track_count"], bins=bins, labels=labels
    )
    return rate_rows(df, ["split", "scope", "user_prior_track_bin"])


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def metric_lookup(
    table: pd.DataFrame,
    *,
    split: str,
    scope: str,
    repeat_type: str,
) -> dict[str, Any]:
    row = table[
        (table["split"] == split)
        & (table["scope"] == scope)
        & (table["repeat_type"] == repeat_type)
    ]
    if row.empty:
        return {"rate": 0.0, "eligible_rate": 0.0, "events": 0, "hits": 0}
    return row.iloc[0].to_dict()


def write_summary(
    path: Path,
    split_table: pd.DataFrame,
    action_table: pd.DataFrame,
    missing_count: int,
    limit: int | None,
) -> None:
    train_session_exact = metric_lookup(
        split_table, split="train", scope="session", repeat_type="exact"
    )
    dev_session_exact = metric_lookup(
        split_table, split="dev_test", scope="session", repeat_type="exact"
    )
    train_user_exact = metric_lookup(
        split_table, split="train", scope="user", repeat_type="exact"
    )
    dev_user_exact = metric_lookup(
        split_table, split="dev_test", scope="user", repeat_type="exact"
    )
    train_session_artist = metric_lookup(
        split_table, split="train", scope="session", repeat_type="artist"
    )
    dev_session_artist = metric_lookup(
        split_table, split="dev_test", scope="session", repeat_type="artist"
    )
    train_user_artist = metric_lookup(
        split_table, split="train", scope="user", repeat_type="artist"
    )
    dev_user_artist = metric_lookup(
        split_table, split="dev_test", scope="user", repeat_type="artist"
    )

    best_action = action_table.sort_values("recall", ascending=False).head(8)
    action_lines = []
    for _, row in best_action.iterrows():
        action_lines.append(
            "- "
            f"{row['split']} / {row['scope']} / {row['source']}@{int(row['topk'])}: "
            f"recall={row['recall']:.4f}, active_rate={row['active_rate']:.4f}"
        )

    limit_note = "full run" if limit is None else f"limit={limit} smoke run"
    text = f"""# 再聴傾向 EDA サマリー

## 実行条件
- run scope: {limit_note}
- analyzed splits: train, dev_test
- metadata join: `input/track-metadata/all_tracks`
- missing metadata rows: {missing_count}
- output scope: `EDA/` only。`mcrs/experiments/` は更新していない。

## 主要結果
- 同一 session 内の exact track 再聴率は train={fmt_pct(train_session_exact['rate'])}、dev_test={fmt_pct(dev_session_exact['rate'])}。
- 同一 user の過去 session 由来の exact track 再聴率は全 event 母数で train={fmt_pct(train_user_exact['rate'])}、dev_test={fmt_pct(dev_user_exact['rate'])}。過去 session が存在する event に限ると train={fmt_pct(train_user_exact['eligible_rate'])}、dev_test={fmt_pct(dev_user_exact['eligible_rate'])}。
- 同一 session 内の artist 継続率は train={fmt_pct(train_session_artist['rate'])}、dev_test={fmt_pct(dev_session_artist['rate'])}。
- 同一 user の過去 session 由来の artist 継続率は train={fmt_pct(train_user_artist['rate'])}、dev_test={fmt_pct(dev_user_artist['rate'])}。

## 解釈
- 会話内では同じ track をそのまま再提示する傾向は観測されず、同じ artist / album / tag 方向に寄せる傾向が強い。
- user の過去 session まで広げると exact track 再登場が観測されるため、再聴 feature は session-local より user-history 側で効く可能性がある。
- tag continuity は非常に広く当たりやすいため、単独の強い signal というより mood/genre continuity の補助指標として扱うのが安全。

## Actionability 上位
{chr(10).join(action_lines)}

## 生成物
- `EDA/tables/relisten_rates_by_split.csv`
- `EDA/tables/relisten_rates_by_turn.csv`
- `EDA/tables/relisten_rates_by_goal.csv`
- `EDA/tables/relisten_rates_by_user_split.csv`
- `EDA/tables/relisten_actionability.csv`
- `EDA/figures/relisten_by_turn.png`
- `EDA/figures/relisten_by_goal.png`
- `EDA/figures/relisten_history_size.png`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def draw_chart_frame(title: str, width: int = 1100, height: int = 700) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 20), title, fill="black", font=font)
    return image, draw


def draw_lines(path: Path, title: str, series: dict[str, list[tuple[int, float]]]) -> None:
    image, draw = draw_chart_frame(title)
    font = ImageFont.load_default()
    left, top, right, bottom = 80, 80, 1040, 620
    draw.rectangle((left, top, right, bottom), outline="black")
    max_x = max((x for values in series.values() for x, _ in values), default=1)
    max_y = max((y for values in series.values() for _, y in values), default=0.01)
    max_y = max(max_y * 1.1, 0.01)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for idx, (name, values) in enumerate(series.items()):
        points = []
        for x, y in values:
            px = left + (x - 1) / max(max_x - 1, 1) * (right - left)
            py = bottom - y / max_y * (bottom - top)
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=colors[idx % len(colors)], width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=colors[idx % len(colors)])
        draw.text((left + 20 + idx * 200, bottom + 35), name, fill=colors[idx % len(colors)], font=font)
    for tick in range(0, 6):
        y = bottom - tick / 5 * (bottom - top)
        value = tick / 5 * max_y
        draw.line((left - 5, y, left, y), fill="black")
        draw.text((20, y - 6), f"{value:.2f}", fill="black", font=font)
    draw.text((left, bottom + 10), "turn_number", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_bars(path: Path, title: str, rows: list[tuple[str, float]]) -> None:
    image, draw = draw_chart_frame(title)
    font = ImageFont.load_default()
    left, top, right, bottom = 260, 80, 1040, 620
    max_value = max((value for _, value in rows), default=0.01)
    max_value = max(max_value * 1.1, 0.01)
    bar_h = max(18, min(42, int((bottom - top) / max(len(rows), 1)) - 8))
    for i, (label, value) in enumerate(rows):
        y = top + i * (bar_h + 8)
        width = value / max_value * (right - left)
        draw.text((20, y + 4), label[:34], fill="black", font=font)
        draw.rectangle((left, y, left + width, y + bar_h), fill="#1f77b4")
        draw.text((left + width + 8, y + 4), f"{value:.3f}", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def create_figures(output_dir: Path, turn_table: pd.DataFrame, goal_table: pd.DataFrame, history_table: pd.DataFrame) -> None:
    figures_dir = output_dir / "figures"
    turn_filtered = turn_table[
        (turn_table["split"] == "dev_test")
        & (turn_table["repeat_type"].isin(["exact", "artist"]))
        & (turn_table["scope"].isin(["session", "user"]))
    ]
    series = {}
    for (scope, repeat_type), group in turn_filtered.groupby(["scope", "repeat_type"]):
        series[f"{scope}_{repeat_type}"] = [
            (int(row["turn_number"]), float(row["rate"]))
            for _, row in group.sort_values("turn_number").iterrows()
        ]
    draw_lines(figures_dir / "relisten_by_turn.png", "Relisten rate by turn (dev_test)", series)

    goal_filtered = goal_table[
        (goal_table["split"] == "dev_test")
        & (goal_table["scope"] == "user")
        & (goal_table["repeat_type"] == "artist")
    ].sort_values("rate", ascending=False)
    goal_rows = [
        (str(row["goal_category"]), float(row["rate"]))
        for _, row in goal_filtered.head(12).iterrows()
    ]
    draw_bars(figures_dir / "relisten_by_goal.png", "User-prior artist continuity by goal (dev_test)", goal_rows)

    hist_filtered = history_table[
        (history_table["split"] == "dev_test")
        & (history_table["repeat_type"].isin(["exact", "artist"]))
    ]
    hist_rows = [
        (f"{row['user_prior_track_bin']} {row['repeat_type']}", float(row["rate"]))
        for _, row in hist_filtered.iterrows()
    ]
    draw_bars(figures_dir / "relisten_history_size.png", "Relisten rate by user history size (dev_test)", hist_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"

    metadata = load_track_metadata(args.track_metadata_dataset)
    artist_to_tracks, album_to_tracks = build_metadata_indexes(metadata)
    conversation_dataset = load_from_disk(args.conversation_dataset)

    all_event_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
    all_missing_rows: list[dict[str, Any]] = []
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    for split in requested_splits:
        sessions = list(conversation_dataset[split])
        if args.limit is not None:
            sessions = sessions[: args.limit]
        event_rows, action_rows, missing_rows = build_event_rows(
            split,
            sessions,
            metadata,
            artist_to_tracks,
            album_to_tracks,
        )
        all_event_rows.extend(event_rows)
        all_action_rows.extend(action_rows)
        all_missing_rows.extend(missing_rows)

    if not all_event_rows:
        raise RuntimeError("No music events were found.")

    event_df = pd.DataFrame(all_event_rows)
    action_df = pd.DataFrame(all_action_rows)

    split_table = rate_rows(event_df, ["split", "scope"]).sort_values(
        ["split", "scope", "repeat_type"]
    )
    turn_table = rate_rows(event_df, ["split", "scope", "turn_number"]).sort_values(
        ["split", "scope", "turn_number", "repeat_type"]
    )
    goal_table = rate_rows(
        event_df, ["split", "scope", "goal_category", "goal_specificity"]
    ).sort_values(["split", "scope", "goal_category", "goal_specificity", "repeat_type"])
    user_split_table = rate_rows(event_df, ["split", "scope", "user_split"]).sort_values(
        ["split", "scope", "user_split", "repeat_type"]
    )
    action_table = summarize_actionability(action_df)
    history_table = history_size_table(event_df)

    write_csv(tables_dir / "relisten_rates_by_split.csv", split_table)
    write_csv(tables_dir / "relisten_rates_by_turn.csv", turn_table)
    write_csv(tables_dir / "relisten_rates_by_goal.csv", goal_table)
    write_csv(tables_dir / "relisten_rates_by_user_split.csv", user_split_table)
    write_csv(tables_dir / "relisten_actionability.csv", action_table)
    write_csv(tables_dir / "relisten_rates_by_history_size.csv", history_table)
    write_jsonl(tables_dir / "relisten_missing_metadata.jsonl", all_missing_rows)

    if not args.no_figures:
        create_figures(output_dir, turn_table, goal_table, history_table)

    write_summary(
        output_dir / "summary" / "relisten-eda-summary.md",
        split_table,
        action_table,
        missing_count=len(all_missing_rows),
        limit=args.limit,
    )

    print(f"events={len(event_df)}")
    print(f"missing_metadata={len(all_missing_rows)}")
    print(f"wrote={output_dir / 'summary' / 'relisten-eda-summary.md'}")


if __name__ == "__main__":
    main()
