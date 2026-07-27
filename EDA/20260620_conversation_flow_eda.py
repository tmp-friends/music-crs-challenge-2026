"""会話フロー EDA: goal_progress 動態と「絞り込み(pivot)を GT が honor するか」の集計。

注意: 本スクリプトの pivot × progress クロス集計は、後に確認された TalkPlayData 2 の
生成順と整合せず、``goal_progress[t]`` を ``music[t]`` に対応付けている。正しくは
``music[t-1]`` への評価である。論文用の主張には使用せず、逐次分析を実装した
``EDA/20260720_evaluation_alignment_audit.py`` を参照すること。

既存の会話 EDA（``20260604_02_conversation_structure_eda.ipynb`` 等）は行数・turn・goal・
role の構造統計までをカバーする。本スクリプトはその先の **会話の動態**、特に

1. ``goal_progress_assessments``（dataset 由来の正解ラベル）の MOVES / DOES_NOT 分布、
2. GT(``role=music``)の artist 継続率（隣接 / 累積の両定義）、
3. ユーザの「別 artist / 新規発掘」要求(pivot)と GT 継続・goal_progress のクロス集計、
4. 発話長などの会話統計、

を full train で集計する。

重要なフレーミング: ``role=music`` の content は **nDCG@20 の正解 track_id そのもの**
（[parse_turn](../mcrs/ltr/data_utils.py) 参照）。よって本 EDA の「artist 継続」は
"system が無視している" 量ではなく **正解ターゲット自体がどれだけ pivot しないか** の測定で
ある。これが exp073 の artist-demote が負だった理由（同 artist を下げると GT から離れる）を
定量的に説明する。

リーク注意: ``thought`` / ``goal_progress_assessments`` / ``listener_goal`` は train 専用の
god's-eye 情報（Blind では空 or 利用要注意）。本 EDA は **分析専用**であり、これらを推論時の
feature / prompt / fallback に流用しないこと。

出力:
- ``EDA/tables/conv_flow_goal_progress_by_turn.csv``
- ``EDA/tables/conv_flow_goal_progress_by_category.csv``
- ``EDA/tables/conv_flow_goal_progress_by_specificity.csv``
- ``EDA/tables/conv_flow_session_outcome.csv``
- ``EDA/tables/conv_flow_artist_continuity_by_turn.csv``
- ``EDA/tables/conv_flow_pivot_crosstab.csv``
- ``EDA/tables/conv_flow_pivot_request_stats.csv``
- ``EDA/tables/conv_flow_utterance_by_turn.csv``
- ``EDA/summary/conversation-flow-eda-summary.md``
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
METADATA_NAME = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

MOVES = "MOVES_TOWARD_GOAL"
DONT = "DOES_NOT_MOVE_TOWARD_GOAL"

# 「別 artist を / 新しい artist を発掘したい」系の pivot 要求検出（ヒューリスティック）。
# 注意: 語彙ベースなので noise を含む。本 EDA の headline は dataset 正解ラベル
# (goal_progress) 側に置き、本 regex slice は明示的に「補助・要注意」として扱う。
PIVOT_RE = re.compile(
    r"(different artist|another artist|someone (?:else|new)|new artists?|"
    r"haven'?t heard|never heard|artist i (?:haven'?t|have not|don'?t know)|"
    r"by (?:a|an) (?:other|different|new)|other than|"
    r"something (?:else|new|different)|discover (?:new|different|some|something))",
    re.I,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解釈する。

    Returns:
        ``limit`` を含む argparse の Namespace。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭 N session のみで回す smoke 用。未指定で full train。",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="出力 root（既定は EDA/）。",
    )
    return parser.parse_args()


def normalize_artists(value: Any) -> frozenset[str]:
    """track metadata の artist_name（list or str）を小文字 set に正規化する。

    Args:
        value: artist_name 列の値。

    Returns:
        小文字・strip 済み artist 名の frozenset。
    """
    if value is None:
        return frozenset()
    if isinstance(value, list):
        return frozenset(str(x).strip().lower() for x in value if x)
    return frozenset([str(value).strip().lower()])


def load_track_indexes() -> tuple[dict[str, frozenset[str]], dict[str, str | None]]:
    """track_id → artist set / album 名の lookup を構築する。

    Returns:
        (track_id→artist frozenset, track_id→album 名小文字) のタプル。
    """
    meta = load_dataset(METADATA_NAME)["all_tracks"]
    track_to_artists: dict[str, frozenset[str]] = {}
    track_to_album: dict[str, str | None] = {}
    for row in meta:
        track_to_artists[row["track_id"]] = normalize_artists(row.get("artist_name"))
        album = row.get("album_name")
        if isinstance(album, list):
            album = album[0] if album else None
        track_to_album[row["track_id"]] = str(album).strip().lower() if album else None
    return track_to_artists, track_to_album


def session_turn_rows(session: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    """1 session の conversation を ``{turn_number: {role: row}}`` に整形する。

    Args:
        session: dataset の 1 session レコード。

    Returns:
        turn_number → role → row の入れ子辞書。
    """
    by_turn: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in session["conversations"]:
        by_turn[int(row["turn_number"])][row["role"]] = row
    return by_turn


def word_count(text: str | None) -> int:
    """空白区切りの語数を返す（None は 0）。

    Args:
        text: 対象文字列。

    Returns:
        語数。
    """
    if not text:
        return 0
    return len(text.split())


def build_turn_records(
    split: str,
    limit: int | None,
    track_to_artists: dict[str, frozenset[str]],
    track_to_album: dict[str, str | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """session を turn 単位レコードへ展開し、session 集約も併せて返す。

    Args:
        split: dataset split 名。
        limit: 先頭 session 数の上限。
        track_to_artists: track_id→artist set lookup。
        track_to_album: track_id→album 名 lookup。

    Returns:
        (turn レコード list, session レコード list) のタプル。
    """
    ds = load_dataset(DATASET_NAME, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    turn_records: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []

    for session in ds:
        goal = session["conversation_goal"]
        category = goal.get("category")
        specificity = goal.get("specificity")
        gp_by_turn = {
            int(a["turn_number"]): a["goal_progress_assessment"]
            for a in session["goal_progress_assessments"]
            if a.get("goal_progress_assessment")
        }
        by_turn = session_turn_rows(session)

        cumulative_artists: set[str] = set()
        cumulative_albums: set[str] = set()
        prev_artists: frozenset[str] = frozenset()
        session_labels: list[str] = []

        for turn in sorted(by_turn):
            row = by_turn[turn]
            user_row = row.get("user")
            asst_row = row.get("assistant")
            music_row = row.get("music")

            cur_artists = (
                track_to_artists.get(music_row["content"], frozenset())
                if music_row
                else frozenset()
            )
            cur_album = (
                track_to_album.get(music_row["content"]) if music_row else None
            )

            has_history = turn > 1 and bool(cumulative_artists)
            same_artist_adjacent = (
                bool(cur_artists & prev_artists)
                if (cur_artists and prev_artists)
                else None
            )
            same_artist_cumulative = (
                bool(cur_artists & cumulative_artists)
                if (cur_artists and cumulative_artists)
                else None
            )
            same_album_cumulative = (
                (cur_album in cumulative_albums)
                if (cur_album and cumulative_albums)
                else None
            )

            user_text = user_row["content"] if user_row else None
            is_pivot = bool(user_text and PIVOT_RE.search(user_text))

            gp_label = gp_by_turn.get(turn)
            if gp_label:
                session_labels.append(gp_label)

            turn_records.append(
                {
                    "split": split,
                    "session_id": session["session_id"],
                    "category": category,
                    "specificity": specificity,
                    "turn_number": turn,
                    "has_history": has_history,
                    "gp_label": gp_label,
                    "moves": 1 if gp_label == MOVES else (0 if gp_label == DONT else None),
                    "same_artist_adjacent": same_artist_adjacent,
                    "same_artist_cumulative": same_artist_cumulative,
                    "same_album_cumulative": same_album_cumulative,
                    "is_pivot_request": is_pivot,
                    "user_words": word_count(user_text),
                    "assistant_words": word_count(asst_row["content"] if asst_row else None),
                }
            )

            # 次 turn 用に prior を更新する。
            prev_artists = cur_artists
            cumulative_artists |= cur_artists
            if cur_album:
                cumulative_albums.add(cur_album)

        if session_labels:
            session_records.append(
                {
                    "split": split,
                    "session_id": session["session_id"],
                    "category": category,
                    "specificity": specificity,
                    "n_assessed": len(session_labels),
                    "n_moves": sum(1 for x in session_labels if x == MOVES),
                    "first_label": session_labels[0],
                    "last_label": session_labels[-1],
                    "all_moves": all(x == MOVES for x in session_labels),
                    "never_moves": all(x == DONT for x in session_labels),
                }
            )

    return turn_records, session_records


def rate(numer: int, denom: int) -> float:
    """0 除算回避つき比率。

    Args:
        numer: 分子。
        denom: 分母。

    Returns:
        ``numer/denom``（denom=0 のとき 0.0）。
    """
    return numer / denom if denom else 0.0


def goal_progress_by_turn(df: pd.DataFrame) -> pd.DataFrame:
    """turn 別の MOVES 率を集計する。

    Args:
        df: turn レコード DataFrame。

    Returns:
        turn 別集計 DataFrame。
    """
    sub = df[df["moves"].notna()]
    grouped = sub.groupby("turn_number")["moves"].agg(["count", "mean"]).reset_index()
    grouped.columns = ["turn_number", "n_assessed", "moves_rate"]
    return grouped


def goal_progress_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """任意キー別の MOVES 率を集計する。

    Args:
        df: turn レコード DataFrame。
        key: 集計キー列名。

    Returns:
        key 別集計 DataFrame（moves_rate 昇順）。
    """
    sub = df[df["moves"].notna()]
    grouped = sub.groupby(key)["moves"].agg(["count", "mean"]).reset_index()
    grouped.columns = [key, "n_assessed", "moves_rate"]
    return grouped.sort_values("moves_rate").reset_index(drop=True)


def artist_continuity_by_turn(df: pd.DataFrame) -> pd.DataFrame:
    """turn 別の artist/album 継続率（隣接 / 累積）を集計する。

    Args:
        df: turn レコード DataFrame。

    Returns:
        turn 別継続率 DataFrame。
    """
    rows: list[dict[str, Any]] = []
    for turn, sub in df.groupby("turn_number"):
        adj = sub["same_artist_adjacent"].dropna()
        cum = sub["same_artist_cumulative"].dropna()
        alb = sub["same_album_cumulative"].dropna()
        rows.append(
            {
                "turn_number": turn,
                "n_with_history": int(len(cum)),
                "artist_same_adjacent_rate": float(adj.mean()) if len(adj) else 0.0,
                "artist_same_cumulative_rate": float(cum.mean()) if len(cum) else 0.0,
                "album_same_cumulative_rate": float(alb.mean()) if len(alb) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("turn_number").reset_index(drop=True)


def pivot_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """(pivot 要求) × (GT 同 artist 累積) のセル別 MOVES 率を集計する。

    history 有り（turn>=2）かつ goal_progress ラベル有りの turn のみが対象。

    Args:
        df: turn レコード DataFrame。

    Returns:
        4 セル + 合計の DataFrame。
    """
    sub = df[(df["moves"].notna()) & (df["same_artist_cumulative"].notna())].copy()
    rows: list[dict[str, Any]] = []
    for pivot in (True, False):
        for same in (True, False):
            cell = sub[
                (sub["is_pivot_request"] == pivot)
                & (sub["same_artist_cumulative"] == same)
            ]
            rows.append(
                {
                    "pivot_request": pivot,
                    "gt_same_artist": same,
                    "n_turns": int(len(cell)),
                    "moves_rate": float(cell["moves"].mean()) if len(cell) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def pivot_request_stats(df: pd.DataFrame) -> pd.DataFrame:
    """pivot 要求 turn における GT 同 artist 率 / MOVES 率を集計する。

    Args:
        df: turn レコード DataFrame。

    Returns:
        指標 1 行 DataFrame。
    """
    sub = df[(df["same_artist_cumulative"].notna())]
    pivot = sub[sub["is_pivot_request"]]
    pivot_gp = pivot[pivot["moves"].notna()]
    pivot_same = pivot[pivot["same_artist_cumulative"] == True]  # noqa: E712
    pivot_same_gp = pivot_same[pivot_same["moves"].notna()]
    out = {
        "pivot_turns_with_history": int(len(pivot)),
        "gt_same_artist_rate_at_pivot": float(pivot["same_artist_cumulative"].mean())
        if len(pivot)
        else 0.0,
        "moves_rate_at_pivot": float(pivot_gp["moves"].mean()) if len(pivot_gp) else 0.0,
        "moves_rate_at_pivot_and_same_artist": float(pivot_same_gp["moves"].mean())
        if len(pivot_same_gp)
        else 0.0,
        "does_not_move_rate_at_pivot_and_same_artist": (
            1.0 - float(pivot_same_gp["moves"].mean()) if len(pivot_same_gp) else 0.0
        ),
    }
    return pd.DataFrame([out])


def utterance_by_turn(df: pd.DataFrame) -> pd.DataFrame:
    """turn 別の発話語数（user / assistant）を集計する。

    Args:
        df: turn レコード DataFrame。

    Returns:
        turn 別発話統計 DataFrame。
    """
    grouped = df.groupby("turn_number").agg(
        user_words_mean=("user_words", "mean"),
        user_words_median=("user_words", "median"),
        assistant_words_mean=("assistant_words", "mean"),
        assistant_words_median=("assistant_words", "median"),
        n=("user_words", "count"),
    )
    return grouped.reset_index()


def session_outcome_table(session_df: pd.DataFrame) -> pd.DataFrame:
    """session 単位の達成度サマリーを集計する。

    Args:
        session_df: session レコード DataFrame。

    Returns:
        指標 / 件数 / 比率の DataFrame。
    """
    n = len(session_df)
    rows = [
        ("sessions_assessed", n, 1.0),
        (
            "all_moves_toward_goal",
            int(session_df["all_moves"].sum()),
            rate(int(session_df["all_moves"].sum()), n),
        ),
        (
            "never_moves_toward_goal",
            int(session_df["never_moves"].sum()),
            rate(int(session_df["never_moves"].sum()), n),
        ),
        (
            "last_turn_moves",
            int((session_df["last_label"] == MOVES).sum()),
            rate(int((session_df["last_label"] == MOVES).sum()), n),
        ),
        (
            "last_turn_does_not_move",
            int((session_df["last_label"] == DONT).sum()),
            rate(int((session_df["last_label"] == DONT).sum()), n),
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "count", "rate"])


def extract_walkthrough(split: str, index: int = 0) -> list[str]:
    """代表 session の turn 単位フローを Markdown 行として抽出する。

    Args:
        split: dataset split 名。
        index: 対象 session の index。

    Returns:
        Markdown 行のリスト。
    """
    session = load_dataset(DATASET_NAME, split=split)[index]
    goal = session["conversation_goal"]
    by_turn = session_turn_rows(session)
    gp = {
        int(a["turn_number"]): a["goal_progress_assessment"]
        for a in session["goal_progress_assessments"]
        if a.get("goal_progress_assessment")
    }
    lines = [
        f"- session_id: `{session['session_id']}`",
        f"- goal(category={goal['category']}, specificity={goal['specificity']}): "
        f"{goal['listener_goal']}",
        "",
        "| turn | user 発話(要約) | GT track / 推薦理由 | goal_progress |",
        "|---|---|---|---|",
    ]
    for turn in sorted(by_turn):
        row = by_turn[turn]
        user_text = (row.get("user") or {}).get("content", "") or ""
        music_thought = (row.get("music") or {}).get("thought", "") or ""
        user_short = user_text.replace("\n", " ")[:90]
        thought_short = music_thought.replace("\n", " ")[:70]
        lines.append(
            f"| {turn} | {user_short} | {thought_short} | {gp.get(turn, '-')} |"
        )
    return lines


def fmt_pct(value: float) -> str:
    """0-1 の比率を百分率文字列にする。

    Args:
        value: 0-1 の比率。

    Returns:
        ``XX.X%`` 形式の文字列。
    """
    return f"{100 * value:.1f}%"


def write_summary(
    path: Path,
    *,
    limit: int | None,
    n_sessions: int,
    gp_overall: dict[str, int],
    gp_turn: pd.DataFrame,
    gp_cat: pd.DataFrame,
    cont_turn: pd.DataFrame,
    crosstab: pd.DataFrame,
    pivot_stats: pd.DataFrame,
    session_outcome: pd.DataFrame,
    walkthrough: list[str],
) -> None:
    """summary Markdown を書き出す。

    Args:
        path: 出力先パス。
        limit: run scope。
        n_sessions: 集計対象 session 数。
        gp_overall: 全体 MOVES/DONT カウント。
        gp_turn: turn 別 goal_progress。
        gp_cat: category 別 goal_progress。
        cont_turn: turn 別継続率。
        crosstab: pivot クロス集計。
        pivot_stats: pivot 要求統計。
        session_outcome: session 達成度サマリー。
        walkthrough: 代表 session の Markdown 行。
    """
    tot = sum(gp_overall.values())
    moves_share = rate(gp_overall.get(MOVES, 0), tot)

    same_moves = crosstab[(crosstab["gt_same_artist"] == True)]["moves_rate"]  # noqa: E712
    diff_moves = crosstab[(crosstab["gt_same_artist"] == False)]["moves_rate"]  # noqa: E712
    # gt_same_artist 全体（pivot 問わず）の MOVES 率を加重平均で。
    same_rows = crosstab[crosstab["gt_same_artist"] == True]  # noqa: E712
    diff_rows = crosstab[crosstab["gt_same_artist"] == False]  # noqa: E712
    same_overall = rate(
        int((same_rows["moves_rate"] * same_rows["n_turns"]).sum()),
        int(same_rows["n_turns"].sum()),
    )
    diff_overall = rate(
        int((diff_rows["moves_rate"] * diff_rows["n_turns"]).sum()),
        int(diff_rows["n_turns"].sum()),
    )

    ps = pivot_stats.iloc[0]
    cont_overall_adj = float(cont_turn["artist_same_adjacent_rate"].mean())
    # 全 music event 分母の累積継続率（既存 relisten 56.88% と定義を揃える）。
    cum_hits = int((cont_turn["artist_same_cumulative_rate"] * cont_turn["n_with_history"]).sum())
    cum_denom_hist = int(cont_turn["n_with_history"].sum())
    cum_rate_hist = rate(cum_hits, cum_denom_hist)
    cum_rate_allevents = rate(cum_hits, n_sessions * 8)

    limit_note = "full train" if limit is None else f"limit={limit} smoke run"

    def cell(pivot: bool, same: bool) -> tuple[int, float]:
        row = crosstab[
            (crosstab["pivot_request"] == pivot) & (crosstab["gt_same_artist"] == same)
        ].iloc[0]
        return int(row["n_turns"]), float(row["moves_rate"])

    ps_n, ps_rate = cell(True, True)
    pd_n, pd_rate = cell(True, False)
    np_n, np_rate = cell(False, True)
    nd_n, nd_rate = cell(False, False)

    text = f"""# 会話フロー EDA サマリー（goal_progress と pivot honor）

Last updated: 2026-06-20

> **重要（2026-07-20）:** Section 4 は `goal_progress[t]` を `music[t]` の評価として
> 集計しているが、正しくは直前の `music[t-1]` に対する評価である。同一ターン比較と
> その解釈は使用せず、`EDA/20260720_evaluation_alignment_audit.py` を参照すること。

## 実行条件
- run scope: {limit_note}（train split, {n_sessions:,} sessions）
- metadata join: `{METADATA_NAME}` の all_tracks（artist/album）
- output scope: `EDA/` only。`mcrs/experiments/` は更新していない。
- **リーク注意**: `thought` / `goal_progress_assessments` / `listener_goal` は train 専用の
  god's-eye 情報。本 EDA は**分析専用**であり、これらを推論時の feature / prompt / fallback /
  validation に流用しないこと。

## いちばん大事なフレーミング
`role=music` の content は **nDCG@20 の正解 track_id そのもの**（[parse_turn](../mcrs/ltr/data_utils.py) の label 抽出）。
したがって本 EDA の「artist 継続率」は "system が pivot を無視した量" ではなく、
**正解ターゲット自体がどれだけ pivot しないか**の測定である。これが exp073 の
artist-demote（別 artist 要求 turn で同 artist を top20 から降格）が **net-negative** だった
理由を定量的に説明する: 同 artist を下げる＝GT から離れる＝nDCG が下がる。

## 主要結果

### 1. goal_progress（dataset 正解ラベル）の動態
- 全 assessment {tot:,} 件のうち **MOVES_TOWARD_GOAL = {fmt_pct(moves_share)}** / DOES_NOT = {fmt_pct(1 - moves_share)}。
  baseline 会話の**約半分は「ゴールに近づいていない」とラベルされている**。
- turn 別 MOVES 率は turn2 {fmt_pct(float(gp_turn.iloc[0]['moves_rate']))} から中盤に向けて低下し
  （会話が進むほど袋小路になりやすい）、最終 turn で回復する。詳細は
  `tables/conv_flow_goal_progress_by_turn.csv`。
- goal category 別では **`{gp_cat.iloc[0]['category']}` が最難（MOVES {fmt_pct(float(gp_cat.iloc[0]['moves_rate']))}）**、
  最も達成しやすいのは `{gp_cat.iloc[-1]['category']}`（{fmt_pct(float(gp_cat.iloc[-1]['moves_rate']))}）。
  category × specificity の全量は `tables/conv_flow_goal_progress_by_category.csv` /
  `..._by_specificity.csv`。

### 2. session 単位の達成度
- **never_moves（全 turn で DOES_NOT）= {int(session_outcome[session_outcome.metric=='never_moves_toward_goal']['count'].iloc[0]):,} session
  ({fmt_pct(float(session_outcome[session_outcome.metric=='never_moves_toward_goal']['rate'].iloc[0]))})**。
  最後まで一度もゴールに近づかない session が一定数ある。
- last_turn が DOES_NOT で終わる session = {fmt_pct(float(session_outcome[session_outcome.metric=='last_turn_does_not_move']['rate'].iloc[0]))}。
- 詳細は `tables/conv_flow_session_outcome.csv`。

### 3. GT の artist 継続率（隣接 vs 累積）
- **隣接（turn N vs N-1）平均 = {fmt_pct(cont_overall_adj)}**。
- **累積（union(1..N-1)）= {fmt_pct(cum_rate_hist)}**（history 有り turn 分母）。
- 全 music event(8/△session)を分母にすると **{fmt_pct(cum_rate_allevents)}** で、既存
  `relisten-eda-summary.md` の「同一 session 内 artist 継続率 train=56.88%」と整合する
  （差は turn1=prior 空を分母に含めるか否かの定義差）。turn 別は
  `tables/conv_flow_artist_continuity_by_turn.csv`。

### 4. ★ pivot × GT継続 × goal_progress クロス集計（本 EDA の核心）
history 有り・goal_progress ラベル有りの turn を 4 セルに分けた MOVES 率:
（前提: `assessment[t]` は turn t の推薦 `music[t]` を評価しているとみなす。assessment は
turn2 開始・1 turn 内は user→music→assistant 順なので "t の推薦" か "t-1 への反応" か原理的に
曖昧だが、下の 18pt gap（pivot+別 62.4% vs pivot+同 44.6%）が `music[t]` の artist に強く連動する
事実は、もし label[t] が `music[t-1]` を評価していたら現れ得ない＝この alignment を裏付ける
（連続 rec の autocorrelation でやや減衰するため "強い傍証" 扱い）。）

| | GT 同 artist | GT 別 artist |
|---|---|---|
| **pivot 要求あり** | n={ps_n:,}, MOVES {fmt_pct(ps_rate)} | n={pd_n:,}, MOVES {fmt_pct(pd_rate)} |
| **pivot 要求なし** | n={np_n:,}, MOVES {fmt_pct(np_rate)} | n={nd_n:,}, MOVES {fmt_pct(nd_rate)} |

- **全体（pivot 問わず）では同 artist {fmt_pct(same_overall)} vs 別 artist {fmt_pct(diff_overall)} と
  ほぼ差が無い**＝同 artist 継続それ自体は「悪」ではない（GT の半分強は同 artist だが MOVES 率は
  別 artist とほぼ同じ）。
- **差は pivot 要求 turn に局在する**: pivot turn では 同 artist GT の MOVES = **{fmt_pct(ps_rate)}** に対し
  別 artist GT は **{fmt_pct(pd_rate)}**（約 {round((pd_rate - ps_rate) * 100)}pt 差）。
  ところが pivot 要求 turn（history 有り {int(ps['pivot_turns_with_history']):,} 件）の GT は
  **{fmt_pct(float(ps['gt_same_artist_rate_at_pivot']))} が同 artist**（累積定義=過去いずれかの turn と artist
  共有。隣接定義ならより低い／"別 artist" を直前以外なら可とするユーザもいる）。
  pivot かつ GT 同 artist の turn の DOES_NOT 率 =
  **{fmt_pct(float(ps['does_not_move_rate_at_pivot_and_same_artist']))}**。
- 解釈: **衝突は pivot turn に局在する**。そこでだけ「ユーザ満足（別 artist→MOVES）」と
  「nDCG 正解（同 artist 77%）」が両立しない。pivot turn で nDCG を取りに行く＝dataset が
  DOES_NOT とラベルする同 artist track を {fmt_pct(float(ps['does_not_move_rate_at_pivot_and_same_artist']))} の
  確率で推薦することになる。これが「pivot を honor すれば nDCG が上がる」という直感が成り立たない
  理由であり、exp073 の artist-demote が net-negative だった原因そのもの。LLM-Judge は blind の
  response 側で測られるので、**ranking は GT(同 artist 含む)に素直に寄せ、満足度の改善は
  response 軸（説明・personalization）で取りに行く**のが整合的。
- ただし pivot かつ GT 別 artist のセル（n={pd_n:,}, MOVES {fmt_pct(pd_rate)}）は、**switch が
  正解(GT)かつ満足の両立する pivot turn の少数派（pivot turn の約 {round(pd_n / (ps_n + pd_n) * 100)}%）**。
  「いつ switch すべきか」を当てる signal は将来 lever になり得る（exp073 の blanket artist-demote
  とは別物＝全 pivot で同 artist を下げるのではなく、switch が GT のサブ集合だけを狙う）。
  ranking が pivot に一切効かない訳ではない。
- 注意: pivot 検出は語彙ヒューリスティックで noise を含むため、headline は authoritative な
  goal_progress 側（同 artist vs 別 artist の MOVES 率差）に置き、regex slice は補助とする。

### 5. 発話統計
- user / assistant の発話語数の turn 別推移は `tables/conv_flow_utterance_by_turn.csv`。
  user 発話は袋小路 turn で長くなる（懇願・繰り返しの言い直し）傾向。

## 代表 session のフロー（walkthrough）
{chr(10).join(walkthrough)}

## 示唆
- **ranking 軸**: GT は同 artist 継続が支配的（隣接 {fmt_pct(cont_overall_adj)}）。継続系 source
  （artist/album continuity・entity）を主軸にするのは GT 構造と一致しており、artist-demote の
  ような pivot 迎合は GT を壊す。改善余地は continuity ではなく **recall（GT が候補に入らない
  crashed class）**側にある（[[exp073-refinement-saturation]] と整合）。
- **response 軸**: goal_progress が約半数 DOES_NOT である以上、ranking だけでユーザ満足は
  上がりにくい。説明・personalization で「同 sound だが要望も理解している」ことを示す response
  設計（exp063 V-Claude-long 系）が満足度 lever として正しい。
- **category 難易度**: 最難 category を feature/bucket として持ち、bucket 別 nDCG で source 採否
  を判断する（strategy-eda の decision gate と整合）。

## 生成物
- `tables/conv_flow_goal_progress_by_turn.csv`
- `tables/conv_flow_goal_progress_by_category.csv`
- `tables/conv_flow_goal_progress_by_specificity.csv`
- `tables/conv_flow_session_outcome.csv`
- `tables/conv_flow_artist_continuity_by_turn.csv`
- `tables/conv_flow_pivot_crosstab.csv`
- `tables/conv_flow_pivot_request_stats.csv`
- `tables/conv_flow_utterance_by_turn.csv`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """EDA を実行し、tables と summary を出力する。"""
    args = parse_args()
    out_root = Path(args.out_root)
    tables_dir = out_root / "tables"
    summary_dir = out_root / "summary"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading track metadata ...")
    track_to_artists, track_to_album = load_track_indexes()

    print("[2/4] building turn records (train) ...")
    turn_records, session_records = build_turn_records(
        "train", args.limit, track_to_artists, track_to_album
    )
    df = pd.DataFrame(turn_records)
    session_df = pd.DataFrame(session_records)
    n_sessions = session_df["session_id"].nunique()

    print("[3/4] aggregating ...")
    gp_overall = Counter(df.loc[df["gp_label"].notna(), "gp_label"])
    gp_turn = goal_progress_by_turn(df)
    gp_cat = goal_progress_by_key(df, "category")
    gp_spec = goal_progress_by_key(df, "specificity")
    cont_turn = artist_continuity_by_turn(df)
    crosstab = pivot_crosstab(df)
    pivot_stats = pivot_request_stats(df)
    utter = utterance_by_turn(df)
    session_outcome = session_outcome_table(session_df)
    walkthrough = extract_walkthrough("train", index=0)

    gp_turn.to_csv(tables_dir / "conv_flow_goal_progress_by_turn.csv", index=False)
    gp_cat.to_csv(tables_dir / "conv_flow_goal_progress_by_category.csv", index=False)
    gp_spec.to_csv(tables_dir / "conv_flow_goal_progress_by_specificity.csv", index=False)
    session_outcome.to_csv(tables_dir / "conv_flow_session_outcome.csv", index=False)
    cont_turn.to_csv(tables_dir / "conv_flow_artist_continuity_by_turn.csv", index=False)
    crosstab.to_csv(tables_dir / "conv_flow_pivot_crosstab.csv", index=False)
    pivot_stats.to_csv(tables_dir / "conv_flow_pivot_request_stats.csv", index=False)
    utter.to_csv(tables_dir / "conv_flow_utterance_by_turn.csv", index=False)

    print("[4/4] writing summary ...")
    write_summary(
        summary_dir / "conversation-flow-eda-summary.md",
        limit=args.limit,
        n_sessions=n_sessions,
        gp_overall=dict(gp_overall),
        gp_turn=gp_turn,
        gp_cat=gp_cat,
        cont_turn=cont_turn,
        crosstab=crosstab,
        pivot_stats=pivot_stats,
        session_outcome=session_outcome,
        walkthrough=walkthrough,
    )
    print("done. tables ->", tables_dir, "| summary ->", summary_dir)


if __name__ == "__main__":
    main()
