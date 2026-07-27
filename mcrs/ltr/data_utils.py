"""dataset build で使う config / task helper（共有ライブラリ）。

exp014 由来。``mcrs/ltr/`` へ切り出した共有モジュールで、旧
``mcrs.experiments.exp014_lightgbm_ltr_reranker.data_utils`` は re-export shim 経由で解決される。
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset


def config_list(config: Any, key: str, default: list[str]) -> list[str]:
    """OmegaConf config の list 値を Python の文字列リストとして取得する。

    Args:
        config: OmegaConf DictConfig など ``get`` を持つ config。
        key: 取得するキー。
        default: 値が存在しない場合に使うデフォルト。

    Returns:
        文字列化済みのリスト。
    """
    value = config.get(key, None)
    if value is None:
        return list(default)
    return [str(item) for item in list(value)]


def validate_config(config: Any) -> None:
    """exp014 の候補生成 config として安全な前提を検証する。

    Args:
        config: 候補生成 config。

    Raises:
        ValueError: retrieval_type または track_split_types が exp014 の前提と異なる場合。
    """
    if config.retrieval_type != "bm25":
        raise ValueError(f"exp014 expects retrieval_type='bm25', got {config.retrieval_type!r}")
    if list(config.track_split_types) != ["all_tracks"]:
        raise ValueError("exp014 dataset build must use track_split_types: ['all_tracks']")


def normalize_conversations(conversations: Any) -> list[dict[str, Any]]:
    """datasets が返す conversation 表現を dict 行リストにそろえる。

    Args:
        conversations: dict-of-list または list-of-dict の conversation データ。

    Returns:
        conversation row のリスト。
    """
    if isinstance(conversations, dict):
        keys = list(conversations.keys())
        if not keys:
            return []
        length = len(conversations[keys[0]])
        return [
            {key: conversations[key][index] for key in keys}
            for index in range(length)
        ]
    return [dict(row) for row in conversations]


def parse_turn(
    conversations: list[dict[str, Any]],
    target_turn_number: int,
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """指定 turn の履歴・ユーザ発話・正解 track_id を取り出す。

    Args:
        conversations: 1 session の conversation row リスト。
        target_turn_number: 抽出対象の turn 番号。

    Returns:
        (履歴, ユーザ発話, 正解 track_id リスト) のタプル。
    """
    history = [
        row
        for row in conversations
        if int(row.get("turn_number", 0)) < target_turn_number
    ]
    current_turn = [
        row
        for row in conversations
        if int(row.get("turn_number", 0)) == target_turn_number
    ]
    user_rows = [row for row in current_turn if row.get("role") == "user"]
    if not user_rows:
        return history, None, []
    user_query = str(user_rows[0].get("content", ""))
    labels = [
        str(row.get("content", ""))
        for row in current_turn
        if row.get("role") == "music"
    ]
    return history, user_query, labels


def prepare_tasks(config: Any, split: str, limit: int | None) -> list[dict[str, Any]]:
    """TalkPlayData の session を session-turn 単位タスクへ展開する。

    Args:
        config: ``test_dataset_name`` を含む候補生成 config。
        split: Hugging Face dataset split 名。
        limit: session 件数上限。None の場合は全件。

    Returns:
        build_dataset.py が処理する task 辞書のリスト。
    """
    db = load_dataset(config.test_dataset_name, split=split)
    tasks: list[dict[str, Any]] = []
    for item_index, item in enumerate(db):
        if limit is not None and item_index >= limit:
            break
        conversations = normalize_conversations(item["conversations"])
        target_turn_numbers = sorted(
            {
                int(row.get("turn_number", 0))
                for row in conversations
                if row.get("role") == "user"
            }
        )
        for target_turn_number in target_turn_numbers:
            history, user_query, labels = parse_turn(conversations, target_turn_number)
            if user_query is None:
                continue
            tasks.append(
                {
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "turn_number": target_turn_number,
                    "history": history,
                    "user_query": user_query,
                    "labels": labels,
                }
            )
    return tasks
