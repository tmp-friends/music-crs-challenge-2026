"""exp015 内で完結する BM25 query view 生成 helper。"""

from __future__ import annotations

from typing import Any

from mcrs.db_item import MusicCatalogDB


def dedupe(items: list[str]) -> list[str]:
    """track_id リストの順序を保ったまま重複を除去する。

    Args:
        items: 重複を含み得る track_id リスト。

    Returns:
        初出順で一意化した track_id リスト。
    """
    return list(dict.fromkeys(items))


def normalize_message(message: dict[str, Any], item_db: MusicCatalogDB) -> dict[str, str]:
    """BM25 クエリ用に会話メッセージを文字列化する。

    role="music" の場合は track_id を track metadata に展開し、検索語として
    track_name / artist_name などが BM25 に届くようにする。

    Args:
        message: 会話メッセージ。
        item_db: track_id から metadata を引く MusicCatalogDB。

    Returns:
        role と content を持つ正規化済みメッセージ。
    """
    role = message.get("role")
    content = message.get("content", "")
    if role == "music":
        role = "assistant"
        try:
            content = item_db.id_to_metadata(str(content))
        except KeyError:
            content = str(content)
    elif role not in {"user", "assistant", "system"}:
        role = "assistant"
        content = str(content)
    return {"role": str(role), "content": str(content)}


def normalize_dialogue_message(message: dict[str, Any]) -> dict[str, str] | None:
    """dialogue-only view 用に music role を除いたメッセージを返す。

    Args:
        message: 会話メッセージ。

    Returns:
        正規化済みメッセージ。music role の場合は None。
    """
    role = message.get("role")
    if role == "music":
        return None
    content = str(message.get("content", ""))
    if role not in {"user", "assistant", "system"}:
        role = "assistant"
    return {"role": str(role), "content": content}


def format_messages(messages: list[dict[str, str]]) -> str:
    """BM25 に渡すため role 付きメッセージ列を改行結合する。

    Args:
        messages: role と content を持つメッセージ列。

    Returns:
        ``role: content`` 形式の複数行テキスト。
    """
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def build_query_views(
    history: list[dict[str, Any]],
    user_query: str,
    target_turn_number: int,
    item_db: MusicCatalogDB,
    recent_turn_count: int,
) -> dict[str, str]:
    """履歴と最新発話から複数の BM25 query view を作る。

    Args:
        history: 現在 turn より前の会話履歴。
        user_query: 現在 turn のユーザ発話。
        target_turn_number: 現在 turn 番号。
        item_db: music role を metadata に展開するための item DB。
        recent_turn_count: recent view に含める直近 turn 数。

    Returns:
        full/current/recent/dialogue-only の query view 辞書。
    """
    current_user = {"role": "user", "content": str(user_query)}

    full_messages = [normalize_message(message, item_db) for message in history]
    full_messages.append(current_user)

    min_recent_turn = max(1, target_turn_number - recent_turn_count)
    recent_history = [
        message
        for message in history
        if min_recent_turn <= int(message.get("turn_number", 0)) < target_turn_number
    ]
    recent_messages = [normalize_message(message, item_db) for message in recent_history]
    recent_messages.append(current_user)

    dialogue_messages = []
    for message in history:
        normalized = normalize_dialogue_message(message)
        if normalized is not None:
            dialogue_messages.append(normalized)
    dialogue_messages.append(current_user)

    return {
        "bm25_full_context": format_messages(full_messages),
        "bm25_current_user_turn": format_messages([current_user]),
        "bm25_recent_turns": format_messages(recent_messages),
        "bm25_dialogue_only": format_messages(dialogue_messages),
    }
