"""metadata 完全一致に基づく候補 retrieval。

正規化した catalog metadata 文字列を index 化し、正規化 query n-gram と照合する。
label、未来 turn、生成モデル由来 signal を使わない軽量な候補生成器。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from datasets import concatenate_datasets, load_dataset


DEFAULT_METADATA_FIELDS = ("track_name", "artist_name", "album_name", "tag_list")

FIELD_PRIORITY = {
    "track_name": 0,
    "artist_name": 1,
    "album_name": 2,
    "tag_list": 3,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
}


def normalize_text(value: Any) -> str:
    """英数字のみを残した小文字正規化テキストへ変換する。"""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def as_list(value: Any) -> list[str]:
    """metadata 値を None 安全な文字列リストへ揃える。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _float_value(value: Any) -> float:
    """metadata 値を sorting 用 float へ安全に変換する。"""
    if isinstance(value, list):
        value = value[0] if value else 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


class MetadataExactRetriever:
    """field 優先度を考慮する metadata 完全一致 retriever。"""

    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        metadata_fields: list[str] | tuple[str, ...] = DEFAULT_METADATA_FIELDS,
        cache_dir: str = "./cache",
        min_single_token_chars: int = 4,
        tag_min_single_token_chars: int = 3,
        max_ngram_tokens: int = 12,
    ) -> None:
        """metadata 完全一致 retriever を初期化して index を構築する。

        Args:
            dataset_name: track metadata dataset 名。
            split_types: 読み込む track split。
            metadata_fields: 完全一致照合に使う metadata フィールド。
            cache_dir: loader IF 互換の cache ディレクトリ。この class では永続 cache は使わない。
            min_single_token_chars: 単独 token key として採用する最小文字数。
            tag_min_single_token_chars: tag_list で単独 token key として採用する最小文字数。
            max_ngram_tokens: query から作る n-gram の最大 token 数。
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.metadata_fields = tuple(metadata_fields)
        self.cache_dir = cache_dir
        self.min_single_token_chars = int(min_single_token_chars)
        self.tag_min_single_token_chars = int(tag_min_single_token_chars)
        self.max_ngram_tokens = int(max_ngram_tokens)
        self.metadata_dict = self._load_corpus()
        self.track_ids = list(self.metadata_dict.keys())
        self.fallback_track_ids = self._build_fallback_track_ids()
        self.index = self._build_index()
        self.max_index_key_tokens = self._max_index_key_tokens()

    def _load_corpus(self) -> dict[str, dict[str, Any]]:
        """指定 split の metadata を読み込み、track_id keyed dict にする。"""
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets(
            [metadata_dataset[split_type] for split_type in self.split_types]
        )
        return {str(item["track_id"]): dict(item) for item in metadata_concat_dataset}

    def _build_fallback_track_ids(self) -> list[str]:
        """完全一致候補が不足したときに補完する popularity 順 track リストを作る。"""
        track_ids = list(self.metadata_dict.keys())
        track_ids.sort(
            key=lambda item_id: (
                -_float_value(self.metadata_dict[item_id].get("popularity")),
                item_id,
            )
        )
        return track_ids

    def _build_index(self) -> dict[str, dict[str, list[str]]]:
        """metadata field ごとの正規化文字列 index を構築する。"""
        index: dict[str, dict[str, list[str]]] = {
            field_name: defaultdict(list) for field_name in self.metadata_fields
        }

        for track_id, metadata in self.metadata_dict.items():
            for field_name in self.metadata_fields:
                for raw_value in as_list(metadata.get(field_name)):
                    key = normalize_text(raw_value)
                    if not self._is_valid_key(key, field_name):
                        continue
                    index[field_name][key].append(track_id)

        # 広い artist/tag match は候補が多くなるため、popularity と track_id で順序を安定させる。
        for field_name in self.metadata_fields:
            for key, track_ids in index[field_name].items():
                unique_track_ids = list(dict.fromkeys(track_ids))
                unique_track_ids.sort(
                    key=lambda item_id: (
                        -_float_value(self.metadata_dict[item_id].get("popularity")),
                        item_id,
                    )
                )
                index[field_name][key] = unique_track_ids
        return {field_name: dict(field_index) for field_name, field_index in index.items()}

    def _max_index_key_tokens(self) -> int:
        """index に存在する key 長から query n-gram の上限を決める。"""
        max_tokens = 1
        for field_index in self.index.values():
            for key in field_index:
                max_tokens = max(max_tokens, len(key.split()))
        return min(max_tokens, self.max_ngram_tokens)

    def _is_valid_key(self, key: str, field_name: str) -> bool:
        """index key として短すぎる語や stopword だけの語を除外する。"""
        tokens = key.split()
        if not tokens:
            return False
        if len(tokens) == 1:
            token = tokens[0]
            if token in STOPWORDS:
                return False
            min_chars = (
                self.tag_min_single_token_chars
                if field_name == "tag_list"
                else self.min_single_token_chars
            )
            return len(token) >= min_chars or (
                len(token) >= 2 and any(ch.isdigit() for ch in token)
            )

        content_tokens = [token for token in tokens if token not in STOPWORDS]
        if not content_tokens:
            return False
        return sum(len(token) for token in content_tokens) >= self.min_single_token_chars

    def _query_ngrams(self, query: str) -> list[str]:
        """query から長い順に重複なしの n-gram key を生成する。"""
        tokens = normalize_text(query).split()
        if not tokens:
            return []
        max_size = min(len(tokens), self.max_index_key_tokens)
        seen = set()
        ngrams = []
        for size in range(max_size, 0, -1):
            for start in range(0, len(tokens) - size + 1):
                key = " ".join(tokens[start : start + size])
                if key in seen:
                    continue
                seen.add(key)
                ngrams.append(key)
        return ngrams

    @staticmethod
    def _dedupe(items: list[str], topk: int | None = None) -> list[str]:
        """順序を保ったまま重複を除き、必要なら topk に切り詰める。"""
        deduped = list(dict.fromkeys(items))
        return deduped[:topk] if topk is not None else deduped

    def _ranked_matches(
        self,
        query: str,
        topk: int,
        fields: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        """field priority と match 具体度で完全一致候補を順位付けする。"""
        if topk <= 0:
            return []
        ngrams = self._query_ngrams(query)
        target_fields = tuple(fields) if fields is not None else self.metadata_fields
        best_rank_by_track: dict[str, tuple[float, ...]] = {}

        for field_name in target_fields:
            field_index = self.index.get(field_name, {})
            field_priority = FIELD_PRIORITY.get(field_name, 99)
            matched_count = 0
            for ngram_rank, key in enumerate(ngrams):
                track_ids = field_index.get(key)
                if not track_ids:
                    continue
                key_tokens = len(key.split())
                key_chars = len(key.replace(" ", ""))
                for track_id in track_ids:
                    rank_key = (
                        -float(key_tokens),
                        -float(key_chars),
                        float(field_priority),
                        float(ngram_rank),
                        -_float_value(self.metadata_dict[track_id].get("popularity")),
                    )
                    previous = best_rank_by_track.get(track_id)
                    if previous is None or rank_key < previous:
                        best_rank_by_track[track_id] = rank_key
                matched_count += len(track_ids)
                if matched_count >= topk * 8:
                    break

        ranked_track_ids = [
            track_id
            for track_id, _ in sorted(
                best_rank_by_track.items(),
                key=lambda item: (*item[1], item[0]),
            )
        ]
        return ranked_track_ids[:topk]

    def ranked_exact_matches(
        self,
        query: str,
        topk: int,
        fields: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        """fallback padding なしで metadata 完全一致候補を返す。

        先頭 20 件までは具体的な multi-token match を優先し、不足分は
        field 順の union で広い recall を確保する。

        Args:
            query: 照合する query 文字列。
            topk: 返す候補数。
            fields: 照合対象 field。None なら初期化時の metadata_fields。

        Returns:
            完全一致由来の track ID リスト。
        """
        if topk <= 0:
            return []
        target_fields = tuple(fields) if fields is not None else self.metadata_fields
        ranked = self._ranked_matches(query, topk=min(topk, 20), fields=target_fields)
        if len(ranked) >= topk:
            return ranked[:topk]

        by_field = self.retrieve_by_field(query, topk=topk, fields=target_fields)
        field_ordered = []
        for field_name in sorted(by_field, key=lambda name: FIELD_PRIORITY.get(name, 99)):
            field_ordered.extend(by_field[field_name])
        ranked.extend(field_ordered)
        return self._dedupe(ranked, topk)

    def retrieve_by_field(
        self,
        query: str,
        topk: int = 20,
        fields: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, list[str]]:
        """field ごとに完全一致候補を取得する。

        Args:
            query: 照合する query 文字列。
            topk: field ごとに返す候補数。
            fields: 照合対象 field。None なら初期化時の metadata_fields。

        Returns:
            field 名から候補 track ID リストへの mapping。
        """
        ngrams = self._query_ngrams(query)
        target_fields = tuple(fields) if fields is not None else self.metadata_fields
        results: dict[str, list[str]] = {}
        for field_name in target_fields:
            field_index = self.index.get(field_name, {})
            candidates = []
            for key in ngrams:
                if key not in field_index:
                    continue
                candidates.extend(field_index[key])
                if len(candidates) >= topk * 8:
                    # 広い artist/tag match は巨大になり得るため、重複除去の余裕だけ残して打ち切る。
                    break
            results[field_name] = self._dedupe(candidates, topk)
        return results

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        """metadata 完全一致候補を返し、不足分は popularity fallback で補完する。"""
        candidates = self.ranked_exact_matches(query, topk=topk)
        if len(candidates) < topk:
            seen = set(candidates)
            candidates.extend(
                track_id
                for track_id in self.fallback_track_ids
                if track_id not in seen
            )
        return self._dedupe(candidates, topk)

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        """複数 query に対して metadata 完全一致 retrieval を逐次適用する。"""
        return [self.text_to_item_retrieval(query, topk=topk) for query in queries]
