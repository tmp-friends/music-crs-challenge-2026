"""exp015 の候補生成ルール ablation 用 source builder。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from mcrs.experiments.exp015_candidate_rules_ablation.query_views import build_query_views, dedupe
from mcrs.retrieval_modules.metadata_exact import as_list, normalize_text


BASE_SOURCES = (
    "bm25_recent_turns",
    "bm25_full_context",
    "metadata_bm25_full_context",
    "metadata_bm25_current_user_turn",
    "exact_current_union",
)

EXACT_SPLIT_SOURCES = (
    "exact_track_current",
    "exact_artist_current",
    "exact_album_current",
    "exact_tag_current",
)

METADATA_SPLIT_SOURCES = (
    "metadata_bm25_text_full_context",
    "metadata_bm25_tag_full_context",
    "metadata_bm25_text_current_user_turn",
    "metadata_bm25_tag_current_user_turn",
)

SEGMENT_POPULARITY_SOURCES = (
    "popular_by_artist_current",
    "popular_by_tag_current",
    "popular_by_user_age_country",
    "popular_by_history_artist",
)

HISTORY_SPLIT_SOURCES = (
    "history_same_artist",
    "history_same_tag",
    "history_cf_mean",
    "history_audio_mean",
    "history_recent_neighbors",
)

TAIL_SOURCES = ("popularity_fallback",)

SOURCE_NAMES = (
    "bm25_recent_turns",
    "bm25_full_context",
    "metadata_bm25_full_context",
    "metadata_bm25_current_user_turn",
    "exact_current_union",
    *EXACT_SPLIT_SOURCES,
    *METADATA_SPLIT_SOURCES,
    *SEGMENT_POPULARITY_SOURCES,
    *HISTORY_SPLIT_SOURCES,
    *TAIL_SOURCES,
)

PROFILE_SOURCES = {
    "A": BASE_SOURCES,
    "B": (
        "bm25_recent_turns",
        "bm25_full_context",
        "metadata_bm25_full_context",
        "metadata_bm25_current_user_turn",
        *EXACT_SPLIT_SOURCES,
    ),
    "C": (
        "bm25_recent_turns",
        "bm25_full_context",
        *METADATA_SPLIT_SOURCES,
        *EXACT_SPLIT_SOURCES,
    ),
    "D": (
        "bm25_recent_turns",
        "bm25_full_context",
        *METADATA_SPLIT_SOURCES,
        *EXACT_SPLIT_SOURCES,
        *SEGMENT_POPULARITY_SOURCES,
    ),
    "E": (
        "bm25_recent_turns",
        "bm25_full_context",
        *METADATA_SPLIT_SOURCES,
        *EXACT_SPLIT_SOURCES,
        *SEGMENT_POPULARITY_SOURCES,
        *HISTORY_SPLIT_SOURCES,
    ),
    "F": (
        "bm25_recent_turns",
        "bm25_full_context",
        *METADATA_SPLIT_SOURCES,
        *EXACT_SPLIT_SOURCES,
        *SEGMENT_POPULARITY_SOURCES,
        *HISTORY_SPLIT_SOURCES,
    ),
    "G": (
        "bm25_recent_turns",
        "bm25_full_context",
        *METADATA_SPLIT_SOURCES,
        *EXACT_SPLIT_SOURCES,
        *SEGMENT_POPULARITY_SOURCES,
        *HISTORY_SPLIT_SOURCES,
        "popularity_fallback",
    ),
}

DEFAULT_SOURCE_CAPS = {
    "bm25_recent_turns": 250,
    "bm25_full_context": 250,
    "metadata_bm25_full_context": 400,
    "metadata_bm25_current_user_turn": 200,
    "exact_current_union": 150,
    "exact_track_current": 100,
    "exact_artist_current": 150,
    "exact_album_current": 100,
    "exact_tag_current": 80,
    "metadata_bm25_text_full_context": 350,
    "metadata_bm25_tag_full_context": 150,
    "metadata_bm25_text_current_user_turn": 150,
    "metadata_bm25_tag_current_user_turn": 80,
    "popular_by_artist_current": 50,
    "popular_by_tag_current": 50,
    "popular_by_user_age_country": 50,
    "popular_by_history_artist": 50,
    "history_same_artist": 100,
    "history_same_tag": 100,
    "history_cf_mean": 100,
    "history_audio_mean": 100,
    "history_recent_neighbors": 100,
    "popularity_fallback": 50,
}

DEFAULT_RAW_TOPKS = {
    "bm25_recent_turns": 1000,
    "bm25_full_context": 1000,
    "metadata_bm25_full_context": 1500,
    "metadata_bm25_current_user_turn": 1000,
    "exact_current_union": 500,
    "exact_track_current": 500,
    "exact_artist_current": 500,
    "exact_album_current": 500,
    "exact_tag_current": 500,
    "metadata_bm25_text_full_context": 1500,
    "metadata_bm25_tag_full_context": 1500,
    "metadata_bm25_text_current_user_turn": 1000,
    "metadata_bm25_tag_current_user_turn": 1000,
    "popular_by_artist_current": 500,
    "popular_by_tag_current": 500,
    "popular_by_user_age_country": 500,
    "popular_by_history_artist": 500,
    "history_same_artist": 500,
    "history_same_tag": 500,
    "history_cf_mean": 1000,
    "history_audio_mean": 1000,
    "history_recent_neighbors": 1000,
    "popularity_fallback": 200,
}

QUOTA_GROUPS = {
    "lexical": (
        "bm25_recent_turns",
        "bm25_full_context",
        "metadata_bm25_text_full_context",
        "metadata_bm25_text_current_user_turn",
    ),
    "exact": EXACT_SPLIT_SOURCES,
    "metadata_tags": ("metadata_bm25_tag_full_context", "metadata_bm25_tag_current_user_turn"),
    "segment": SEGMENT_POPULARITY_SOURCES,
    "history": HISTORY_SPLIT_SOURCES,
    "tail": TAIL_SOURCES,
}

DEFAULT_QUOTAS = {
    "lexical": 260,
    "exact": 80,
    "metadata_tags": 70,
    "segment": 40,
    "history": 40,
    "tail": 10,
}


@dataclass(frozen=True)
class SourceRank:
    """候補 track の source 横断 ranking 情報。"""

    score: float
    best_rank: int
    source_count: int
    first_source_order: int


def source_profile_to_sources(source_profile: str | None) -> tuple[str, ...]:
    """source profile ID から有効 source 名のタプルを返す。"""
    profile = str(source_profile or "A").upper()
    if profile not in PROFILE_SOURCES:
        raise ValueError(f"Unknown exp015 source_profile: {source_profile!r}")
    return PROFILE_SOURCES[profile]


def _infer_target_turn_number(history: list[dict[str, Any]]) -> int:
    turn_numbers = [
        int(message.get("turn_number", 0))
        for message in history
        if message.get("turn_number") is not None
    ]
    return max(turn_numbers, default=0) + 1


def _history_track_ids(session_memory: list[dict[str, Any]]) -> list[str]:
    track_ids: list[str] = []
    for message in session_memory:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "music":
            track_ids.append(content)
        elif role == "assistant":
            match = re.search(r"track_id:\s*([0-9a-fA-F-]{36})", content)
            if match:
                track_ids.append(match.group(1))
    return track_ids


def classify_intent(user_query: str) -> str:
    """current user turn を軽量 rule で intent bucket へ分類する。"""
    query = normalize_text(user_query)
    if re.search(r"\b(similar|like this|more like|another one|again|same vibe|reminds me)\b", query):
        return "similarity_request"
    if re.search(r"\b(song|track|artist|album|by|called|named)\b", query):
        return "explicit_entity"
    if re.search(r"\b(mood|genre|vibe|workout|party|sleep|study|chill|dance|sad|happy|rock|pop|jazz|metal|rap|hip hop)\b", query):
        return "mood_or_genre"
    return "ambiguous"


def _value_list(metadata: dict[str, Any], field: str) -> list[str]:
    return [normalize_text(value) for value in as_list(metadata.get(field)) if normalize_text(value)]


def _float_value(value: Any) -> float:
    if isinstance(value, list):
        value = value[0] if value else 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


class SegmentPopularityIndex:
    """train split 由来の segment popularity 候補を保持する。"""

    def __init__(
        self,
        *,
        conversation_dataset_name: str,
        user_dataset_name: str,
        item_db: Any,
        user_split_types: list[str],
        split: str = "train",
    ) -> None:
        """train split の music turn だけから popularity 統計を作る。

        Args:
            conversation_dataset_name: TalkPlayData conversation dataset 名。
            user_dataset_name: user metadata dataset 名。
            item_db: track metadata DB。
            user_split_types: user metadata split。
            split: popularity 統計に使う conversation split。
        """
        self.item_db = item_db
        self.user_profiles = self._load_user_profiles(user_dataset_name, user_split_types)
        self.global_counts: Counter[str] = Counter()
        self.artist_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.tag_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.age_country_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.age_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._build(conversation_dataset_name, split)

    @staticmethod
    def _load_user_profiles(dataset_name: str, split_types: list[str]) -> dict[str, dict[str, Any]]:
        dataset = load_dataset(dataset_name)
        profiles: dict[str, dict[str, Any]] = {}
        for split_type in split_types:
            for item in dataset[split_type]:
                profiles[str(item["user_id"])] = dict(item)
        return profiles

    @staticmethod
    def _normalize_conversations(conversations: Any) -> list[dict[str, Any]]:
        if isinstance(conversations, dict):
            keys = list(conversations.keys())
            return [
                {key: conversations[key][index] for key in keys}
                for index in range(len(conversations[keys[0]]))
            ]
        return [dict(row) for row in conversations]

    def _build(self, conversation_dataset_name: str, split: str) -> None:
        dataset = load_dataset(conversation_dataset_name, split=split)
        for session in dataset:
            user_id = str(session.get("user_id"))
            profile = self.user_profiles.get(user_id, {})
            age_group = str(profile.get("age_group") or "")
            country_code = str(profile.get("country_code") or "")
            age_country_key = f"{age_group}|{country_code}" if age_group or country_code else ""
            for row in self._normalize_conversations(session["conversations"]):
                if row.get("role") != "music":
                    continue
                track_id = str(row.get("content", ""))
                metadata = self.item_db.metadata_dict.get(track_id)
                if metadata is None:
                    continue
                self.global_counts[track_id] += 1
                if age_group:
                    self.age_counts[age_group][track_id] += 1
                if age_country_key:
                    self.age_country_counts[age_country_key][track_id] += 1
                for artist in _value_list(metadata, "artist_name"):
                    self.artist_counts[artist][track_id] += 1
                for tag in _value_list(metadata, "tag_list"):
                    self.tag_counts[tag][track_id] += 1

    @staticmethod
    def _rank(counter: Counter[str], topk: int) -> list[str]:
        return [
            track_id
            for track_id, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:topk]
        ]

    def by_artist(self, artists: list[str], topk: int) -> list[str]:
        merged: Counter[str] = Counter()
        for artist in artists:
            merged.update(self.artist_counts.get(artist, Counter()))
        return self._rank(merged, topk)

    def by_tag(self, tags: list[str], topk: int) -> list[str]:
        merged: Counter[str] = Counter()
        for tag in tags:
            merged.update(self.tag_counts.get(tag, Counter()))
        return self._rank(merged, topk)

    def by_user(self, user_id: str | None, topk: int) -> list[str]:
        profile = self.user_profiles.get(str(user_id), {})
        age_group = str(profile.get("age_group") or "")
        country_code = str(profile.get("country_code") or "")
        age_country_key = f"{age_group}|{country_code}" if age_group or country_code else ""
        if age_country_key and self.age_country_counts.get(age_country_key):
            return self._rank(self.age_country_counts[age_country_key], topk)
        if age_group and self.age_counts.get(age_group):
            return self._rank(self.age_counts[age_group], topk)
        return self._rank(self.global_counts, topk)


class Exp015CandidateBuilder:
    """exp015 の複数 candidate source profile を構築する builder。"""

    def __init__(
        self,
        *,
        item_db: Any,
        baseline_bm25: Any,
        metadata_bm25: Any,
        exact_retriever: Any,
        metadata_text_bm25: Any | None = None,
        metadata_tag_bm25: Any | None = None,
        track_embedding_db: Any | None = None,
        audio_embedding_db: Any | None = None,
        recent_turn_count: int = 2,
        enabled_sources: list[str] | None = None,
        source_profile: str = "A",
        union_mode: str = "rrf",
        source_weights: dict[str, float] | None = None,
        source_caps: dict[str, int] | None = None,
        raw_topks: dict[str, int] | None = None,
        rrf_k: int = 60,
        quota_by_group: dict[str, int] | None = None,
        segment_popularity_index: SegmentPopularityIndex | None = None,
        parallel_bm25_sources: bool = False,
        bm25_source_workers: int = 4,
    ) -> None:
        """候補生成に必要な retriever と source profile を保持する。"""
        self.item_db = item_db
        self.baseline_bm25 = baseline_bm25
        self.metadata_bm25 = metadata_bm25
        self.metadata_text_bm25 = metadata_text_bm25 or metadata_bm25
        self.metadata_tag_bm25 = metadata_tag_bm25 or metadata_bm25
        self.exact_retriever = exact_retriever
        self.track_embedding_db = track_embedding_db
        self.audio_embedding_db = audio_embedding_db
        self.recent_turn_count = int(recent_turn_count)
        self.source_profile = str(source_profile or "A").upper()
        default_sources = source_profile_to_sources(self.source_profile)
        self.source_names = tuple(enabled_sources) if enabled_sources else default_sources
        self.enabled_sources = set(self.source_names)
        self.union_mode = str(union_mode or ("quota" if self.source_profile == "G" else "rrf"))
        self.source_weights = dict(source_weights or {})
        self.source_caps = dict(DEFAULT_SOURCE_CAPS)
        if source_caps:
            self.source_caps.update({k: int(v) for k, v in source_caps.items()})
        self.raw_topks = dict(DEFAULT_RAW_TOPKS)
        if raw_topks:
            self.raw_topks.update({k: int(v) for k, v in raw_topks.items()})
        self.rrf_k = int(rrf_k)
        self.quota_by_group = dict(DEFAULT_QUOTAS)
        if quota_by_group:
            self.quota_by_group.update({k: int(v) for k, v in quota_by_group.items()})
        self.segment_popularity_index = segment_popularity_index
        self.parallel_bm25_sources = bool(parallel_bm25_sources)
        self.bm25_source_workers = int(bm25_source_workers)
        self._popularity_track_ids = list(exact_retriever.fallback_track_ids)
        self._field_keys = {
            field_name: list(exact_retriever.index.get(field_name, {}).keys())
            for field_name in ("artist_name", "tag_list")
        }
        self._artist_index, self._tag_index = self._build_metadata_indexes()
        self._cf_track_matrix, self._cf_track_ids = self._build_track_matrix(track_embedding_db)
        self._audio_track_matrix, self._audio_track_ids = self._build_track_matrix(audio_embedding_db)

    def _build_metadata_indexes(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        artist_index: dict[str, list[str]] = defaultdict(list)
        tag_index: dict[str, list[str]] = defaultdict(list)
        for track_id, metadata in self.item_db.metadata_dict.items():
            for artist in _value_list(metadata, "artist_name"):
                artist_index[artist].append(track_id)
            for tag in _value_list(metadata, "tag_list"):
                tag_index[tag].append(track_id)

        def sort_ids(track_ids: list[str]) -> list[str]:
            unique_ids = list(dict.fromkeys(track_ids))
            unique_ids.sort(
                key=lambda item_id: (
                    -_float_value(self.item_db.metadata_dict[item_id].get("popularity")),
                    item_id,
                )
            )
            return unique_ids

        return (
            {key: sort_ids(values) for key, values in artist_index.items()},
            {key: sort_ids(values) for key, values in tag_index.items()},
        )

    @staticmethod
    def _build_track_matrix(embedding_db: Any | None) -> tuple[torch.Tensor | None, list[str]]:
        if embedding_db is None:
            return None, []
        track_ids: list[str] = []
        vectors: list[torch.Tensor] = []
        for track_id, vector in embedding_db.embeddings.items():
            if embedding_db.is_valid(vector):
                track_ids.append(track_id)
                vectors.append(vector)
        if not vectors:
            return None, []
        return F.normalize(torch.stack(vectors, dim=0), p=2, dim=1), track_ids

    @staticmethod
    def _cosine_topk(
        query_vector: torch.Tensor | None,
        *,
        matrix: torch.Tensor | None,
        track_ids: list[str],
        topk: int,
    ) -> list[str]:
        if query_vector is None or matrix is None or not track_ids:
            return []
        normalized = F.normalize(query_vector.unsqueeze(0), p=2, dim=1).squeeze(0)
        scores = torch.matmul(matrix, normalized)
        k = min(topk, scores.shape[0])
        indices = torch.topk(scores, k=k).indices.tolist()
        return [track_ids[index] for index in indices]

    def query_views(self, *, history: list[dict[str, Any]], user_query: str) -> dict[str, str]:
        query_views = build_query_views(
            history=history,
            user_query=user_query,
            target_turn_number=_infer_target_turn_number(history),
            item_db=self.item_db,
            recent_turn_count=self.recent_turn_count,
        )
        query_views["intent_type"] = classify_intent(user_query)
        return query_views

    def _cap(self, source_name: str, candidates: list[str]) -> list[str]:
        return candidates[: self.source_caps.get(source_name, len(candidates))]

    def _is_allowed_by_intent(self, source_name: str, intent_type: str) -> bool:
        if self.source_profile != "F" and self.source_profile != "G":
            return True
        if source_name in HISTORY_SPLIT_SOURCES:
            return intent_type in {"similarity_request", "ambiguous"}
        if source_name in SEGMENT_POPULARITY_SOURCES:
            return intent_type in {"mood_or_genre", "ambiguous"}
        if source_name in {"exact_tag_current", "metadata_bm25_tag_full_context", "metadata_bm25_tag_current_user_turn"}:
            return intent_type in {"mood_or_genre", "ambiguous", "similarity_request"}
        return True

    def _query_metadata_values(self, query: str, fields: tuple[str, ...]) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = {field: [] for field in fields}
        norm_query = normalize_text(query)
        for field in fields:
            for value in self._field_keys.get(field, []):
                if value and value in norm_query:
                    matches[field].append(value)
        return {field: list(dict.fromkeys(values)) for field, values in matches.items()}

    def _source_exact(self, query: str, source_name: str) -> list[str]:
        field_by_source = {
            "exact_current_union": None,
            "exact_track_current": ["track_name"],
            "exact_artist_current": ["artist_name"],
            "exact_album_current": ["album_name"],
            "exact_tag_current": ["tag_list"],
        }
        fields = field_by_source[source_name]
        return dedupe(
            self.exact_retriever.ranked_exact_matches(
                query,
                topk=self.raw_topks[source_name],
                fields=fields,
            )
        )

    def _source_history_same_artist(self, history: list[dict[str, Any]]) -> list[str]:
        artists: list[str] = []
        for track_id in _history_track_ids(history):
            metadata = self.item_db.metadata_dict.get(track_id, {})
            artists.extend(_value_list(metadata, "artist_name"))
        candidates = []
        for artist in dict.fromkeys(artists):
            candidates.extend(self._artist_index.get(artist, []))
        return dedupe(candidates)

    def _source_history_same_tag(self, history: list[dict[str, Any]]) -> list[str]:
        tags: list[str] = []
        for track_id in _history_track_ids(history):
            metadata = self.item_db.metadata_dict.get(track_id, {})
            tags.extend(_value_list(metadata, "tag_list"))
        candidates = []
        for tag in dict.fromkeys(tags):
            candidates.extend(self._tag_index.get(tag, []))
        return dedupe(candidates)

    def _source_history_embedding(
        self,
        history: list[dict[str, Any]],
        *,
        embedding_db: Any | None,
        matrix: torch.Tensor | None,
        track_ids: list[str],
        source_name: str,
    ) -> list[str]:
        if embedding_db is None:
            return []
        history_ids = _history_track_ids(history)
        if not history_ids:
            return []
        vector = embedding_db.mean_vector(history_ids)
        if not embedding_db.is_valid(vector):
            return []
        return self._cosine_topk(
            vector,
            matrix=matrix,
            track_ids=track_ids,
            topk=self.raw_topks[source_name],
        )

    def _source_history_recent_neighbors(self, history: list[dict[str, Any]]) -> list[str]:
        if self.track_embedding_db is None:
            return []
        history_ids = _history_track_ids(history)
        for track_id in reversed(history_ids):
            vector = self.track_embedding_db.get(track_id)
            if self.track_embedding_db.is_valid(vector):
                return self._cosine_topk(
                    vector,
                    matrix=self._cf_track_matrix,
                    track_ids=self._cf_track_ids,
                    topk=self.raw_topks["history_recent_neighbors"],
                )
        return []

    def _source_segment_popularity(
        self,
        source_name: str,
        query_views: dict[str, str],
        history: list[dict[str, Any]],
        user_id: str | None,
    ) -> list[str]:
        if self.segment_popularity_index is None:
            return []
        topk = self.raw_topks[source_name]
        if source_name == "popular_by_user_age_country":
            return self.segment_popularity_index.by_user(user_id, topk)
        query_matches = self._query_metadata_values(
            query_views["bm25_current_user_turn"],
            ("artist_name", "tag_list"),
        )
        if source_name == "popular_by_artist_current":
            return self.segment_popularity_index.by_artist(query_matches["artist_name"], topk)
        if source_name == "popular_by_tag_current":
            return self.segment_popularity_index.by_tag(query_matches["tag_list"], topk)
        if source_name == "popular_by_history_artist":
            artists: list[str] = []
            for track_id in _history_track_ids(history):
                artists.extend(_value_list(self.item_db.metadata_dict.get(track_id, {}), "artist_name"))
            return self.segment_popularity_index.by_artist(list(dict.fromkeys(artists)), topk)
        return []

    def _dispatch_source(
        self,
        source_name: str,
        query_views: dict[str, str],
        history: list[dict[str, Any]],
        user_id: str | None,
        bm25_batches: dict[str, list[str]] | None = None,
    ) -> tuple[list[str], bool]:
        if source_name not in self.enabled_sources:
            return [], False
        intent_type = query_views.get("intent_type", "ambiguous")
        if not self._is_allowed_by_intent(source_name, intent_type):
            return [], False
        if bm25_batches and source_name in bm25_batches:
            return dedupe(bm25_batches[source_name]), True
        current_query = query_views["bm25_current_user_turn"]
        full_query = query_views["bm25_full_context"]
        if source_name == "bm25_recent_turns":
            return dedupe(self.baseline_bm25.text_to_item_retrieval(query_views["bm25_recent_turns"], self.raw_topks[source_name])), True
        if source_name == "bm25_full_context":
            return dedupe(self.baseline_bm25.text_to_item_retrieval(full_query, self.raw_topks[source_name])), True
        if source_name == "metadata_bm25_full_context":
            return dedupe(self.metadata_bm25.text_to_item_retrieval(full_query, self.raw_topks[source_name])), True
        if source_name == "metadata_bm25_current_user_turn":
            return dedupe(self.metadata_bm25.text_to_item_retrieval(current_query, self.raw_topks[source_name])), True
        if source_name == "metadata_bm25_text_full_context":
            return dedupe(self.metadata_text_bm25.text_to_item_retrieval(full_query, self.raw_topks[source_name])), True
        if source_name == "metadata_bm25_tag_full_context":
            return dedupe(self.metadata_tag_bm25.text_to_item_retrieval(full_query, self.raw_topks[source_name])), True
        if source_name == "metadata_bm25_text_current_user_turn":
            return dedupe(self.metadata_text_bm25.text_to_item_retrieval(current_query, self.raw_topks[source_name])), True
        if source_name == "metadata_bm25_tag_current_user_turn":
            return dedupe(self.metadata_tag_bm25.text_to_item_retrieval(current_query, self.raw_topks[source_name])), True
        if source_name.startswith("exact_"):
            return self._source_exact(current_query, source_name), True
        if source_name in SEGMENT_POPULARITY_SOURCES:
            items = self._source_segment_popularity(source_name, query_views, history, user_id)
            return items, bool(items)
        if source_name == "history_same_artist":
            items = self._source_history_same_artist(history)
            return items, bool(items)
        if source_name == "history_same_tag":
            items = self._source_history_same_tag(history)
            return items, bool(items)
        if source_name == "history_cf_mean":
            items = self._source_history_embedding(
                history,
                embedding_db=self.track_embedding_db,
                matrix=self._cf_track_matrix,
                track_ids=self._cf_track_ids,
                source_name=source_name,
            )
            return items, bool(items)
        if source_name == "history_audio_mean":
            items = self._source_history_embedding(
                history,
                embedding_db=self.audio_embedding_db,
                matrix=self._audio_track_matrix,
                track_ids=self._audio_track_ids,
                source_name=source_name,
            )
            return items, bool(items)
        if source_name == "history_recent_neighbors":
            items = self._source_history_recent_neighbors(history)
            return items, bool(items)
        if source_name == "popularity_fallback":
            return self._popularity_track_ids[: self.raw_topks[source_name]], True
        raise ValueError(f"Unsupported exp015 source: {source_name}")

    def candidates_by_source(
        self,
        *,
        history: list[dict[str, Any]],
        user_query: str,
        user_id: str | None,
    ) -> tuple[dict[str, list[str]], dict[str, str], dict[str, bool]]:
        """1 task 分の source 別候補を生成する。"""
        query_views = self.query_views(history=history, user_query=user_query)
        candidates: dict[str, list[str]] = {}
        active: dict[str, bool] = {}
        for source_name in self.source_names:
            items, is_active = self._dispatch_source(source_name, query_views, history, user_id)
            candidates[source_name] = self._cap(source_name, items)
            active[source_name] = is_active
        return candidates, query_views, active

    def batch_candidates_by_source(
        self,
        *,
        histories: list[list[dict[str, Any]]],
        user_queries: list[str],
        user_ids: list[str | None],
    ) -> tuple[list[dict[str, list[str]]], list[dict[str, str]], list[dict[str, bool]]]:
        """複数 task の候補を生成する。BM25 系だけ batch API を使う。"""
        query_views_list = [
            self.query_views(history=history, user_query=user_query)
            for history, user_query in zip(histories, user_queries)
        ]
        query_by_source = {
            "bm25_recent_turns": ("baseline", [v["bm25_recent_turns"] for v in query_views_list]),
            "bm25_full_context": ("baseline", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_full_context": ("metadata", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_current_user_turn": ("metadata", [v["bm25_current_user_turn"] for v in query_views_list]),
            "metadata_bm25_text_full_context": ("metadata_text", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_tag_full_context": ("metadata_tag", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_text_current_user_turn": ("metadata_text", [v["bm25_current_user_turn"] for v in query_views_list]),
            "metadata_bm25_tag_current_user_turn": ("metadata_tag", [v["bm25_current_user_turn"] for v in query_views_list]),
        }
        retriever_by_kind = {
            "baseline": self.baseline_bm25,
            "metadata": self.metadata_bm25,
            "metadata_text": self.metadata_text_bm25,
            "metadata_tag": self.metadata_tag_bm25,
        }
        batch_results: dict[str, list[list[str]]] = {}
        for source_name, (kind, queries) in query_by_source.items():
            if source_name not in self.enabled_sources:
                continue
            batch_results[source_name] = retriever_by_kind[kind].batch_text_to_item_retrieval(
                queries,
                topk=self.raw_topks[source_name],
            )

        all_candidates: list[dict[str, list[str]]] = []
        all_active: list[dict[str, bool]] = []
        for index, query_views in enumerate(query_views_list):
            bm25_batches = {
                source_name: results[index]
                for source_name, results in batch_results.items()
            }
            candidates: dict[str, list[str]] = {}
            active: dict[str, bool] = {}
            for source_name in self.source_names:
                items, is_active = self._dispatch_source(
                    source_name,
                    query_views,
                    histories[index],
                    user_ids[index],
                    bm25_batches=bm25_batches,
                )
                candidates[source_name] = self._cap(source_name, items)
                active[source_name] = is_active
            all_candidates.append(candidates)
            all_active.append(active)
        return all_candidates, query_views_list, all_active

    def batch_candidates_with_scores(
        self,
        *,
        histories: list[list[dict[str, Any]]],
        user_queries: list[str],
        user_ids: list[str | None],
    ) -> tuple[
        list[dict[str, list[str]]],
        list[dict[str, str]],
        list[dict[str, bool]],
        list[dict[str, dict[str, float]]],
    ]:
        """``batch_candidates_by_source`` に加え bm25 系 source の実 score も返す。

        候補集合（track ID リスト）の作り方は ``batch_candidates_by_source`` と完全に
        同一で、追加で bm25 / metadata-bm25 系 source の ``{track_id: bm25_score}`` を
        返す。Stage C の source confidence feature が train / inference の両経路で同じ
        score を参照できるようにするための拡張で、``union()`` の候補決定ロジックには
        一切影響しない（score は feature 計算専用）。exact 系など score を持たない
        source は score 辞書に含めない（後段で 0 fallback 扱い）。

        Args:
            histories: 各 task の対話履歴。
            user_queries: 各 task の現在 user query。
            user_ids: 各 task の user ID。

        Returns:
            ``(candidates_list, query_views_list, active_list, scores_list)``。
            ``scores_list[i]`` は ``{source_name: {track_id: bm25_score}}``。
        """
        query_views_list = [
            self.query_views(history=history, user_query=user_query)
            for history, user_query in zip(histories, user_queries)
        ]
        # batch_candidates_by_source と同じ source → query / retriever マッピング
        query_by_source = {
            "bm25_recent_turns": ("baseline", [v["bm25_recent_turns"] for v in query_views_list]),
            "bm25_full_context": ("baseline", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_full_context": ("metadata", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_current_user_turn": ("metadata", [v["bm25_current_user_turn"] for v in query_views_list]),
            "metadata_bm25_text_full_context": ("metadata_text", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_tag_full_context": ("metadata_tag", [v["bm25_full_context"] for v in query_views_list]),
            "metadata_bm25_text_current_user_turn": ("metadata_text", [v["bm25_current_user_turn"] for v in query_views_list]),
            "metadata_bm25_tag_current_user_turn": ("metadata_tag", [v["bm25_current_user_turn"] for v in query_views_list]),
        }
        retriever_by_kind = {
            "baseline": self.baseline_bm25,
            "metadata": self.metadata_bm25,
            "metadata_text": self.metadata_text_bm25,
            "metadata_tag": self.metadata_tag_bm25,
        }
        # score 付き batch retrieval を 1 回だけ実行し、id-only batch は score から導出する
        batch_scored: dict[str, list[list[tuple[str, float]]]] = {}
        batch_results: dict[str, list[list[str]]] = {}
        for source_name, (kind, queries) in query_by_source.items():
            if source_name not in self.enabled_sources:
                continue
            scored = retriever_by_kind[kind].batch_text_to_item_retrieval_with_scores(
                queries,
                topk=self.raw_topks[source_name],
            )
            batch_scored[source_name] = scored
            batch_results[source_name] = [[tid for tid, _ in row] for row in scored]

        all_candidates: list[dict[str, list[str]]] = []
        all_active: list[dict[str, bool]] = []
        all_scores: list[dict[str, dict[str, float]]] = []
        for index, query_views in enumerate(query_views_list):
            bm25_batches = {
                source_name: results[index]
                for source_name, results in batch_results.items()
            }
            # source ごとの {track_id: score}。重複 track_id は降順先頭（最高 score）を保持
            score_lookup: dict[str, dict[str, float]] = {}
            for source_name, scored in batch_scored.items():
                lookup: dict[str, float] = {}
                for tid, score in scored[index]:
                    lookup.setdefault(tid, score)
                score_lookup[source_name] = lookup
            candidates: dict[str, list[str]] = {}
            active: dict[str, bool] = {}
            scores: dict[str, dict[str, float]] = {}
            for source_name in self.source_names:
                items, is_active = self._dispatch_source(
                    source_name,
                    query_views,
                    histories[index],
                    user_ids[index],
                    bm25_batches=bm25_batches,
                )
                capped = self._cap(source_name, items)
                candidates[source_name] = capped
                active[source_name] = is_active
                # cap / dedupe 後の track_id にだけ score を対応付ける（score は位置非依存）
                if source_name in score_lookup:
                    lookup = score_lookup[source_name]
                    scores[source_name] = {
                        tid: lookup[tid] for tid in capped if tid in lookup
                    }
            all_candidates.append(candidates)
            all_active.append(active)
            all_scores.append(scores)
        return all_candidates, query_views_list, all_active, all_scores

    def union(self, candidates_by_source: dict[str, list[str]], topk: int) -> list[str]:
        """source profile に応じて候補を統合する。"""
        if self.union_mode == "quota":
            return self._quota_union(candidates_by_source, topk)
        return self._rrf_union(candidates_by_source, topk, self.source_names)

    def _rrf_union(
        self,
        candidates_by_source: dict[str, list[str]],
        topk: int,
        source_names: tuple[str, ...],
    ) -> list[str]:
        rank_by_track: dict[str, SourceRank] = {}
        source_order = {name: index for index, name in enumerate(source_names)}
        for source_name in source_names:
            for rank, track_id in enumerate(candidates_by_source.get(source_name, []), start=1):
                score_delta = 1.0 / float(self.rrf_k + rank)
                previous = rank_by_track.get(track_id)
                if previous is None:
                    rank_by_track[track_id] = SourceRank(score_delta, rank, 1, source_order[source_name])
                    continue
                rank_by_track[track_id] = SourceRank(
                    previous.score + score_delta,
                    min(previous.best_rank, rank),
                    previous.source_count + 1,
                    min(previous.first_source_order, source_order[source_name]),
                )
        ranked = sorted(
            rank_by_track.items(),
            key=lambda item: (-item[1].score, -item[1].source_count, item[1].best_rank, item[1].first_source_order, item[0]),
        )
        return [track_id for track_id, _ in ranked[:topk]]

    def _quota_union(self, candidates_by_source: dict[str, list[str]], topk: int) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for group_name, group_sources in QUOTA_GROUPS.items():
            quota = self.quota_by_group.get(group_name, 0)
            if quota <= 0:
                continue
            group_ranked = self._rrf_union(candidates_by_source, quota * 3, group_sources)
            group_added = 0
            for track_id in group_ranked:
                if track_id in seen:
                    continue
                selected.append(track_id)
                seen.add(track_id)
                group_added += 1
                if group_added >= quota:
                    break
        if len(selected) < topk:
            fill = self._rrf_union(candidates_by_source, topk * 2, self.source_names)
            for track_id in fill:
                if track_id not in seen:
                    selected.append(track_id)
                    seen.add(track_id)
                if len(selected) >= topk:
                    break
        return selected[:topk]
