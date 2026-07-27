"""exp015 LightGBM LTR の特徴量抽出 utility。"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F

from mcrs.experiments.exp015_candidate_rules_ablation.candidate_sources import (
    SOURCE_NAMES,
    SourceRank,
    source_profile_to_sources,
)


# Run A の text ベース主要 5 ソース。build_dataset.py の default に使う。
PRIMARY_TEXT5_SOURCES = source_profile_to_sources("A")


def as_list(value: Any) -> list[str]:
    """任意の値を文字列リストに変換する。

    Args:
        value: 変換対象の値。None / リスト / スカラーを受け付ける。

    Returns:
        文字列のリスト。None 要素は除外される。
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def normalize_text(value: Any) -> str:
    """テキストを小文字化し、英数字以外を空白に置換して正規化する。

    Args:
        value: 正規化対象の値。

    Returns:
        正規化済みの文字列。
    """
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: Any) -> set[str]:
    """正規化テキストのトークン集合を返す。

    Args:
        value: トークン化対象の値。

    Returns:
        正規化済みトークンの集合。
    """
    return set(normalize_text(value).split())


def phrase_in_text(phrase: str, text: str) -> bool:
    """フレーズが text 内に単語境界を持って含まれるか判定する。

    Args:
        phrase: 検索するフレーズ（正規化済み）。
        text: 検索対象テキスト（正規化済み）。

    Returns:
        フレーズが含まれる場合 True。
    """
    # 前後にスペースを付加して部分文字列マッチを防ぐ（例: "rock" が "stockrock" にマッチしない）
    return bool(phrase) and f" {phrase} " in f" {text} "


def history_track_ids(session_memory: list[dict[str, Any]]) -> list[str]:
    """対話履歴から過去に再生・推薦された track_id リストを抽出する。

    Args:
        session_memory: 対話履歴のメッセージリスト。各メッセージは
            role / content キーを持つ辞書。

    Returns:
        抽出された track_id のリスト（登場順）。
    """
    track_ids: list[str] = []
    for message in session_memory:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "music":
            # role="music" は content がそのまま track_id
            track_ids.append(content)
        elif role == "assistant":
            # role="assistant" では "track_id: <UUID>" 形式の埋め込みから抽出する
            match = re.search(r"track_id:\s*([0-9a-fA-F-]{36})", content)
            if match:
                track_ids.append(match.group(1))
    return track_ids


def safe_cosine(reference: torch.Tensor | None, candidate: torch.Tensor | None) -> float:
    """ゼロベクトルガード付きのコサイン類似度を返す。

    Args:
        reference: 参照ベクトル。
        candidate: 候補ベクトル。

    Returns:
        コサイン類似度（0.0 以上 1.0 以下）。いずれかが無効な場合は 0.0。
    """
    if not is_valid_vector(reference) or not is_valid_vector(candidate):
        return 0.0
    score = F.cosine_similarity(reference.unsqueeze(0), candidate.unsqueeze(0), dim=1).item()
    # 負のコサインは「反対向き」を意味するが候補のペナルティにはしない（0 クリップ）
    return max(float(score), 0.0)


def is_valid_vector(vector: torch.Tensor | None) -> bool:
    """ベクトルが有効（非 None・非空・非ゼロノルム）かを判定する。

    Args:
        vector: 検査するテンソル。

    Returns:
        有効なベクトルの場合 True。
    """
    return vector is not None and vector.numel() > 0 and torch.linalg.vector_norm(vector).item() > 0.0


def mean_vector(embedding_db: Any | None, track_ids: list[str]) -> torch.Tensor | None:
    """embedding DB から複数 track_id の平均ベクトルを取得する。

    Args:
        embedding_db: mean_vector メソッドを持つ embedding DB。None の場合は None を返す。
        track_ids: 平均化する track_id のリスト。

    Returns:
        平均ベクトル。DB が None または mean_vector 未実装の場合は None。
    """
    if embedding_db is None:
        return None
    if hasattr(embedding_db, "mean_vector"):
        return embedding_db.mean_vector(track_ids)
    return None


def max_history_cosine(embedding_db: Any | None, track_ids: list[str], candidate_id: str) -> float:
    """候補 track と履歴 track 群の最大コサイン類似度を返す。

    Args:
        embedding_db: embedding DB。
        track_ids: 履歴 track_id のリスト。
        candidate_id: 類似度を計算する候補 track_id。

    Returns:
        最大コサイン類似度。DB が None または履歴が空の場合は 0.0。
    """
    if embedding_db is None or not track_ids:
        return 0.0
    candidate = embedding_db.get(candidate_id)
    if not is_valid_vector(candidate):
        return 0.0
    scores = [safe_cosine(embedding_db.get(track_id), candidate) for track_id in track_ids]
    return max(scores) if scores else 0.0


def parse_release_year(metadata: dict[str, Any]) -> float:
    """メタデータから release_date / release_year / year を探して西暦年を返す。

    Args:
        metadata: track のメタデータ辞書。

    Returns:
        4 桁の西暦年（float）。見つからない場合は 0.0。
    """
    for key in ("release_date", "release_year", "year"):
        value = metadata.get(key)
        if value is None:
            continue
        match = re.search(r"(19|20)\d{2}", str(value))
        if match:
            return float(match.group(0))
    return 0.0


class LGBMFeatureBuilder:
    """候補ごとの pointwise 特徴量行を構築するクラス。

    offline 学習（build_dataset.py）と online 推論（lightgbm_ltr.py）の両方で
    同一の特徴量定義を共有する。
    """

    def __init__(
        self,
        *,
        item_db: Any,
        track_embedding_db: Any | None = None,
        user_embedding_db: Any | None = None,
        audio_embedding_db: Any | None = None,
        metadata_embedding_db: Any | None = None,
        source_names: tuple[str, ...] = SOURCE_NAMES,
        source_weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        """各種 DB とパラメータを受け取って初期化する。

        Args:
            item_db: track メタデータを保持する MusicCatalogDB。
            track_embedding_db: CF-BPR track embedding DB。
            user_embedding_db: CF-BPR user embedding DB。
            audio_embedding_db: audio embedding DB（LAION CLAP）。
            metadata_embedding_db: metadata embedding DB（任意）。
            source_names: 全候補ソース名のタプル（特徴量列名の生成に使用）。
            source_weights: 互換用の source 別重み。現在の RRF 計算では使用しない。
            rrf_k: RRF の平滑化定数（デフォルト 60）。
        """
        self.item_db = item_db
        self.track_embedding_db = track_embedding_db
        self.user_embedding_db = user_embedding_db
        self.audio_embedding_db = audio_embedding_db
        self.metadata_embedding_db = metadata_embedding_db
        self.source_names = tuple(source_names)
        self.source_weights = dict(source_weights or {})
        self.rrf_k = int(rrf_k)

    def build_rows(
        self,
        *,
        candidates_by_source: dict[str, list[str]],
        union_candidates: list[str],
        query_views: dict[str, str],
        user_id: str | None,
        session_memory: list[dict[str, Any]],
        session_id: str | None = None,
        turn_number: int | None = None,
        labels: list[str] | None = None,
        topk: int | None = None,
        scores_by_source: dict[str, dict[str, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """全候補の特徴量行リストを構築するメインメソッド。

        Args:
            candidates_by_source: ソース名 → track_id リストのマップ。
            union_candidates: 全候補の統合リスト（重複可、先頭優先で dedup される）。
            query_views: BM25 query の各ビュー辞書。
            user_id: ユーザ ID。
            session_memory: 対話履歴。
            session_id: セッション ID（行の識別子として保持）。
            turn_number: ターン番号（行の識別子として保持）。
            labels: 正例 track_id のリスト（学習時に label 列を生成するために使用）。
            topk: 上位何件の候補を処理するか。None で全件。
            scores_by_source: ソース名 → {track_id: retrieval score} のマップ。
                Stage C 系の source confidence feature 用。None の場合は base 実装の
                フックが no-op になり、従来どおりの特徴量のみを生成する。

        Returns:
            各候補 1 行に対応する特徴量辞書のリスト。
        """
        candidates = list(dict.fromkeys(union_candidates))
        if topk is not None:
            candidates = candidates[:topk]

        source_rank_maps = {
            source: {track_id: rank for rank, track_id in enumerate(items, start=1)}
            for source, items in candidates_by_source.items()
        }
        rrf_by_track = self._source_rank_features(candidates_by_source)
        # score 由来の追加特徴量用コンテキスト。base 実装は None を返す no-op。
        score_context = self._prepare_score_context(candidates_by_source, scores_by_source)

        label_set = set(labels or [])
        full_query = query_views.get("bm25_full_context") or query_views.get("bm25_recent_turns") or ""
        current_query = query_views.get("bm25_current_user_turn") or ""
        recent_query = query_views.get("bm25_recent_turns") or current_query
        intent_type = str(query_views.get("intent_type") or "ambiguous")
        full_tokens = token_set(full_query)
        current_tokens = token_set(current_query)
        recent_tokens = token_set(recent_query)

        hist_ids = history_track_ids(session_memory)
        hist_artists, hist_tags = self._history_metadata_sets(hist_ids)

        cf_history_mean = mean_vector(self.track_embedding_db, hist_ids)
        audio_history_mean = mean_vector(self.audio_embedding_db, hist_ids)
        metadata_history_mean = mean_vector(self.metadata_embedding_db, hist_ids)
        user_vector = self.user_embedding_db.get(user_id) if self.user_embedding_db is not None else None

        rows: list[dict[str, Any]] = []
        for candidate_rank, track_id in enumerate(candidates, start=1):
            metadata = self.item_db.metadata_dict.get(track_id, {})
            row: dict[str, Any] = {
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(turn_number or 0),
                "track_id": track_id,
                "candidate_rank": candidate_rank,
                "label": 1 if track_id in label_set else 0,
            }
            row.update(self._generic_rank_features(candidate_rank))
            row.update(self._source_features(track_id, source_rank_maps, rrf_by_track))
            # source score 由来の追加特徴量。base 実装は空辞書を返す no-op。
            row.update(
                self._extra_source_features(
                    track_id, source_rank_maps, rrf_by_track, score_context
                )
            )
            row.update(
                self._lexical_metadata_features(
                    metadata=metadata,
                    full_query=full_query,
                    current_query=current_query,
                    full_tokens=full_tokens,
                    current_tokens=current_tokens,
                    recent_tokens=recent_tokens,
                )
            )
            row.update(
                self._history_features(
                    metadata=metadata,
                    hist_artists=hist_artists,
                    hist_tags=hist_tags,
                    hist_ids=hist_ids,
                    track_id=track_id,
                    cf_history_mean=cf_history_mean,
                    audio_history_mean=audio_history_mean,
                    metadata_history_mean=metadata_history_mean,
                    user_vector=user_vector,
                )
            )
            row.update(
                {
                    "f_turn_number": float(turn_number or 0),
                    "f_history_music_count": float(len(hist_ids)),
                    "f_full_query_len": float(len(full_tokens)),
                    "f_current_query_len": float(len(current_tokens)),
                    "f_recent_query_len": float(len(recent_tokens)),
                    "f_has_explicit_source_hit": float(
                        any(
                            track_id in source_rank_maps.get(source_name, {})
                            for source_name in (
                                "exact_current_union",
                                "exact_track_current",
                                "exact_artist_current",
                                "exact_album_current",
                            )
                        )
                    ),
                    "f_intent_explicit_entity": float(intent_type == "explicit_entity"),
                    "f_intent_mood_or_genre": float(intent_type == "mood_or_genre"),
                    "f_intent_similarity_request": float(intent_type == "similarity_request"),
                    "f_intent_ambiguous": float(intent_type == "ambiguous"),
                    "f_missing_user_embedding": 0.0 if is_valid_vector(user_vector) else 1.0,
                    "f_missing_history_embedding": 0.0 if is_valid_vector(cf_history_mean) else 1.0,
                }
            )
            rows.append(row)
        return rows

    def _prepare_score_context(
        self,
        candidates_by_source: dict[str, list[str]],
        scores_by_source: dict[str, dict[str, float]] | None,
    ) -> Any:
        """source score 由来特徴量の事前計算コンテキストを返すフック。

        base 実装は no-op で常に None を返す。source confidence feature を持つ
        サブクラス（exp015d 等）が override し、build_rows のループ前に 1 回だけ
        実行される per-task の集約値（source ごとの max score 等）を返す。

        Args:
            candidates_by_source: ソース名 → track_id リストのマップ。
            scores_by_source: ソース名 → {track_id: score} のマップ（None 可）。

        Returns:
            ループ内 ``_extra_source_features`` へ渡すコンテキスト。base では None。
        """
        del candidates_by_source, scores_by_source
        return None

    def _extra_source_features(
        self,
        track_id: str,
        source_rank_maps: dict[str, dict[str, int]],
        rrf_by_track: dict[str, SourceRank],
        score_context: Any,
    ) -> dict[str, float | None]:
        """候補ごとの追加 source 特徴量を返すフック。

        base 実装は no-op で空辞書を返すため、既存実験の出力は不変。サブクラスが
        override して ``f_conf_*`` 等の confidence feature を追加する。

        Args:
            track_id: 対象候補の track_id。
            source_rank_maps: ソース名 → {track_id: rank} のマップ。
            rrf_by_track: ``_source_rank_features`` の事前計算結果。
            score_context: ``_prepare_score_context`` が返したコンテキスト。

        Returns:
            追加特徴量の辞書。base では空辞書。
        """
        del track_id, source_rank_maps, rrf_by_track, score_context
        return {}

    def _source_rank_features(
        self,
        candidates_by_source: dict[str, list[str]],
    ) -> dict[str, SourceRank]:
        """全候補の等重み RRF スコアを事前計算して返す。

        build_rows の内側ループで O(candidates × sources) の計算を避けるため、
        候補ごとに全ソースを集約した SourceRank を先にまとめて計算する。

        Args:
            candidates_by_source: ソース名 → track_id リストのマップ。

        Returns:
            track_id → SourceRank のマップ。
        """
        rank_by_track: dict[str, SourceRank] = {}
        source_order = {name: idx for idx, name in enumerate(self.source_names)}
        for source_name in self.source_names:
            for rank, track_id in enumerate(candidates_by_source.get(source_name, []), start=1):
                score_delta = 1.0 / float(self.rrf_k + rank)
                previous = rank_by_track.get(track_id)
                if previous is None:
                    rank_by_track[track_id] = SourceRank(
                        score=score_delta,
                        best_rank=rank,
                        source_count=1,
                        first_source_order=source_order.get(source_name, len(self.source_names)),
                    )
                    continue
                rank_by_track[track_id] = SourceRank(
                    score=previous.score + score_delta,
                    best_rank=min(previous.best_rank, rank),
                    source_count=previous.source_count + 1,
                    first_source_order=min(
                        previous.first_source_order,
                        source_order.get(source_name, len(self.source_names)),
                    ),
                )
        return rank_by_track

    @staticmethod
    def _generic_rank_features(candidate_rank: int) -> dict[str, float]:
        """候補の順位に基づくランク特徴量を返す。

        Args:
            candidate_rank: union 候補における 1-indexed の順位。

        Returns:
            f_candidate_rank と f_retrieval_rank_score を含む辞書。
        """
        return {
            "f_candidate_rank": float(candidate_rank),
            "f_retrieval_rank_score": 1.0 / math.log2(candidate_rank + 1.0),
        }

    def _source_features(
        self,
        track_id: str,
        source_rank_maps: dict[str, dict[str, int]],
        rrf_by_track: dict[str, SourceRank],
    ) -> dict[str, float | None]:
        """各候補ソースのヒット・順位・RRF スコア特徴量を返す。

        Args:
            track_id: 特徴量を計算する候補 track_id。
            source_rank_maps: ソース名 → {track_id: rank} のマップ。
            rrf_by_track: _source_rank_features で事前計算した SourceRank マップ。

        Returns:
            f_rrf_score, f_best_rank_any_source, f_source_hit_count と
            各ソースの hit / rank / rrf 特徴量を含む辞書。
        """
        rank_info = rrf_by_track.get(track_id)
        features: dict[str, float | None] = {
            "f_rrf_score": float(rank_info.score) if rank_info else 0.0,
            "f_best_rank_any_source": float(rank_info.best_rank) if rank_info else None,
            "f_source_hit_count": float(rank_info.source_count) if rank_info else 0.0,
        }
        for source_name in self.source_names:
            rank = source_rank_maps.get(source_name, {}).get(track_id)
            prefix = f"f_source_{source_name}"
            features[f"{prefix}_hit"] = 1.0 if rank is not None else 0.0
            features[f"{prefix}_rank"] = float(rank) if rank is not None else None
            features[f"{prefix}_rrf"] = 1.0 / float(self.rrf_k + rank) if rank is not None else 0.0
        return features

    def _lexical_metadata_features(
        self,
        *,
        metadata: dict[str, Any],
        full_query: str,
        current_query: str,
        full_tokens: set[str],
        current_tokens: set[str],
        recent_tokens: set[str],
    ) -> dict[str, float]:
        """クエリとトラックメタデータの語彙マッチ特徴量を返す。

        Args:
            metadata: track のメタデータ辞書。
            full_query: 全ターンを結合したクエリ文字列。
            current_query: 現在ターンのユーザ発話。
            full_tokens: full_query のトークン集合。
            current_tokens: current_query のトークン集合。
            recent_tokens: 直近ターンクエリのトークン集合。

        Returns:
            track_name / artist_name / album_name の exact match と
            tag overlap・popularity・release_year 等の特徴量辞書。
        """
        track_values = [normalize_text(value) for value in as_list(metadata.get("track_name"))]
        artist_values = [normalize_text(value) for value in as_list(metadata.get("artist_name"))]
        album_values = [normalize_text(value) for value in as_list(metadata.get("album_name"))]
        tag_values = [normalize_text(value) for value in as_list(metadata.get("tag_list"))]
        tag_tokens = set().union(*(set(tag.split()) for tag in tag_values if tag)) if tag_values else set()

        current_norm = normalize_text(current_query)
        full_norm = normalize_text(full_query)
        current_tag_overlap = len(tag_tokens & current_tokens)
        full_tag_overlap = len(tag_tokens & full_tokens)
        recent_tag_overlap = len(tag_tokens & recent_tokens)
        return {
            "f_track_name_exact_current": float(any(phrase_in_text(value, current_norm) for value in track_values)),
            "f_artist_name_exact_current": float(any(phrase_in_text(value, current_norm) for value in artist_values)),
            "f_album_name_exact_current": float(any(phrase_in_text(value, current_norm) for value in album_values)),
            "f_track_name_exact_full": float(any(phrase_in_text(value, full_norm) for value in track_values)),
            "f_artist_name_exact_full": float(any(phrase_in_text(value, full_norm) for value in artist_values)),
            "f_album_name_exact_full": float(any(phrase_in_text(value, full_norm) for value in album_values)),
            "f_tag_overlap_current": float(current_tag_overlap),
            "f_tag_overlap_full_context": float(full_tag_overlap),
            "f_tag_overlap_recent": float(recent_tag_overlap),
            "f_popularity": float(metadata.get("popularity") or 0.0),
            "f_release_year": parse_release_year(metadata),
            "f_metadata_field_count": float(sum(1 for value in metadata.values() if value not in (None, "", []))),
            "f_tag_count": float(len(tag_values)),
        }

    def _history_metadata_sets(self, hist_ids: list[str]) -> tuple[set[str], set[str]]:
        """履歴 track のアーティスト名とタグの正規化済み集合を返す。

        Args:
            hist_ids: 履歴 track_id のリスト。

        Returns:
            (正規化済みアーティスト名の集合, 正規化済みタグの集合) のタプル。
        """
        artists: set[str] = set()
        tags: set[str] = set()
        for track_id in hist_ids:
            metadata = self.item_db.metadata_dict.get(track_id, {})
            for artist in as_list(metadata.get("artist_name")):
                norm = normalize_text(artist)
                if norm:
                    artists.add(norm)
            for tag in as_list(metadata.get("tag_list")):
                norm = normalize_text(tag)
                if norm:
                    tags.add(norm)
        return artists, tags

    def _history_features(
        self,
        *,
        metadata: dict[str, Any],
        hist_artists: set[str],
        hist_tags: set[str],
        hist_ids: list[str],
        track_id: str,
        cf_history_mean: torch.Tensor | None,
        audio_history_mean: torch.Tensor | None,
        metadata_history_mean: torch.Tensor | None,
        user_vector: torch.Tensor | None,
    ) -> dict[str, float]:
        """候補 track と再生履歴の embedding 類似度・メタデータ一致特徴量を返す。

        Args:
            metadata: 候補 track のメタデータ辞書。
            hist_artists: 履歴のアーティスト名集合。
            hist_tags: 履歴のタグ集合。
            hist_ids: 履歴 track_id のリスト（最大コサイン計算に使用）。
            track_id: 特徴量を計算する候補 track_id。
            cf_history_mean: CF-BPR 履歴 track の平均ベクトル。
            audio_history_mean: audio 履歴 track の平均ベクトル。
            metadata_history_mean: metadata 履歴 track の平均ベクトル。
            user_vector: CF-BPR ユーザベクトル。

        Returns:
            同アーティスト数・同タグ数・各 embedding コサイン類似度を含む辞書。
        """
        candidate_artists = {normalize_text(value) for value in as_list(metadata.get("artist_name")) if normalize_text(value)}
        candidate_tags = {normalize_text(value) for value in as_list(metadata.get("tag_list")) if normalize_text(value)}
        cf_vector = self.track_embedding_db.get(track_id) if self.track_embedding_db is not None else None
        audio_vector = self.audio_embedding_db.get(track_id) if self.audio_embedding_db is not None else None
        metadata_vector = self.metadata_embedding_db.get(track_id) if self.metadata_embedding_db is not None else None
        return {
            "f_same_artist_history_count": float(len(candidate_artists & hist_artists)),
            "f_same_tag_history_count": float(len(candidate_tags & hist_tags)),
            "f_history_cf_bpr_mean_cosine": safe_cosine(cf_history_mean, cf_vector),
            "f_history_cf_bpr_max_cosine": max_history_cosine(self.track_embedding_db, hist_ids, track_id),
            "f_user_cf_bpr_cosine": safe_cosine(user_vector, cf_vector),
            "f_history_audio_mean_cosine": safe_cosine(audio_history_mean, audio_vector),
            "f_history_metadata_mean_cosine": safe_cosine(metadata_history_mean, metadata_vector),
        }


def feature_columns(rows_or_columns: Any) -> list[str]:
    """DataFrame またはカラムリストから f_ プレフィックスの特徴量列名を抽出する。

    Args:
        rows_or_columns: DataFrame / 行リスト / カラム名リストのいずれか。

    Returns:
        f_ で始まるカラム名のソート済みリスト。
    """
    if hasattr(rows_or_columns, "columns"):
        columns = list(rows_or_columns.columns)
    elif rows_or_columns and isinstance(rows_or_columns, list):
        columns = list(rows_or_columns[0].keys())
    else:
        columns = list(rows_or_columns)
    return sorted(column for column in columns if str(column).startswith("f_"))
