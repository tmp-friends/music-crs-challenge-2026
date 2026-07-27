"""exp015 candidate rule ablation 用の source-aware 推論 pipeline。"""

from __future__ import annotations

from typing import Any

import torch

from mcrs.crs_baseline import CRS_BASELINE
from mcrs.db_item.track_embeddings import TrackEmbeddingDB
from mcrs.experiments.exp015_candidate_rules_ablation.candidate_sources import (
    Exp015CandidateBuilder,
    PROFILE_SOURCES,
    SegmentPopularityIndex,
    source_profile_to_sources,
)
from mcrs.experiments.exp015_candidate_rules_ablation.lightgbm_ltr import LightGBMLTRReranker
from mcrs.retrieval_modules.bm25 import BM25_MODEL
from mcrs.retrieval_modules.metadata_exact import DEFAULT_METADATA_FIELDS, MetadataExactRetriever


class Exp015LightGBMCRS(CRS_BASELINE):
    """exp015 candidate builder と exp015 LightGBM reranker を接続する CRS。"""

    def __init__(
        self,
        *args: Any,
        metadata_bm25_corpus_types: list[str] | None = None,
        metadata_exact_fields: list[str] | None = None,
        recent_turn_count: int = 2,
        track_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_embedding_name: str = "cf-bpr",
        audio_embedding_name: str = "audio-laion_clap",
        user_embedding_name: str = "cf-bpr",
        user_embedding_split_types: list[str] | None = None,
        enabled_sources: list[str] | None = None,
        source_profile: str = "A",
        union_mode: str | None = None,
        reranker_type: str | None = None,
        reranker_model_name: str | None = None,
        reranker_weights: dict[str, float] | None = None,
        conversation_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
        segment_popularity_split: str = "train",
        **kwargs: Any,
    ) -> None:
        """base CRS を初期化し、exp015 専用候補生成と reranker を接続する。

        Args:
            *args: CRS_BASELINE へ渡す位置引数。
            metadata_bm25_corpus_types: copied base 用 metadata BM25 fields。
            metadata_exact_fields: exact retrieval 用 fields。
            recent_turn_count: recent query view に含める直近 turn 数。
            track_embedding_dataset_name: track embedding dataset 名。
            user_embedding_dataset_name: user embedding dataset 名。
            track_embedding_name: CF-BPR track embedding 名。
            audio_embedding_name: audio embedding 名。
            user_embedding_name: user embedding 名。
            user_embedding_split_types: user embedding split。
            enabled_sources: source profile を上書きする source list。
            source_profile: Run A-G の profile ID。
            union_mode: ``rrf`` または ``quota``。
            reranker_type: ``lightgbm_ltr`` の場合のみ exp015 reranker を作る。
            reranker_model_name: LightGBM model path。
            reranker_weights: RRF weight 互換設定。
            conversation_dataset_name: segment popularity 統計を作る conversation dataset。
            segment_popularity_split: popularity 統計に使う split。
            **kwargs: CRS_BASELINE へ渡す追加引数。
        """
        self.source_profile = str(source_profile or "A").upper()
        source_names = tuple(enabled_sources) if enabled_sources else source_profile_to_sources(self.source_profile)
        self.exp015_source_names = source_names
        self.exp015_reranker_type = reranker_type
        self.exp015_reranker_model_name = reranker_model_name
        self.exp015_reranker_weights = reranker_weights

        # CRS_BASELINE の共通 reranker loader は exp014 source 名固定なので使わない。
        super().__init__(
            *args,
            track_embedding_dataset_name=track_embedding_dataset_name,
            user_embedding_dataset_name=user_embedding_dataset_name,
            track_embedding_name=track_embedding_name,
            audio_embedding_name=audio_embedding_name,
            user_embedding_name=user_embedding_name,
            user_embedding_split_types=user_embedding_split_types,
            reranker_type=None,
            reranker_weights=None,
            reranker_model_name=reranker_model_name or "",
            metadata_exact_fields=metadata_exact_fields,
            **kwargs,
        )

        metadata_fields = metadata_exact_fields or list(DEFAULT_METADATA_FIELDS)
        metadata_bm25_fields = metadata_bm25_corpus_types or list(DEFAULT_METADATA_FIELDS)
        self.metadata_bm25 = BM25_MODEL(self.item_db_name, self.track_split_types, metadata_bm25_fields, self.cache_dir)
        self.metadata_text_bm25 = BM25_MODEL(
            self.item_db_name,
            self.track_split_types,
            ["track_name", "artist_name", "album_name", "release_date"],
            self.cache_dir,
        )
        self.metadata_tag_bm25 = BM25_MODEL(
            self.item_db_name,
            self.track_split_types,
            ["tag_list"],
            self.cache_dir,
        )
        self.exact_retriever = MetadataExactRetriever(
            dataset_name=self.item_db_name,
            split_types=self.track_split_types,
            metadata_fields=metadata_fields,
            cache_dir=self.cache_dir,
        )

        needs_history = bool(set(source_names) & {
            "history_cf_mean",
            "history_audio_mean",
            "history_recent_neighbors",
        })
        track_embedding_db = (
            TrackEmbeddingDB(
                dataset_name=track_embedding_dataset_name,
                split_types=self.track_split_types,
                embedding_name=track_embedding_name,
            )
            if needs_history else None
        )
        audio_embedding_db = (
            TrackEmbeddingDB(
                dataset_name=track_embedding_dataset_name,
                split_types=self.track_split_types,
                embedding_name=audio_embedding_name,
            )
            if needs_history else None
        )
        segment_index = (
            SegmentPopularityIndex(
                conversation_dataset_name=conversation_dataset_name,
                user_dataset_name=self.user_db_name,
                item_db=self.item_db,
                user_split_types=self.user_split_types,
                split=segment_popularity_split,
            )
            if bool(set(source_names) & {
                "popular_by_artist_current",
                "popular_by_tag_current",
                "popular_by_user_age_country",
                "popular_by_history_artist",
            })
            else None
        )

        self.wide_builder = Exp015CandidateBuilder(
            item_db=self.item_db,
            baseline_bm25=self.retrieval,
            metadata_bm25=self.metadata_bm25,
            metadata_text_bm25=self.metadata_text_bm25,
            metadata_tag_bm25=self.metadata_tag_bm25,
            exact_retriever=self.exact_retriever,
            track_embedding_db=track_embedding_db,
            audio_embedding_db=audio_embedding_db,
            recent_turn_count=recent_turn_count,
            enabled_sources=list(source_names),
            source_profile=self.source_profile,
            union_mode=union_mode or ("quota" if self.source_profile == "G" else "rrf"),
            source_weights=reranker_weights,
            segment_popularity_index=segment_index,
        )

        if reranker_type == "lightgbm_ltr":
            if not reranker_model_name:
                raise ValueError("exp015 lightgbm_ltr requires reranker_model_name.")
            self.reranker = LightGBMLTRReranker(
                model_path=reranker_model_name,
                item_db_name=self.item_db_name,
                track_embedding_dataset_name=track_embedding_dataset_name,
                user_embedding_dataset_name=user_embedding_dataset_name,
                track_split_types=self.track_split_types,
                corpus_types=self.corpus_types,
                user_embedding_split_types=user_embedding_split_types,
                track_embedding_name=track_embedding_name,
                audio_embedding_name=audio_embedding_name,
                user_embedding_name=user_embedding_name,
                source_weights=reranker_weights,
                cache_dir=self.cache_dir,
            )

    def _rerank_wide_items(
        self,
        *,
        candidates_by_source: dict[str, list[str]],
        candidate_items: list[str],
        query_views: dict[str, str],
        user_id: str | None,
        session_memory: list[dict[str, Any]],
        scores_by_source: dict[str, dict[str, float]] | None = None,
    ) -> list[str]:
        if self.reranker is not None and hasattr(self.reranker, "rerank_with_sources"):
            return self.reranker.rerank_with_sources(
                candidates_by_source=candidates_by_source,
                union_candidates=candidate_items,
                query_views=query_views,
                user_id=user_id,
                session_memory=session_memory,
                topk=self.rerank_topk,
                scores_by_source=scores_by_source,
            )
        return candidate_items[: self.rerank_topk]

    def batch_chat(self, batch_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """複数 turn を batch で候補生成・rerank・応答生成する。"""
        sys_prompts = []
        raw_session_memories = []
        normalized_session_memories = []
        user_ids = []
        user_queries = []
        for data in batch_data:
            user_query = str(data["user_query"])
            raw_session_memory = list(data["session_memory"])
            normalized_session_memory = self._normalize_session_memory(raw_session_memory)
            normalized_session_memory.append({"role": "user", "content": user_query})
            sys_prompts.append(self._get_system_prompt(data.get("user_id")))
            raw_session_memories.append(raw_session_memory)
            normalized_session_memories.append(normalized_session_memory)
            user_ids.append(data.get("user_id"))
            user_queries.append(user_query)

        # reranker の feature_builder が source score を消費する場合のみ score 付き
        # retrieval を使う。消費しない既存 reranker では従来経路（後方互換）。
        capture_scores = (
            self.reranker is not None
            and getattr(
                getattr(self.reranker, "feature_builder", None),
                "consumes_source_scores",
                False,
            )
            and hasattr(self.wide_builder, "batch_candidates_with_scores")
        )
        if capture_scores:
            (
                batch_candidates_by_source,
                query_views_by_task,
                _,
                batch_scores_by_task,
            ) = self.wide_builder.batch_candidates_with_scores(
                histories=raw_session_memories,
                user_queries=user_queries,
                user_ids=user_ids,
            )
        else:
            batch_candidates_by_source, query_views_by_task, _ = self.wide_builder.batch_candidates_by_source(
                histories=raw_session_memories,
                user_queries=user_queries,
                user_ids=user_ids,
            )
            batch_scores_by_task = None
        batch_candidate_items = [
            self.wide_builder.union(candidates, topk=self.retrieval_topk)
            for candidates in batch_candidates_by_source
        ]
        batch_retrieval_items = [
            self._rerank_wide_items(
                candidates_by_source=batch_candidates_by_source[index],
                candidate_items=batch_candidate_items[index],
                query_views=query_views_by_task[index],
                user_id=user_ids[index],
                session_memory=raw_session_memories[index],
                scores_by_source=(
                    batch_scores_by_task[index] if batch_scores_by_task is not None else None
                ),
            )
            for index in range(len(batch_data))
        ]
        recommend_items = [self.item_db.id_to_metadata(items[0]) for items in batch_retrieval_items]
        if hasattr(self.lm, "batch_response_generation"):
            responses = self.lm.batch_response_generation(
                sys_prompts,
                normalized_session_memories,
                recommend_items,
                max_new_tokens=self.lm_max_new_tokens,
            )
        else:
            responses = [
                self.lm.response_generation(
                    sys_prompts[index],
                    normalized_session_memories[index],
                    recommend_items[index],
                    max_new_tokens=self.lm_max_new_tokens,
                )
                for index in range(len(batch_data))
            ]
        return [
            {
                "user_id": data.get("user_id"),
                "user_query": data["user_query"],
                "retrieval_items": batch_retrieval_items[index],
                "recommend_item": recommend_items[index],
                "response": responses[index],
            }
            for index, data in enumerate(batch_data)
        ]


def load_crs_baseline(
    lm_type: str = "meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type: str = "bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types: list[str] = ["all_tracks"],
    user_split_types: list[str] = ["all_users"],
    corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
    cache_dir: str = "./cache",
    device: str = "cuda",
    attn_implementation: str = "eager",
    dtype: Any = torch.bfloat16,
    reranker_type: str | None = None,
    retrieval_topk: int = 500,
    rerank_topk: int = 20,
    track_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    user_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    track_embedding_name: str = "cf-bpr",
    audio_embedding_name: str = "audio-laion_clap",
    user_embedding_name: str = "cf-bpr",
    user_embedding_split_types: list[str] | None = None,
    reranker_weights: dict[str, float] | None = None,
    reranker_model_name: str | None = None,
    reranker_device: str | None = None,
    reranker_attn_implementation: str | None = None,
    reranker_dtype: Any = None,
    reranker_batch_size: int = 8,
    reranker_max_length: int = 8192,
    reranker_instruction: str | None = None,
    lm_max_new_tokens: int = 1024,
    metadata_exact_fields: list[str] | None = None,
    metadata_bm25_corpus_types: list[str] | None = None,
    enabled_sources: list[str] | None = None,
    source_profile: str = "A",
    union_mode: str | None = None,
):
    """exp015 専用 CRS pipeline を初期化する。"""
    del reranker_device, reranker_attn_implementation, reranker_dtype, reranker_batch_size
    del reranker_max_length, reranker_instruction
    if retrieval_type != "bm25":
        raise ValueError(f"exp015 expects retrieval_type='bm25', got {retrieval_type!r}")
    if list(track_split_types) != ["all_tracks"]:
        raise ValueError("exp015 must use track_split_types: ['all_tracks']")
    if enabled_sources is not None and source_profile == "A":
        normalized_sources = tuple(enabled_sources)
        for profile_id, profile_sources in PROFILE_SOURCES.items():
            if normalized_sources == tuple(profile_sources):
                source_profile = profile_id
                break
    return Exp015LightGBMCRS(
        lm_type=lm_type,
        retrieval_type=retrieval_type,
        item_db_name=item_db_name,
        user_db_name=user_db_name,
        track_split_types=track_split_types,
        user_split_types=user_split_types,
        corpus_types=corpus_types,
        cache_dir=cache_dir,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
        reranker_type=reranker_type,
        retrieval_topk=retrieval_topk,
        rerank_topk=rerank_topk,
        track_embedding_dataset_name=track_embedding_dataset_name,
        user_embedding_dataset_name=user_embedding_dataset_name,
        track_embedding_name=track_embedding_name,
        audio_embedding_name=audio_embedding_name,
        user_embedding_name=user_embedding_name,
        user_embedding_split_types=user_embedding_split_types,
        reranker_weights=reranker_weights,
        reranker_model_name=reranker_model_name,
        lm_max_new_tokens=lm_max_new_tokens,
        metadata_bm25_corpus_types=metadata_bm25_corpus_types or list(DEFAULT_METADATA_FIELDS),
        metadata_exact_fields=metadata_exact_fields or list(DEFAULT_METADATA_FIELDS),
        enabled_sources=enabled_sources,
        source_profile=source_profile,
        union_mode=union_mode,
    )
