"""wide candidate source builder（共有ライブラリ）。

exp012 の候補生成をベースに exp014 で実装したものを、experiment フォルダから
``mcrs/candidates/`` へ切り出した共有モジュール。9 種類の候補ソースを
reciprocal rank fusion (RRF) で統合する。旧
``mcrs.experiments.exp014_lightgbm_ltr_reranker.candidate_sources`` は re-export shim 経由で
このモジュールへ解決される。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from mcrs.candidates.query_views import build_query_views, dedupe


DEFAULT_SOURCE_CAPS = {
    "bm25_recent_turns": 250,
    "bm25_full_context": 250,
    "metadata_bm25_full_context": 400,
    "metadata_bm25_current_user_turn": 200,
    "exact_current_union": 150,
    "dense_text_recent": 200,
    "user_cf_bpr": 200,
    "history_cf_bpr_item2item": 150,
    "popularity_fallback": 50,
}

DEFAULT_RAW_TOPKS = {
    "bm25_recent_turns": 1000,
    "bm25_full_context": 1000,
    "metadata_bm25_full_context": 1500,
    "metadata_bm25_current_user_turn": 1000,
    "exact_current_union": 500,
    "dense_text_recent": 1000,
    "user_cf_bpr": 2000,
    "history_cf_bpr_item2item": 1000,
    "popularity_fallback": 200,
}

TEXT_SOURCE_NAMES = (
    "bm25_recent_turns",
    "bm25_full_context",
    "metadata_bm25_full_context",
    "metadata_bm25_current_user_turn",
    "exact_current_union",
)

EMBED_SOURCE_NAMES = (
    "dense_text_recent",
    "user_cf_bpr",
    "history_cf_bpr_item2item",
)

FALLBACK_SOURCE_NAMES = (
    "popularity_fallback",
)

SOURCE_NAMES = (*TEXT_SOURCE_NAMES, *EMBED_SOURCE_NAMES, *FALLBACK_SOURCE_NAMES)
UNION_SOURCE_NAME = "wide_candidate_union_rrf"


@dataclass(frozen=True)
class SourceRank:
    """候補 track の source 横断 ranking 情報。"""

    score: float
    best_rank: int
    source_count: int
    first_source_order: int


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


class WideCandidateBuilder:
    """9 種類の候補ソースから retrieval candidates を構築する。"""

    def __init__(
        self,
        *,
        item_db: Any,
        baseline_bm25: Any,
        metadata_bm25: Any,
        exact_retriever: Any,
        bert_retriever: Any | None = None,
        track_embedding_db: Any | None = None,
        user_embedding_db: Any | None = None,
        recent_turn_count: int = 2,
        source_weights: dict[str, float] | None = None,
        source_caps: dict[str, int] | None = None,
        raw_topks: dict[str, int] | None = None,
        rrf_k: int = 60,
        enabled_sources: list[str] | None = None,
        parallel_bm25_sources: bool = False,
        bm25_source_workers: int = 4,
    ) -> None:
        """候補生成に使う retriever / embedding DB と RRF 設定を保持する。

        Args:
            item_db: track metadata DB。
            baseline_bm25: 通常 corpus 用 BM25 retriever。
            metadata_bm25: metadata field 拡張 corpus 用 BM25 retriever。
            exact_retriever: metadata exact match retriever。
            bert_retriever: dense_text_recent 用 BERT retriever。
            track_embedding_db: track embedding DB。
            user_embedding_db: user embedding DB。
            recent_turn_count: recent query view に含める直近 turn 数。
            source_weights: 互換用の source 別重み。現在の RRF 計算では使用しない。
            source_caps: source 別候補数 cap の上書き値。
            raw_topks: retrieval 直後の source 別 topk の上書き値。
            rrf_k: RRF の平滑化定数。
            enabled_sources: 使用する source 名リスト。None の場合は全 source。
            parallel_bm25_sources: BM25 source を外側で並列実行するか。
            bm25_source_workers: BM25 source 並列の worker 数。
        """
        self.item_db = item_db
        self.baseline_bm25 = baseline_bm25
        self.metadata_bm25 = metadata_bm25
        self.exact_retriever = exact_retriever
        self.bert_retriever = bert_retriever
        self.track_embedding_db = track_embedding_db
        self.user_embedding_db = user_embedding_db
        self.recent_turn_count = int(recent_turn_count)
        self.enabled_sources: set[str] = set(enabled_sources) if enabled_sources else set(SOURCE_NAMES)
        self.parallel_bm25_sources = bool(parallel_bm25_sources)
        self.bm25_source_workers = int(bm25_source_workers)

        self.source_weights = {k: float(v) for k, v in (source_weights or {}).items()}
        self.source_caps = dict(DEFAULT_SOURCE_CAPS)
        if source_caps:
            self.source_caps.update({k: int(v) for k, v in source_caps.items()})
        self.raw_topks = dict(DEFAULT_RAW_TOPKS)
        if raw_topks:
            self.raw_topks.update({k: int(v) for k, v in raw_topks.items()})
        self.rrf_k = int(rrf_k)

        self._build_track_matrix()
        self._popularity_track_ids = list(exact_retriever.fallback_track_ids)

    def _build_track_matrix(self) -> None:
        if self.track_embedding_db is None:
            self._track_matrix: torch.Tensor | None = None
            self._track_id_list: list[str] = []
            return
        track_ids: list[str] = []
        vectors: list[torch.Tensor] = []
        for track_id, vector in self.track_embedding_db.embeddings.items():
            if self.track_embedding_db.is_valid(vector):
                track_ids.append(track_id)
                vectors.append(vector)
        if not vectors:
            self._track_matrix = None
            self._track_id_list = []
            return
        self._track_matrix = F.normalize(torch.stack(vectors, dim=0), p=2, dim=1)
        self._track_id_list = track_ids

    def _cosine_topk(self, query_vector: torch.Tensor, topk: int) -> list[str]:
        if self._track_matrix is None or not self._track_id_list:
            return []
        normalized = F.normalize(query_vector.unsqueeze(0), p=2, dim=1).squeeze(0)
        scores = torch.matmul(self._track_matrix, normalized)
        k = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=k).indices.tolist()
        return [self._track_id_list[i] for i in top_indices]

    def query_views(
        self,
        *,
        history: list[dict[str, Any]],
        user_query: str,
    ) -> dict[str, str]:
        """履歴と最新発話から候補生成用の query view を作る。

        Args:
            history: 現在 turn より前の会話履歴。
            user_query: 現在 turn のユーザ発話。

        Returns:
            BM25 source ごとに使う query view 辞書。
        """
        return build_query_views(
            history=history,
            user_query=user_query,
            target_turn_number=_infer_target_turn_number(history),
            item_db=self.item_db,
            recent_turn_count=self.recent_turn_count,
        )

    def _source_bm25_recent_turns(
        self, query_views: dict[str, str],
    ) -> tuple[list[str], bool]:
        raw_topk = self.raw_topks["bm25_recent_turns"]
        return dedupe(
            self.baseline_bm25.text_to_item_retrieval(query_views["bm25_recent_turns"], topk=raw_topk)
        ), True

    def _source_bm25_full_context(
        self, query_views: dict[str, str],
    ) -> tuple[list[str], bool]:
        raw_topk = self.raw_topks["bm25_full_context"]
        return dedupe(
            self.baseline_bm25.text_to_item_retrieval(query_views["bm25_full_context"], topk=raw_topk)
        ), True

    def _source_metadata_bm25_full_context(
        self, query_views: dict[str, str],
    ) -> tuple[list[str], bool]:
        raw_topk = self.raw_topks["metadata_bm25_full_context"]
        return dedupe(
            self.metadata_bm25.text_to_item_retrieval(query_views["bm25_full_context"], topk=raw_topk)
        ), True

    def _source_metadata_bm25_current_user_turn(
        self, query_views: dict[str, str],
    ) -> tuple[list[str], bool]:
        raw_topk = self.raw_topks["metadata_bm25_current_user_turn"]
        return dedupe(
            self.metadata_bm25.text_to_item_retrieval(query_views["bm25_current_user_turn"], topk=raw_topk)
        ), True

    def _source_exact_current_union(
        self, query_views: dict[str, str],
    ) -> tuple[list[str], bool]:
        raw_topk = self.raw_topks["exact_current_union"]
        return dedupe(
            self.exact_retriever.ranked_exact_matches(query_views["bm25_current_user_turn"], topk=raw_topk)
        ), True

    def _source_dense_text_recent(
        self, query_views: dict[str, str],
    ) -> tuple[list[str], bool]:
        if self.bert_retriever is None:
            return [], False
        raw_topk = self.raw_topks["dense_text_recent"]
        return dedupe(
            self.bert_retriever.text_to_item_retrieval(query_views["bm25_recent_turns"], topk=raw_topk)
        ), True

    def _source_user_cf_bpr(
        self, user_id: str | None,
    ) -> tuple[list[str], bool]:
        if self.user_embedding_db is None or self._track_matrix is None:
            return [], False
        user_vector = self.user_embedding_db.get(user_id)
        if not self.user_embedding_db.is_valid(user_vector):
            return [], False
        raw_topk = self.raw_topks["user_cf_bpr"]
        return self._cosine_topk(user_vector, raw_topk), True

    def _source_history_cf_bpr_item2item(
        self, history: list[dict[str, Any]],
    ) -> tuple[list[str], bool]:
        if self.track_embedding_db is None or self._track_matrix is None:
            return [], False
        history_ids = _history_track_ids(history)
        if not history_ids:
            return [], False
        mean_vector = self.track_embedding_db.mean_vector(history_ids)
        if not self.track_embedding_db.is_valid(mean_vector):
            return [], False
        raw_topk = self.raw_topks["history_cf_bpr_item2item"]
        return self._cosine_topk(mean_vector, raw_topk), True

    def _source_popularity_fallback(self) -> tuple[list[str], bool]:
        raw_topk = self.raw_topks["popularity_fallback"]
        return self._popularity_track_ids[:raw_topk], True

    def _dispatch_source(
        self,
        source_name: str,
        query_views: dict[str, str],
        user_id: str | None,
        history: list[dict[str, Any]],
    ) -> tuple[list[str], bool]:
        if source_name not in self.enabled_sources:
            return [], False
        dispatch = {
            "bm25_recent_turns": lambda: self._source_bm25_recent_turns(query_views),
            "bm25_full_context": lambda: self._source_bm25_full_context(query_views),
            "metadata_bm25_full_context": lambda: self._source_metadata_bm25_full_context(query_views),
            "metadata_bm25_current_user_turn": lambda: self._source_metadata_bm25_current_user_turn(query_views),
            "exact_current_union": lambda: self._source_exact_current_union(query_views),
            "dense_text_recent": lambda: self._source_dense_text_recent(query_views),
            "user_cf_bpr": lambda: self._source_user_cf_bpr(user_id),
            "history_cf_bpr_item2item": lambda: self._source_history_cf_bpr_item2item(history),
            "popularity_fallback": lambda: self._source_popularity_fallback(),
        }
        return dispatch[source_name]()

    def candidates_by_source(
        self,
        *,
        history: list[dict[str, Any]],
        user_query: str,
        user_id: str | None,
    ) -> tuple[dict[str, list[str]], dict[str, str], dict[str, bool]]:
        """1 task 分の source 別候補を生成する。

        Args:
            history: 現在 turn より前の会話履歴。
            user_query: 現在 turn のユーザ発話。
            user_id: user_cf source で使う user ID。

        Returns:
            (source 別候補, query view, source active flag) のタプル。
        """
        query_views = self.query_views(history=history, user_query=user_query)

        candidates: dict[str, list[str]] = {}
        active: dict[str, bool] = {}
        for source_name in SOURCE_NAMES:
            items, is_active = self._dispatch_source(source_name, query_views, user_id, history)
            candidates[source_name] = self._cap(source_name, items)
            active[source_name] = is_active
        return candidates, query_views, active

    def _needs_recent_bm25(self) -> bool:
        return "bm25_recent_turns" in self.enabled_sources

    def _needs_full_bm25(self) -> bool:
        return "bm25_full_context" in self.enabled_sources

    def _needs_meta_full(self) -> bool:
        return "metadata_bm25_full_context" in self.enabled_sources

    def _needs_meta_current(self) -> bool:
        return "metadata_bm25_current_user_turn" in self.enabled_sources

    def _needs_dense(self) -> bool:
        return "dense_text_recent" in self.enabled_sources and self.bert_retriever is not None

    def batch_candidates_by_source(
        self,
        *,
        histories: list[list[dict[str, Any]]],
        user_queries: list[str],
        user_ids: list[str | None],
    ) -> tuple[list[dict[str, list[str]]], list[dict[str, str]], list[dict[str, bool]]]:
        """複数 task の source 別候補をまとめて生成する。

        BM25 retriever の batch API を使い、source が有効なものだけ計算する。

        Args:
            histories: task ごとの会話履歴。
            user_queries: task ごとの現在 turn ユーザ発話。
            user_ids: task ごとの user ID。

        Returns:
            task ごとの (source 別候補, query view, source active flag)。
        """
        n = len(histories)
        query_views_list = [
            self.query_views(history=history, user_query=user_query)
            for history, user_query in zip(histories, user_queries)
        ]

        recent_queries = [v["bm25_recent_turns"] for v in query_views_list]
        full_queries = [v["bm25_full_context"] for v in query_views_list]
        current_queries = [v["bm25_current_user_turn"] for v in query_views_list]

        empty_batch: list[list[str]] = [[] for _ in range(n)]

        bm25_jobs: dict[str, Any] = {}
        if self._needs_recent_bm25():
            bm25_jobs["bm25_recent_turns"] = (
                lambda: self.baseline_bm25.batch_text_to_item_retrieval(
                    recent_queries,
                    topk=self.raw_topks["bm25_recent_turns"],
                )
            )
        if self._needs_full_bm25():
            bm25_jobs["bm25_full_context"] = (
                lambda: self.baseline_bm25.batch_text_to_item_retrieval(
                    full_queries,
                    topk=self.raw_topks["bm25_full_context"],
                )
            )
        if self._needs_meta_full():
            bm25_jobs["metadata_bm25_full_context"] = (
                lambda: self.metadata_bm25.batch_text_to_item_retrieval(
                    full_queries,
                    topk=self.raw_topks["metadata_bm25_full_context"],
                )
            )
        if self._needs_meta_current():
            bm25_jobs["metadata_bm25_current_user_turn"] = (
                lambda: self.metadata_bm25.batch_text_to_item_retrieval(
                    current_queries,
                    topk=self.raw_topks["metadata_bm25_current_user_turn"],
                )
            )

        if self.parallel_bm25_sources and len(bm25_jobs) > 1:
            worker_count = max(1, min(self.bm25_source_workers, len(bm25_jobs)))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    source_name: executor.submit(job)
                    for source_name, job in bm25_jobs.items()
                }
                bm25_results = {
                    source_name: future.result()
                    for source_name, future in futures.items()
                }
        else:
            bm25_results = {
                source_name: job()
                for source_name, job in bm25_jobs.items()
            }

        bm25_recent_batch = bm25_results.get("bm25_recent_turns", empty_batch)
        bm25_full_batch = bm25_results.get("bm25_full_context", empty_batch)
        meta_full_batch = bm25_results.get("metadata_bm25_full_context", empty_batch)
        meta_current_batch = bm25_results.get("metadata_bm25_current_user_turn", empty_batch)
        dense_batch = (
            self.bert_retriever.batch_text_to_item_retrieval(recent_queries, topk=self.raw_topks["dense_text_recent"])
            if self._needs_dense() else empty_batch
        )

        all_candidates: list[dict[str, list[str]]] = []
        all_query_views: list[dict[str, str]] = query_views_list
        all_active: list[dict[str, bool]] = []

        exact_enabled = "exact_current_union" in self.enabled_sources
        exact_topk = self.raw_topks["exact_current_union"]

        for i in range(n):
            raw: dict[str, tuple[list[str], bool]] = {}

            raw["bm25_recent_turns"] = (dedupe(bm25_recent_batch[i]), True) if self._needs_recent_bm25() else ([], False)
            raw["bm25_full_context"] = (dedupe(bm25_full_batch[i]), True) if self._needs_full_bm25() else ([], False)
            raw["metadata_bm25_full_context"] = (dedupe(meta_full_batch[i]), True) if self._needs_meta_full() else ([], False)
            raw["metadata_bm25_current_user_turn"] = (dedupe(meta_current_batch[i]), True) if self._needs_meta_current() else ([], False)
            raw["exact_current_union"] = (
                (dedupe(self.exact_retriever.ranked_exact_matches(current_queries[i], topk=exact_topk)), True)
                if exact_enabled else ([], False)
            )
            raw["dense_text_recent"] = (dedupe(dense_batch[i]), True) if self._needs_dense() else ([], False)
            raw["user_cf_bpr"] = self._source_user_cf_bpr(user_ids[i]) if "user_cf_bpr" in self.enabled_sources else ([], False)
            raw["history_cf_bpr_item2item"] = self._source_history_cf_bpr_item2item(histories[i]) if "history_cf_bpr_item2item" in self.enabled_sources else ([], False)
            raw["popularity_fallback"] = self._source_popularity_fallback() if "popularity_fallback" in self.enabled_sources else ([], False)

            candidates: dict[str, list[str]] = {}
            active: dict[str, bool] = {}
            for source_name in SOURCE_NAMES:
                items, is_active = raw[source_name]
                candidates[source_name] = self._cap(source_name, items)
                active[source_name] = is_active

            all_candidates.append(candidates)
            all_active.append(active)

        return all_candidates, all_query_views, all_active

    def union(self, candidates_by_source: dict[str, list[str]], topk: int) -> list[str]:
        """source 別候補を等重み RRF で統合し、上位 track_id を返す。

        Args:
            candidates_by_source: source 名から track_id リストへの mapping。
            topk: 統合後に返す候補数。

        Returns:
            RRF score の降順で並んだ track_id リスト。
        """
        rank_by_track: dict[str, SourceRank] = {}
        source_order = {name: idx for idx, name in enumerate(SOURCE_NAMES)}

        for source_name in SOURCE_NAMES:
            for rank, track_id in enumerate(candidates_by_source.get(source_name, []), start=1):
                score_delta = 1.0 / float(self.rrf_k + rank)
                previous = rank_by_track.get(track_id)
                if previous is None:
                    rank_by_track[track_id] = SourceRank(
                        score=score_delta,
                        best_rank=rank,
                        source_count=1,
                        first_source_order=source_order[source_name],
                    )
                    continue
                rank_by_track[track_id] = SourceRank(
                    score=previous.score + score_delta,
                    best_rank=min(previous.best_rank, rank),
                    source_count=previous.source_count + 1,
                    first_source_order=min(previous.first_source_order, source_order[source_name]),
                )

        ranked = sorted(
            rank_by_track.items(),
            key=lambda item: (
                -item[1].score,
                -item[1].source_count,
                item[1].best_rank,
                item[1].first_source_order,
                item[0],
            ),
        )
        return [track_id for track_id, _ in ranked[:topk]]

    def _cap(self, source_name: str, candidates: list[str]) -> list[str]:
        cap = self.source_caps.get(source_name)
        if cap is None:
            return candidates
        return candidates[:cap]
