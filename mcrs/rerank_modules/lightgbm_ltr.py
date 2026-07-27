"""LightGBM LambdaRank 学習済みモデルで retrieval 候補を rerank する。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mcrs.db_item import MusicCatalogDB
from mcrs.db_item.track_embeddings import TrackEmbeddingDB
from mcrs.db_user.user_embeddings import UserEmbeddingDB


def require_lightgbm():
    """lightgbm をインポートして返す。未インストール時は ImportError を送出する。"""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm_ltr reranker requires lightgbm. Install it with `uv pip install lightgbm`."
        ) from exc
    return lgb


class LightGBMLTRReranker:
    """LightGBM LambdaRank で候補リストをリランクするモジュール。

    offline 学習済みモデルと LGBMFeatureBuilder を組み合わせ、
    各候補の特徴量行を構築してスコアリングすることでリランクを行う。
    """

    def __init__(
        self,
        *,
        model_path: str,
        item_db_name: str,
        track_embedding_dataset_name: str,
        user_embedding_dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        user_embedding_split_types: list[str] | None = None,
        track_embedding_name: str = "cf-bpr",
        audio_embedding_name: str = "audio-laion_clap",
        user_embedding_name: str = "cf-bpr",
        metadata_embedding_name: str | None = None,
        source_weights: dict[str, float] | None = None,
        cache_dir: str = "./cache",
    ) -> None:
        """モデルと各種 DB を初期化する。

        Args:
            model_path: 保存済み LightGBM モデルのパス。
            item_db_name: track metadata HF dataset 名。
            track_embedding_dataset_name: track embedding HF dataset 名。
            user_embedding_dataset_name: user embedding HF dataset 名。
            track_split_types: track の split フィルタ（通常 ["all_tracks"]）。
            corpus_types: BM25 corpus テキストの種類リスト。
            user_embedding_split_types: user embedding の split フィルタ（None で全 split）。
            track_embedding_name: CF-BPR track embedding 名。
            audio_embedding_name: audio embedding 名（LAION CLAP）。
            user_embedding_name: user embedding 名。
            metadata_embedding_name: metadata embedding 名（None でスキップ）。
            source_weights: 候補ソースの RRF 重み辞書。
            cache_dir: 他の reranker と IF を揃えるために受け取るが、このクラスでは使用しない。
        """
        # 他の reranker インターフェースに合わせて受け取るが、このクラスでは使用しない
        del cache_dir
        lgb = require_lightgbm()
        self.booster = lgb.Booster(model_file=model_path)
        self.feature_names = list(self.booster.feature_name())
        self.item_db = MusicCatalogDB(item_db_name, track_split_types, corpus_types)
        self.track_embeddings = TrackEmbeddingDB(
            dataset_name=track_embedding_dataset_name,
            split_types=track_split_types,
            embedding_name=track_embedding_name,
        )
        self.user_embeddings = UserEmbeddingDB(
            dataset_name=user_embedding_dataset_name,
            split_types=user_embedding_split_types,
            embedding_name=user_embedding_name,
        )
        self.audio_embeddings = TrackEmbeddingDB(
            dataset_name=track_embedding_dataset_name,
            split_types=track_split_types,
            embedding_name=audio_embedding_name,
        )
        self.metadata_embeddings = (
            TrackEmbeddingDB(
                dataset_name=track_embedding_dataset_name,
                split_types=track_split_types,
                embedding_name=metadata_embedding_name,
            )
            if metadata_embedding_name
            else None
        )
        from mcrs.experiments.exp014_lightgbm_ltr_reranker.features import LGBMFeatureBuilder

        inferred_source_names = self._infer_source_names(self.feature_names)
        self.feature_builder = LGBMFeatureBuilder(
            item_db=self.item_db,
            track_embedding_db=self.track_embeddings,
            user_embedding_db=self.user_embeddings,
            audio_embedding_db=self.audio_embeddings,
            metadata_embedding_db=self.metadata_embeddings,
            source_names=inferred_source_names,
            source_weights=source_weights,
        )

    @staticmethod
    def _infer_source_names(feature_names: list[str]) -> tuple[str, ...]:
        """LightGBM feature 名から候補 source 名を復元する。

        exp016 のように experiment-local source を増やしたモデルでも online
        scoring と offline training の feature schema を揃えるため、保存済み
        Booster の feature 名を一次情報として使う。

        Args:
            feature_names: LightGBM Booster に保存されている feature 名。

        Returns:
            `f_source_{source}_hit/rank/rrf` から復元した source 名。見つからない
            場合は exp014 の既定 source 名を返す。
        """
        from mcrs.experiments.exp014_lightgbm_ltr_reranker.candidate_sources import SOURCE_NAMES

        source_names: list[str] = []
        for feature_name in feature_names:
            if not feature_name.startswith("f_source_"):
                continue
            suffix = None
            for candidate_suffix in ("_hit", "_rank", "_rrf"):
                if feature_name.endswith(candidate_suffix):
                    suffix = candidate_suffix
                    break
            if suffix is None:
                continue
            source_name = feature_name[len("f_source_"):-len(suffix)]
            if source_name and source_name not in source_names:
                source_names.append(source_name)
        return tuple(source_names) if source_names else tuple(SOURCE_NAMES)

    def rerank(
        self,
        candidate_track_ids: list[str],
        user_id: str | None = None,
        session_memory: list[dict[str, Any]] | None = None,
        topk: int = 20,
        query: str | None = None,
    ) -> list[str]:
        """source 情報なしの簡易リランク API。

        source 別リストを持たない呼び出し元向けのラッパー。
        内部で dummy source を構築して rerank_with_sources に委譲する。

        Args:
            candidate_track_ids: リランク対象の track_id リスト。
            user_id: ユーザ ID（user embedding の取得に使用）。
            session_memory: 対話履歴。
            topk: 返す上位件数。
            query: クエリ文字列（None の場合は空文字列で補完）。

        Returns:
            スコア降順の track_id リスト（上位 topk 件）。
        """
        # source 情報がない簡易 API では dummy source 名で渡して feature_builder との整合を取る
        candidates_by_source = {"retrieval_input": list(dict.fromkeys(candidate_track_ids))}
        # feature_builder が複数の query view key を要求するため、同一クエリで全 key を埋める
        query_views = {
            "bm25_recent_turns": query or "",
            "bm25_current_user_turn": query or "",
            "bm25_full_context": query or "",
        }
        return self.rerank_with_sources(
            candidates_by_source=candidates_by_source,
            union_candidates=candidate_track_ids,
            query_views=query_views,
            user_id=user_id,
            session_memory=session_memory or [],
            topk=topk,
        )

    def batch_rerank(
        self,
        batch_candidate_track_ids: list[list[str]],
        user_ids: list[str | None],
        session_memories: list[list[dict[str, Any]]],
        topk: int = 20,
        queries: list[str] | None = None,
    ) -> list[list[str]]:
        """rerank を複数件まとめて実行するラッパー。

        Args:
            batch_candidate_track_ids: 各アイテムの候補 track_id リストのリスト。
            user_ids: 各アイテムのユーザ ID リスト。
            session_memories: 各アイテムの対話履歴リスト。
            topk: 返す上位件数。
            queries: 各アイテムのクエリ文字列リスト（None の場合は空文字列）。

        Returns:
            各アイテムのスコア降順 track_id リストのリスト。
        """
        return [
            self.rerank(
                candidate_track_ids=candidates,
                user_id=user_ids[index],
                session_memory=session_memories[index],
                topk=topk,
                query=queries[index] if queries else None,
            )
            for index, candidates in enumerate(batch_candidate_track_ids)
        ]

    def rerank_with_sources(
        self,
        *,
        candidates_by_source: dict[str, list[str]],
        union_candidates: list[str],
        query_views: dict[str, str],
        user_id: str | None,
        session_memory: list[dict[str, Any]],
        topk: int = 20,
    ) -> list[str]:
        """source 別候補リストを受け取り、LightGBM でスコアリングして上位 topk を返す。

        Args:
            candidates_by_source: ソース名 → track_id リストのマップ。
            union_candidates: 全候補の統合リスト（重複可、順序が特徴量計算に影響する）。
            query_views: BM25 query の各ビュー
                （bm25_recent_turns / bm25_current_user_turn / bm25_full_context）。
            user_id: ユーザ ID（user embedding の取得に使用）。
            session_memory: 対話履歴。
            topk: 返す上位件数。

        Returns:
            スコア降順の track_id リスト（上位 topk 件）。
        """
        unique_candidates = list(dict.fromkeys(union_candidates))
        if not unique_candidates:
            return []
        # feature_builder が期待する全 source_name を先にデフォルト空リストで埋め、追加ソースを後から補完する
        normalized_sources = {
            source: candidates_by_source.get(source, [])
            for source in self.feature_builder.source_names
        }
        normalized_sources.update(
            {
                source: candidates
                for source, candidates in candidates_by_source.items()
                if source not in normalized_sources
            }
        )
        rows = self.feature_builder.build_rows(
            candidates_by_source=normalized_sources,
            union_candidates=unique_candidates,
            query_views=query_views,
            user_id=user_id,
            session_memory=session_memory,
            topk=len(unique_candidates),
        )
        frame = pd.DataFrame(rows)
        for feature_name in self.feature_names:
            if feature_name not in frame.columns:
                frame[feature_name] = np.nan
            frame[feature_name] = pd.to_numeric(frame[feature_name], errors="coerce")
        scores = self.booster.predict(
            frame[self.feature_names],
            num_iteration=self.booster.best_iteration,
        )
        # 同スコア時は union_candidates への投入順（arange）を維持してタイブレークする
        order = np.lexsort((np.arange(len(unique_candidates)), -scores))[:topk]
        return [unique_candidates[index] for index in order.tolist()]
