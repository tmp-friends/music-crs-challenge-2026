"""fine-tuned bi-encoder checkpoint を使う dense retriever（共有版）。

exp016 の ``dense_retriever.py`` から forward port した共有モジュール。exp016 は
凍結 fork として残し、exp017 以降の新規コード（hard negative mining / 学習 /
index 構築）はこちらを import する（実験管理ガイド §1-2: retrieval backend は
``mcrs/retrieval_modules/`` へ置き、expNNN フォルダ間 import を増やさない）。

artifact 形式（HF checkpoint dir + ``embeddings.pt`` / ``track_ids.json`` /
``passage_texts.jsonl`` / ``index_meta.json``）は exp016 と完全互換であり、
ここで作った model/index は exp016 の凍結 pipeline（b2dense 系）からそのまま
読み込める。pooling は exp016 と同じ mean pooling を維持する（推論側の
``DenseRetrieverFT.mean_pool`` と学習側で一致させないと embedding 空間が
ズレるため、変更しないこと）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DenseRetrieverFT:
    """保存済み dense index に対して cosine similarity 検索を行う retriever。"""

    def __init__(
        self,
        *,
        model_dir: str,
        index_dir: str,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        """retriever checkpoint と all_tracks index を読み込む。

        Args:
            model_dir: Hugging Face 形式の encoder checkpoint。
            index_dir: `embeddings.pt` と `track_ids.json` を含む index directory。
            device: 推論 device。None の場合は CUDA があれば CUDA。
            batch_size: index/query encode 時の batch size。
            max_length: tokenizer 最大長。

        Raises:
            FileNotFoundError: index artifact が見つからない場合。
        """
        self.model_dir = model_dir
        self.index_dir = Path(index_dir)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        embeddings_path = self.index_dir / "embeddings.pt"
        track_ids_path = self.index_dir / "track_ids.json"
        if not embeddings_path.exists() or not track_ids_path.exists():
            raise FileNotFoundError(
                f"Dense index is incomplete: embeddings={embeddings_path}, track_ids={track_ids_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        self.embeddings = torch.load(embeddings_path, map_location="cpu")
        with open(track_ids_path, "r", encoding="utf-8") as f:
            self.track_ids = json.load(f)
        self.search_embeddings = self.embeddings.to(self.device)

        # query 側 prefix を index_meta.json から読む（e5 系の "query: " 等）。
        # passage 側 prefix は index 構築時に passage text へ既に焼き込まれているため
        # 推論側で必要なのは query prefix のみ。prefix 未保存の既存 index（bge_small /
        # bge_base）では "" となり従来挙動と完全一致する（後方互換）。学習・index 構築・
        # 推論で同じ prefix を使わないと embedding 空間がズレるため、推論側の唯一の真実は
        # この index_meta.json に保存された query_prefix とする。
        self.query_prefix = ""
        # pool_type は encoder 系（bge/e5/gte/mpnet）が "mean"、decoder LLM embedder
        # （Qwen3-Embedding 等）が "last"（最終非 pad token = EOS 表現）。prefix と同様に
        # index_meta.json を推論側の唯一の真実とし、学習・index・推論で pooling を一致させる。
        # 未保存の既存 index は "mean" となり従来挙動と完全一致（後方互換）。
        self.pool_type = "mean"
        meta_path = self.index_dir / "index_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.query_prefix = str(meta.get("query_prefix", "") or "")
            self.pool_type = str(meta.get("pool_type", "mean") or "mean")

    @staticmethod
    def mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """attention mask を考慮して token embedding を平均 pooling する。"""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        summed = torch.sum(last_hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """各系列の最終非 pad token の hidden state を返す（decoder LLM embedder 用）。

        left-padding / right-padding の両方に対応する（Qwen3-Embedding 公式実装と同じ）。
        left-padding なら最終列がそのまま最終 token、right-padding なら attention_mask の
        長さ-1 の位置を拾う。
        """
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths]

    @classmethod
    def pool(cls, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str) -> torch.Tensor:
        """pool_type に応じて mean / last-token pooling を切り替える共通 helper。"""
        if pool_type == "last":
            return cls.last_token_pool(last_hidden_states, attention_mask)
        return cls.mean_pool(last_hidden_states, attention_mask)

    def encode_texts(
        self, texts: list[str], *, truncation_side: str = "left", prefix: str = ""
    ) -> torch.Tensor:
        """テキスト列を L2 正規化済み embedding に変換する。

        Args:
            texts: encode 対象テキスト。
            truncation_side: tokenizer の truncation 方向。
            prefix: 各テキスト先頭へ付与する prefix（e5 系の "query: " / "passage: "）。
                truncation の前に prepend するため、truncation_side="left" の query では
                prefix が落ちないよう短い prefix を前提とする。空文字なら従来挙動。

        Returns:
            CPU 上の `[len(texts), hidden]` embedding tensor。
        """
        if prefix:
            texts = [prefix + text for text in texts]
        embeddings: list[torch.Tensor] = []
        self.model.eval()
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = truncation_side
        try:
            with torch.no_grad():
                for start in range(0, len(texts), self.batch_size):
                    batch_texts = texts[start:start + self.batch_size]
                    batch = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    outputs = self.model(**batch)
                    pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"], self.pool_type)
                    embeddings.append(F.normalize(pooled, p=2, dim=1).detach().cpu())
        finally:
            self.tokenizer.truncation_side = original_side
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        """単一 query に対する上位 track ID を返す。"""
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        """複数 query に対する上位 track ID を返す。

        Args:
            queries: 検索 query text。
            topk: query ごとに返す件数。

        Returns:
            query ごとの track_id ranked list。
        """
        if not queries:
            return []
        query_embs = self.encode_texts(
            queries, truncation_side="left", prefix=self.query_prefix
        ).to(self.device)
        scores = torch.matmul(self.search_embeddings, query_embs.T)
        k = min(int(topk), scores.shape[0])
        top_indices_by_query = torch.topk(scores, k=k, dim=0).indices.T.cpu().tolist()
        return [[self.track_ids[index] for index in top_indices] for top_indices in top_indices_by_query]

    def batch_text_to_item_retrieval_with_scores(
        self, queries: list[str], topk: int
    ) -> tuple[list[list[str]], list[list[float]]]:
        """複数 query に対する上位 track ID と cosine score を返す。

        ``batch_text_to_item_retrieval`` と同一の検索（L2 正規化済み embedding の内積
        = cosine similarity の topk）を行い、track_id に加えてその score も返す。
        Stage C 系の source confidence feature（``f_dense_cos_*``）が train / inference
        の両経路で同じ score を参照できるようにするための拡張で、ranking 用の候補
        決定ロジック自体は ``batch_text_to_item_retrieval`` と完全に一致する。

        Args:
            queries: 検索 query text。
            topk: query ごとに返す件数。

        Returns:
            ``(ids_by_query, scores_by_query)``。``scores_by_query[i][j]`` は
            ``ids_by_query[i][j]`` の cosine similarity（L2 正規化内積、概ね [-1, 1]）。
        """
        if not queries:
            return [], []
        query_embs = self.encode_texts(
            queries, truncation_side="left", prefix=self.query_prefix
        ).to(self.device)
        scores = torch.matmul(self.search_embeddings, query_embs.T)
        k = min(int(topk), scores.shape[0])
        top = torch.topk(scores, k=k, dim=0)
        indices_by_query = top.indices.T.cpu().tolist()
        values_by_query = top.values.T.cpu().tolist()
        ids_by_query = [
            [self.track_ids[index] for index in indices] for indices in indices_by_query
        ]
        return ids_by_query, values_by_query


def save_dense_index(
    *,
    model_dir: str,
    index_dir: str,
    track_ids: list[str],
    passage_texts: list[str],
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 192,
    index_meta: dict[str, Any] | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    pool_type: str = "mean",
) -> None:
    """track passage embedding index を作成して保存する。

    Args:
        model_dir: encode に使う checkpoint。
        index_dir: 保存先 directory。
        track_ids: embedding 行に対応する track_id。
        passage_texts: track metadata から作った passage text。
        device: encode device。
        batch_size: encode batch size。
        max_length: tokenizer 最大長。
        index_meta: `index_meta.json` に追記する任意 metadata。
        query_prefix: 推論時に query 先頭へ付与する prefix（e5 系 "query: " 等）。
            `index_meta.json` に保存され、`DenseRetrieverFT` が読み込んで適用する。
        passage_prefix: passage encode 前に付与する prefix（e5 系 "passage: " 等）。
            ここで passage text へ焼き込むため、推論側では passage prefix は不要。
            query/passage prefix は学習時と同一にすること（ズレると embedding 空間が崩れる）。
        pool_type: "mean"（encoder 系）または "last"（decoder LLM embedder, 最終 token）。
            学習時と同一にすること。`index_meta.json` に保存され推論側が読み込む。
    """
    os.makedirs(index_dir, exist_ok=True)
    retriever = object.__new__(DenseRetrieverFT)
    retriever.model_dir = model_dir
    retriever.index_dir = Path(index_dir)
    retriever.batch_size = int(batch_size)
    retriever.max_length = int(max_length)
    retriever.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    retriever.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    retriever.model = AutoModel.from_pretrained(model_dir)
    retriever.model.to(retriever.device).eval()
    # object.__new__ で __init__ を通さないため、encode_texts が参照する pool_type を明示設定。
    retriever.pool_type = pool_type

    embeddings = retriever.encode_texts(
        passage_texts, truncation_side="right", prefix=passage_prefix
    ).contiguous()
    torch.save(embeddings, Path(index_dir) / "embeddings.pt")
    with open(Path(index_dir) / "track_ids.json", "w", encoding="utf-8") as f:
        json.dump(track_ids, f, ensure_ascii=False, indent=2)
    with open(Path(index_dir) / "passage_texts.jsonl", "w", encoding="utf-8") as f:
        for track_id, text in zip(track_ids, passage_texts):
            f.write(json.dumps({"track_id": track_id, "text": text}, ensure_ascii=False) + "\n")
    meta = {
        "model_dir": model_dir,
        "num_tracks": len(track_ids),
        "max_length": max_length,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        # query_prefix は推論側 DenseRetrieverFT が読み込んで query へ適用する。
        # passage_prefix は既に passage text へ焼き込み済みだが、再現用に記録する。
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        # pool_type も推論側が読み込んで pooling を一致させる（mean/last）。
        "pool_type": pool_type,
    }
    if index_meta:
        meta.update(index_meta)
    with open(Path(index_dir) / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
