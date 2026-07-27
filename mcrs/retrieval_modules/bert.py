"""BERT embedding による track metadata retrieval utility。

指定 metadata フィールドを BERT encoder で embedding 化して cache し、
query embedding との cosine similarity で候補 track を取得する。
"""
import os
import json
import re
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel


class BERT_MODEL:
    """mean pooling した BERT embedding で検索する metadata retriever。"""

    def __init__(self,
        dataset_name,
        split_types,
        corpus_types,
        cache_dir: str = "./cache",
        model_name: str = "bert-base-uncased",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 128
    ) -> None:
        """BERT retriever を初期化し、必要なら embedding index を構築する。

        Args:
            dataset_name: track metadata を含む Hugging Face dataset 名。
            split_types: 読み込んで連結する split 名。
            corpus_types: embedding 化する metadata フィールド。
            cache_dir: embedding index と付随 artifact の cache ディレクトリ。
            model_name: encoder として使う Hugging Face モデル名。
            device: torch device。None なら CUDA があれば CUDA、なければ CPU。
            batch_size: index 構築時の embedding 計算 batch size。
            max_length: tokenizer の最大系列長。
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.index_dir = os.path.join(
            self.cache_dir,
            "bert",
            self._cache_key(),
        )
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.metadata_dict = self._load_corpus()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

        if os.path.exists(os.path.join(self.index_dir, "embeddings.pt")) and \
           os.path.exists(os.path.join(self.index_dir, "track_ids.json")):
            self.embeddings, self.track_ids = self._load_index()
        else:
            self.build_index()
            self.embeddings, self.track_ids = self._load_index()
        self.search_embeddings = self.embeddings.to(self.device)

    def _cache_key(self) -> str:
        """index 内容に影響する設定を反映した cache key を作る。"""
        model_key = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model_name).strip("_")
        split_key = "_".join(self.split_types)
        return f"{model_key}__{split_key}__{self.corpus_name}__max{self.max_length}"

    def _load_index(self) -> Tuple[torch.Tensor, List[str]]:
        """cache 済み embedding 行列と track ID リストを読み込む。

        Returns:
            embedding 行列 ``[num_items, dim]`` と track ID リスト。
        """
        embeddings = torch.load(os.path.join(self.index_dir, "embeddings.pt"), map_location="cpu")
        track_ids = json.load(open(os.path.join(self.index_dir, "track_ids.json"), "r"))
        return embeddings, track_ids

    def _load_corpus(self) -> Dict[str, Dict]:
        """設定された split の metadata を読み込んで track_id keyed dict にする。

        Returns:
            track_id から metadata 辞書への mapping。
        """
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in self.split_types])
        metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        return metadata_dict

    def _stringify_metadata(self, metadata: Dict[str, object]) -> str:
        """metadata 辞書を BERT encoder へ渡す複数行文字列へ変換する。

        Args:
            metadata: ``self.corpus_types`` のフィールドを持つ track metadata。

        Returns:
            選択フィールドごとの ``field: value`` を改行連結した文字列。
        """
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata[corpus_type]
            if isinstance(entity, list):
                entity = ", ".join(entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def _mean_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """attention mask を考慮して token embedding を平均 pooling する。

        Args:
            last_hidden_states: ``[batch, seq_len, hidden]`` の token embedding。
            attention_mask: ``[batch, seq_len]`` の attention mask。

        Returns:
            ``[batch, hidden]`` の pooled embedding。
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        summed = torch.sum(last_hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def build_index(self) -> None:
        """読み込み済み corpus から embedding index を構築して cache に保存する。"""
        track_ids = list(self.metadata_dict.keys())
        corpus_texts = []
        for track_id in track_ids:
            metadata = self.metadata_dict[track_id]
            corpus_texts.append(self._stringify_metadata(metadata))
        os.makedirs(self.index_dir, exist_ok=True)
        embeddings: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(corpus_texts), self.batch_size):
                batch_texts = corpus_texts[start:start + self.batch_size]
                self.tokenizer.truncation_side = "right"
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
                # 検索時は内積で cosine similarity を計算できるよう、保存前に L2 正規化する。
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.detach().cpu())

        embedding_mat = torch.cat(embeddings, dim=0).contiguous()
        torch.save(embedding_mat, os.path.join(self.index_dir, "embeddings.pt"))
        with open(os.path.join(self.index_dir, "track_ids.json"), "w") as f:
            json.dump(track_ids, f, indent=2)

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        """自然言語 query に対する上位 track ID を cosine similarity で返す。

        Args:
            query: embedding 化して corpus と照合するユーザ query。
            topk: 返す track 数。

        Returns:
            cosine similarity 降順の track ID リスト。
        """
        self.model.eval()
        with torch.no_grad():
            # CRS の retrieval 入力は末尾に最新発話が来るため、長い履歴では左側を切り詰める。
            self.tokenizer.truncation_side = "left"
            batch = self.tokenizer([query], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            query_emb = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            query_emb = F.normalize(query_emb, p=2, dim=1).squeeze(0)
        # corpus/query ともに L2 正規化済みなので、内積が cosine similarity になる。
        scores = torch.matmul(self.search_embeddings, query_emb)
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.cpu().tolist()
        return [self.track_ids[i] for i in top_indices]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        """複数 query に対する上位 track ID を batch 取得する。

        Args:
            queries: embedding 化して corpus と照合する query リスト。
            topk: query ごとに返す track 数。

        Returns:
            query ごとの cosine similarity 降順 track ID リスト。
        """
        self.model.eval()
        with torch.no_grad():
            # 最新発話を残すため、batch 検索でも左側 truncation に揃える。
            self.tokenizer.truncation_side = "left"
            batch = self.tokenizer(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            query_embs = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            query_embs = F.normalize(query_embs, p=2, dim=1)
        # 全 query をまとめて cosine similarity 計算する。
        scores = torch.matmul(self.search_embeddings, query_embs.T)
        results = []
        topk = min(topk, scores.shape[0])
        top_indices_by_query = torch.topk(scores, k=topk, dim=0).indices.T.cpu().tolist()
        for top_indices in top_indices_by_query:
            results.append([self.track_ids[idx] for idx in top_indices])
        return results
