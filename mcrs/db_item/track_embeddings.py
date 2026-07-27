"""reranking で使う track embedding accessor。"""

from typing import Iterable

import torch
from datasets import concatenate_datasets, load_dataset


class TrackEmbeddingDB:
    """track_id から embedding tensor を取得する軽量 DB。"""

    def __init__(
        self,
        dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        split_types: list[str] | None = None,
        embedding_name: str = "cf-bpr",
    ) -> None:
        """指定 embedding カラムを読み込み、track_id ごとに tensor 化する。

        Args:
            dataset_name: Hugging Face 上の track embedding dataset 名。
            split_types: 読み込む split 名。None の場合は ``["all_tracks"]``。
            embedding_name: 使用する embedding カラム名。
        """
        self.dataset_name = dataset_name
        self.split_types = split_types or ["all_tracks"]
        self.embedding_name = embedding_name
        embedding_dataset = load_dataset(dataset_name)
        embedding_concat_dataset = concatenate_datasets(
            [embedding_dataset[split_type] for split_type in self.split_types]
        )
        self.embeddings = {}
        for item in embedding_concat_dataset:
            values = item.get(self.embedding_name)
            self.embeddings[item["track_id"]] = self._to_tensor(values)

    def _to_tensor(self, values) -> torch.Tensor | None:
        """dataset の list 値を float32 tensor へ変換する。"""
        if values is None or len(values) == 0:
            return None
        vector = torch.tensor(values, dtype=torch.float32)
        if vector.numel() == 0:
            return None
        return vector

    def get(self, track_id: str) -> torch.Tensor | None:
        """track_id に対応する embedding を返す。未登録なら None を返す。"""
        return self.embeddings.get(track_id)

    def mean_vector(self, track_ids: Iterable[str]) -> torch.Tensor | None:
        """複数 track の有効 embedding の平均ベクトルを返す。

        Args:
            track_ids: 平均対象の track ID iterable。

        Returns:
            有効な embedding が 1 つ以上あれば平均 tensor、なければ None。
        """
        vectors = [self.get(track_id) for track_id in track_ids]
        valid_vectors = [vector for vector in vectors if self.is_valid(vector)]
        if not valid_vectors:
            return None
        return torch.stack(valid_vectors, dim=0).mean(dim=0)

    @staticmethod
    def is_valid(vector: torch.Tensor | None) -> bool:
        """reranking に使える非ゼロ embedding かどうかを判定する。"""
        return vector is not None and vector.numel() > 0 and torch.linalg.vector_norm(vector).item() > 0.0
