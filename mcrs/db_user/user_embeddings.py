"""personalization reranking で使う user embedding accessor。"""

import torch
from datasets import concatenate_datasets, load_dataset


class UserEmbeddingDB:
    """user_id から embedding tensor を取得する軽量 DB。"""

    def __init__(
        self,
        dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        split_types: list[str] | None = None,
        embedding_name: str = "cf-bpr",
    ) -> None:
        """指定 embedding カラムを読み込み、user_id ごとに tensor 化する。

        Args:
            dataset_name: Hugging Face 上の user embedding dataset 名。
            split_types: 読み込む split 名。None の場合は既定の warm/cold split を読む。
            embedding_name: 使用する embedding カラム名。
        """
        self.dataset_name = dataset_name
        self.split_types = split_types or ["train", "test_warm", "test_cold"]
        self.embedding_name = embedding_name
        embedding_dataset = load_dataset(dataset_name)
        embedding_concat_dataset = concatenate_datasets(
            [embedding_dataset[split_type] for split_type in self.split_types]
        )
        self.embeddings = {}
        for item in embedding_concat_dataset:
            values = item.get(self.embedding_name)
            self.embeddings[item["user_id"]] = self._to_tensor(values)

    def _to_tensor(self, values) -> torch.Tensor | None:
        """dataset の list 値を float32 tensor へ変換する。"""
        if values is None or len(values) == 0:
            return None
        vector = torch.tensor(values, dtype=torch.float32)
        if vector.numel() == 0:
            return None
        return vector

    def get(self, user_id: str | None) -> torch.Tensor | None:
        """user_id に対応する embedding を返す。未登録または None なら None を返す。"""
        if user_id is None:
            return None
        return self.embeddings.get(user_id)

    @staticmethod
    def is_valid(vector: torch.Tensor | None) -> bool:
        """reranking に使える非ゼロ embedding かどうかを判定する。"""
        return vector is not None and vector.numel() > 0 and torch.linalg.vector_norm(vector).item() > 0.0
