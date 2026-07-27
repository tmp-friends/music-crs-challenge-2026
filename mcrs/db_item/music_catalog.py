"""track metadata dataset を retrieval/応答生成向けに参照する。"""

import os
import torch
import json
from datasets import load_dataset, concatenate_datasets

class MusicCatalogDB:
    """track_id から metadata と LLM 向け文字列表現を取得する DB。"""

    def __init__(self,
            dataset_name: str,
            split_types: list[str],
            corpus_types: list[str],
        ):
        """指定 split の track metadata を読み込み、track_id で引ける形にする。

        Args:
            dataset_name: Hugging Face 上の track metadata dataset 名。
            split_types: 読み込む track split。提出では通常 ``["all_tracks"]``。
            corpus_types: 文字列表現へ含める metadata フィールド。
        """
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        """track_id に対応する metadata を 1 行の説明文字列へ変換する。

        Args:
            track_id: 参照する track ID。
            use_semantic_id: 互換性のために受け取る未使用フラグ。

        Returns:
            ``track_id`` と指定 metadata フィールドを連結した文字列。
        """
        metadata = self.metadata_dict[track_id]
        track_id = metadata['track_id']
        entity_str = f"track_id: {track_id}"
        for corpus_type in self.corpus_types:
            value = metadata[corpus_type]
            if isinstance(value, list):
                corpus_type_value = ", ".join(str(item) for item in value)
            elif value is None:
                corpus_type_value = ""
            else:
                corpus_type_value = str(value)
            corpus_type_value = corpus_type_value.lower()
            entity_str += f", {corpus_type}: {corpus_type_value}"
        return entity_str
