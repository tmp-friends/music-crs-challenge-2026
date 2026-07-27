"""候補生成 retrieval module の loader。"""

from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .metadata_exact import MetadataExactRetriever

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache"
    ):
    """retrieval_type に対応する retrieval module を初期化する。

    Args:
        retrieval_type: ``bm25``、``bert``、``metadata_exact`` のいずれか。
        dataset_name: track metadata dataset 名。
        track_split_types: 読み込む track split。
        corpus_types: retrieval に使う metadata フィールド。
        cache_dir: index cache ディレクトリ。

    Returns:
        text_to_item_retrieval API を持つ retrieval module。

    Raises:
        ValueError: 未対応の retrieval_type が指定された場合。
    """
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        return BERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "metadata_exact":
        return MetadataExactRetriever(dataset_name, track_split_types, corpus_types, cache_dir)
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
