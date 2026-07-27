"""retrieval 候補を並べ替える reranker module の loader。"""

from .lightgbm_ltr import LightGBMLTRReranker


def load_reranker_module(
    reranker_type: str | None,
    track_split_types: list[str],
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    corpus_types: list[str] | None = None,
    cache_dir: str = "./cache",
    track_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    user_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    track_embedding_name: str = "cf-bpr",
    audio_embedding_name: str = "audio-laion_clap",
    user_embedding_name: str = "cf-bpr",
    user_embedding_split_types: list[str] | None = None,
    reranker_weights: dict[str, float] | None = None,
    reranker_model_name: str = "Qwen/Qwen3-Reranker-4B",
):
    """reranker_type に対応する reranker module を初期化する。

    Args:
        reranker_type: 使用する reranker 名。None/空文字/``none`` は reranker なし。
            現在サポートするのは ``lightgbm_ltr`` のみ。
        track_split_types: track metadata/embedding の読み込み split。
        item_db_name: track metadata dataset 名。
        corpus_types: feature 構築や prompt 表示に使う metadata フィールド。
        cache_dir: reranker が必要に応じて使う cache ディレクトリ。
        track_embedding_dataset_name: track embedding dataset 名。
        user_embedding_dataset_name: user embedding dataset 名。
        track_embedding_name: CF などの track embedding カラム名。
        audio_embedding_name: audio embedding カラム名。
        user_embedding_name: user embedding カラム名。
        user_embedding_split_types: user embedding の読み込み split。
        reranker_weights: reranker 固有の重み設定。
        reranker_model_name: LightGBM reranker のモデルパス。

    Returns:
        rerank API を持つ reranker、または reranker 無効時の None。

    Raises:
        ValueError: 未対応の reranker_type が指定された場合。
    """
    if reranker_type is None or str(reranker_type).lower() in {"", "none", "null"}:
        return None
    if reranker_type == "lightgbm_ltr":
        return LightGBMLTRReranker(
            model_path=reranker_model_name,
            item_db_name=item_db_name,
            track_embedding_dataset_name=track_embedding_dataset_name,
            user_embedding_dataset_name=user_embedding_dataset_name,
            track_split_types=track_split_types,
            corpus_types=corpus_types or ["track_name", "artist_name", "album_name", "release_date"],
            user_embedding_split_types=user_embedding_split_types,
            track_embedding_name=track_embedding_name,
            audio_embedding_name=audio_embedding_name,
            user_embedding_name=user_embedding_name,
            source_weights=reranker_weights,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unsupported reranker type: {reranker_type}")
