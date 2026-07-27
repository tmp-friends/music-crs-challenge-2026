"""Music-CRS baseline を組み立てる公開ファクトリを提供する。"""

import torch
from .crs_baseline import CRS_BASELINE

def load_crs_baseline(
    lm_type="meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type="bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types: list[str] = ["all_tracks"],
    user_split_types: list[str] = ["all_users"],
    corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
    cache_dir="./cache",
    device="cuda",
    attn_implementation="eager",
    dtype=torch.bfloat16,
    reranker_type=None,
    retrieval_topk: int = 20,
    rerank_topk: int = 20,
    track_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    user_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    track_embedding_name: str = "cf-bpr",
    audio_embedding_name: str = "audio-laion_clap",
    user_embedding_name: str = "cf-bpr",
    user_embedding_split_types: list[str] | None = None,
    reranker_weights: dict[str, float] | None = None,
    reranker_model_name: str = "Qwen/Qwen3-Reranker-4B",
    reranker_device: str | None = None,
    reranker_attn_implementation: str | None = None,
    reranker_dtype=None,
    reranker_batch_size: int = 8,
    reranker_max_length: int = 8192,
    reranker_instruction: str | None = None,
    lm_max_new_tokens: int = 1024,
    metadata_exact_fields: list[str] | None = None,
):
    """CRS baseline の主要コンポーネントを設定値から初期化する。

    Args:
        lm_type: 応答生成に使う Hugging Face causal LM のモデル名。
        retrieval_type: 候補生成に使う retrieval backend 名。
        item_db_name: track metadata dataset 名。
        user_db_name: user metadata dataset 名。
        track_split_types: track metadata の読み込み split。提出では通常 ``["all_tracks"]``。
        user_split_types: user metadata の読み込み split。
        corpus_types: retrieval と応答表示に使う track metadata のフィールド。
        cache_dir: index や中間 artifact の cache ディレクトリ。
        device: LM と一部 reranker を配置する torch device。
        attn_implementation: Transformers に渡す attention 実装名。
        dtype: LM の torch dtype。
        reranker_type: retrieval 後に使う reranker 名。None なら rerank しない。
        retrieval_topk: retrieval が返す候補数。
        rerank_topk: 最終的に保持する推薦候補数。
        track_embedding_dataset_name: track embedding dataset 名。
        user_embedding_dataset_name: user embedding dataset 名。
        track_embedding_name: CF などの track embedding カラム名。
        audio_embedding_name: audio embedding カラム名。
        user_embedding_name: user embedding カラム名。
        user_embedding_split_types: user embedding の読み込み split。
        reranker_weights: reranker 固有の重み設定。
        reranker_model_name: LLM/LightGBM reranker のモデル名またはパス。
        reranker_device: reranker 専用 device。None なら ``device`` を使う。
        reranker_attn_implementation: reranker 専用 attention 実装。
        reranker_dtype: reranker 専用 dtype。
        reranker_batch_size: reranker の scoring batch size。
        reranker_max_length: LLM reranker の最大 token 長。
        reranker_instruction: LLM reranker に渡す判定 instruction。
        lm_max_new_tokens: 応答生成時の最大生成 token 数。
        metadata_exact_fields: metadata_exact retrieval で照合するフィールド。

    Returns:
        初期化済みの ``CRS_BASELINE`` インスタンス。
    """
    return CRS_BASELINE(
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
        reranker_device=reranker_device,
        reranker_attn_implementation=reranker_attn_implementation,
        reranker_dtype=reranker_dtype,
        reranker_batch_size=reranker_batch_size,
        reranker_max_length=reranker_max_length,
        reranker_instruction=reranker_instruction,
        lm_max_new_tokens=lm_max_new_tokens,
        metadata_exact_fields=metadata_exact_fields,
    )
