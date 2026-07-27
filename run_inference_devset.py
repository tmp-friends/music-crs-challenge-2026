"""
Batch inference script for Music CRS.
"""

import os
import json
import torch
import argparse
from importlib import import_module
from mcrs import load_crs_baseline as default_load_crs_baseline
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import pandas as pd
from mcrs.experiments.configs import load_experiment_config


def resolve_load_crs_baseline(config):
    experiment_module = config.get("experiment_module", None)
    if experiment_module is None or str(experiment_module).strip().lower() in {"", "none", "null"}:
        return default_load_crs_baseline

    module = import_module(str(experiment_module))
    if not hasattr(module, "load_crs_baseline"):
        raise AttributeError(
            f"Experiment module '{experiment_module}' must expose load_crs_baseline(...)."
        )
    return module.load_crs_baseline


def chat_history_parser(conversations, music_crs, target_turn_number):
    """
    Parse conversation history up to a target turn.

    Args:
        conversations (List[Dict]): List of conversation turn dictionaries containing:
            - turn_number: Turn index (1-8)
            - role: Speaker role ('user', 'assistant', 'music')
            - content: Message content or track ID
        music_crs: CRS baseline instance (used to convert track IDs to metadata)
        target_turn_number (int): The turn to predict (history excludes this turn)

    Returns:
        Tuple[List[Dict], str]:
            - chat_history: List of previous messages formatted as [{"role": ..., "content": ...}]
            - user_query: The user query at the target turn
    """
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation['turn_number'] < target_turn_number]
    chat_history = []
    for turn_data in df_history.to_dict(orient="records"):
        current_role = turn_data['role']
        current_content = turn_data['content']
        chat_history.append({
            "role": current_role,
            "content": current_content
        })
    df_current_turn = df_conversation[df_conversation['turn_number'] == target_turn_number]
    user_query = df_current_turn.iloc[0]['content']
    return chat_history, user_query

def main(args):
    """
    Run batch inference on TalkPlayData-2 test dataset.

    Args:
        args: Namespace object containing:
            - tid (str): Task/configuration identifier
            - batch_size (int): Batch size for inference
            - save_path (str): Output directory (currently unused)

    Returns:
        None. Results are saved to exp/inference/{tid}.json

    Processing:
        - Loads model configuration from mcrs/experiments/*/configs/{tid}.yaml
        - Processes all sessions × 8 turns in batches
        - Tracks progress with tqdm progress bar
        - Saves comprehensive results for evaluation
    """
    print("Removing cache directory for preventing memory issues...")
    os.system("rm -rf cache")
    config, config_path = load_experiment_config(args.tid)
    print(f"Loaded config: {config_path}")
    load_crs_baseline = resolve_load_crs_baseline(config)
    crs_kwargs = {
        "lm_type": config.lm_type,
        "retrieval_type": config.retrieval_type,
        "item_db_name": config.item_db_name,
        "user_db_name": config.user_db_name,
        "track_split_types": config.track_split_types,
        "user_split_types": config.user_split_types,
        "corpus_types": config.corpus_types,
        "cache_dir": config.cache_dir,
        "device": config.device,
        "attn_implementation": config.attn_implementation,
        "dtype": torch.bfloat16,
        "reranker_type": config.get("reranker_type", None),
        "retrieval_topk": config.get("retrieval_topk", 20),
        "rerank_topk": config.get("rerank_topk", 20),
        "track_embedding_dataset_name": config.get(
            "track_embedding_dataset_name",
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        ),
        "user_embedding_dataset_name": config.get(
            "user_embedding_dataset_name",
            "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        ),
        "track_embedding_name": config.get("track_embedding_name", "cf-bpr"),
        "audio_embedding_name": config.get("audio_embedding_name", "audio-laion_clap"),
        "user_embedding_name": config.get("user_embedding_name", "cf-bpr"),
        "user_embedding_split_types": config.get("user_embedding_split_types", None),
        "reranker_weights": config.get("reranker_weights", None),
        "reranker_model_name": config.get("reranker_model_name", "Qwen/Qwen3-Reranker-4B"),
        "reranker_device": config.get("reranker_device", config.device),
        "reranker_attn_implementation": config.get(
            "reranker_attn_implementation",
            config.attn_implementation,
        ),
        "reranker_dtype": config.get("reranker_dtype", torch.bfloat16),
        "reranker_batch_size": config.get("reranker_batch_size", 8),
        "reranker_max_length": config.get("reranker_max_length", 8192),
        "reranker_instruction": config.get("reranker_instruction", None),
        "lm_max_new_tokens": config.get("lm_max_new_tokens", 1024),
    }
    metadata_exact_fields = config.get("metadata_exact_fields", None)
    if metadata_exact_fields is not None:
        crs_kwargs["metadata_exact_fields"] = metadata_exact_fields
    metadata_bm25_corpus_types = config.get("metadata_bm25_corpus_types", None)
    if metadata_bm25_corpus_types is not None:
        crs_kwargs["metadata_bm25_corpus_types"] = list(metadata_bm25_corpus_types)
    bert_model_name = config.get("bert_model_name", None)
    if bert_model_name is not None:
        crs_kwargs["bert_model_name"] = str(bert_model_name)
        crs_kwargs["bert_max_length"] = config.get("bert_max_length", 128)
    enabled_sources = config.get("enabled_sources", None)
    if enabled_sources is not None:
        crs_kwargs["enabled_sources"] = list(enabled_sources)
    policy_id = config.get("policy_id", None)
    if policy_id is not None:
        crs_kwargs["policy_id"] = str(policy_id)
    source_topk = config.get("source_topk", None)
    if source_topk is not None:
        crs_kwargs["source_topk"] = int(source_topk)
    final_candidate_cap = config.get("final_candidate_cap", None)
    if final_candidate_cap is not None:
        crs_kwargs["final_candidate_cap"] = final_candidate_cap
    # fine-tuned dense source (exp016) 用 config の forwarding。config に存在する
    # experiment だけ有効化されるため、他 experiment の config には影響しない（後方互換）。
    ft_dense_model_dir = config.get("ft_dense_model_dir", None)
    if ft_dense_model_dir is not None:
        crs_kwargs["ft_dense_model_dir"] = str(ft_dense_model_dir)
        crs_kwargs["ft_dense_index_dir"] = config.get("ft_dense_index_dir", None)
        crs_kwargs["ft_dense_device"] = config.get("ft_dense_device", None)
        crs_kwargs["ft_dense_max_length"] = int(config.get("ft_dense_max_length", 256))
        crs_kwargs["ft_dense_batch_size"] = int(config.get("ft_dense_batch_size", 32))
    music_crs = load_crs_baseline(**crs_kwargs)
    db = load_dataset(config.test_dataset_name, split="test")
    # Prepare all batch data at once
    batch_data, metadata = [], []
    for item in db:
        user_id = item['user_id']
        session_id = item['session_id']
        for target_turn_number in range(1, 9):
            chat_history, user_query = chat_history_parser(item['conversations'], music_crs, target_turn_number)
            batch_data.append({
                'user_query': user_query,
                'user_id': user_id,
                'session_memory': chat_history
            })
            metadata.append({
                'session_id': session_id,
                'user_id': user_id,
                'turn_number': target_turn_number
            })
    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch = batch_data[i:i+args.batch_size]
        batch_metadata = metadata[i:i+args.batch_size]
        results = music_crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id": batch_metadata[j]['session_id'],
                "user_id": batch_metadata[j]['user_id'],
                "turn_number": batch_metadata[j]['turn_number'],
                "predicted_track_ids": result['retrieval_items'],
                "predicted_response": result["response"]
            })
    os.makedirs("exp/inference/devset", exist_ok=True)
    with open(f"exp/inference/devset/{args.tid}.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_bm25_devset",
        help="Task identifier matching mcrs/experiments/*/configs/{tid}.yaml"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of queries to process in parallel. Reduce if encountering GPU memory issues."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/inference",
        help="Base directory for saving results (currently not used, results saved to exp/inference/)"
    )
    args = parser.parse_args()
    main(args)
