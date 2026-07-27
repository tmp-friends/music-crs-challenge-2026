"""Music-CRS の retrieval・reranking・応答生成を束ねる baseline pipeline。"""

import os
import torch
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.rerank_modules import load_reranker_module

class CRS_BASELINE:
    """Music-CRS baseline の対話推薦 pipeline。

    retrieval module で候補 track を取得し、任意の reranker で並べ替えた後、
    LLM module で推薦理由を含む自然言語応答を生成する。

    Attributes:
        cache_dir: index や artifact の cache ディレクトリ。
        lm_type: 応答生成に使う LLM の識別子。
        retrieval_type: 候補生成に使う retrieval backend 名。
        item_db_name: track metadata dataset 名。
        user_db_name: user metadata dataset 名。
        track_split_types: 読み込む track split 名。
        user_split_types: 読み込む user split 名。
        corpus_types: retrieval と表示に使う item metadata フィールド。
        lm: 応答生成 module。
        retrieval: 候補 track を返す retrieval module。
        reranker: retrieval 候補を並べ替える reranker。未使用時は None。
        item_db: track metadata accessor。
        user_db: user profile accessor。
        session_memory: 単発 chat API で保持する現在セッションの履歴。
    """
    def __init__(self,
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"], # 提出仕様では全 catalog を対象にする
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        attn_implementation="eager",
        dtype=torch.bfloat16,
        reranker_type: Optional[str] = None,
        retrieval_topk: int = 20,
        rerank_topk: int = 20,
        track_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embedding_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_embedding_name: str = "cf-bpr",
        audio_embedding_name: str = "audio-laion_clap",
        user_embedding_name: str = "cf-bpr",
        user_embedding_split_types: Optional[list[str]] = None,
        reranker_weights: Optional[dict[str, float]] = None,
        reranker_model_name: str = "Qwen/Qwen3-Reranker-4B",
        reranker_device: Optional[str] = None,
        reranker_attn_implementation: Optional[str] = None,
        reranker_dtype: Any = None,
        reranker_batch_size: int = 8,
        reranker_max_length: int = 8192,
        reranker_instruction: Optional[str] = None,
        lm_max_new_tokens: int = 1024,
        metadata_exact_fields: Optional[list[str]] = None,
    ):
        """CRS baseline の各 module と metadata DB を初期化する。

        Args:
            lm_type: 応答生成に使う Hugging Face causal LM のモデル名。
            retrieval_type: 候補生成に使う retrieval backend 名。
            item_db_name: track metadata dataset 名。
            user_db_name: user metadata dataset 名。
            track_split_types: track metadata の読み込み split。
            user_split_types: user metadata の読み込み split。
            corpus_types: retrieval と応答表示に使う metadata フィールド。
            cache_dir: index や中間 artifact の cache ディレクトリ。
            device: LM を配置する torch device。
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
            reranker_model_name: LightGBM reranker のモデルパス。
            reranker_device: （互換のため残置・現状未使用）LLM reranker 専用 device。
            reranker_attn_implementation: （互換のため残置・現状未使用）LLM reranker の attention 実装。
            reranker_dtype: （互換のため残置・現状未使用）LLM reranker の dtype。
            reranker_batch_size: （互換のため残置・現状未使用）LLM reranker の scoring batch size。
            reranker_max_length: （互換のため残置・現状未使用）LLM reranker の最大 token 長。
            reranker_instruction: （互換のため残置・現状未使用）LLM reranker に渡す判定 instruction。
            lm_max_new_tokens: 応答生成時の最大生成 token 数。
            metadata_exact_fields: metadata_exact retrieval で照合するフィールド。
        """
        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.reranker_type = reranker_type
        self.retrieval_topk = retrieval_topk
        self.rerank_topk = rerank_topk
        self.lm_max_new_tokens = int(lm_max_new_tokens)
        self.metadata_exact_fields = metadata_exact_fields
        self.lm = load_lm_module(self.lm_type, self.device, self.attn_implementation, self.dtype)
        retrieval_corpus_types = (
            list(metadata_exact_fields)
            if self.retrieval_type == "metadata_exact" and metadata_exact_fields is not None
            else self.corpus_types
        )
        self.retrieval = load_retrieval_module(
            self.retrieval_type,
            self.item_db_name,
            self.track_split_types,
            retrieval_corpus_types,
            self.cache_dir,
        )
        # 現状サポートする reranker は lightgbm_ltr のみ。LLM reranker 用の
        # device / attn / dtype / batch / max_length / instruction や user_db 系は
        # loader が受け取らないため渡さない（呼び出し側 __init__ は互換のため引数を残す）。
        self.reranker = load_reranker_module(
            self.reranker_type,
            track_split_types=self.track_split_types,
            item_db_name=self.item_db_name,
            corpus_types=self.corpus_types,
            cache_dir=self.cache_dir,
            track_embedding_dataset_name=track_embedding_dataset_name,
            user_embedding_dataset_name=user_embedding_dataset_name,
            track_embedding_name=track_embedding_name,
            audio_embedding_name=audio_embedding_name,
            user_embedding_name=user_embedding_name,
            user_embedding_split_types=user_embedding_split_types,
            reranker_weights=reranker_weights,
            reranker_model_name=reranker_model_name,
        )
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "response_generation": open(f"{self.prompts_dir}/response_generation.txt", "r", encoding="utf-8").read(),
        }
        self.session_memory = []

    def _reset_session_memory(self):
        """現在の単発 chat セッション履歴を空にする。"""
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]):
        """外部で保持している会話履歴を単発 chat 用メモリへ反映する。

        Args:
            chat_history: ``role`` と ``content`` を持つ message dict のリスト。
        """
        self.session_memory = chat_history

    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """ユーザープロファイルを必要に応じて含めた system prompt を作る。

        Args:
            user_id: personalization を行う user ID。None なら汎用 prompt のみ。

        Returns:
            LLM に渡す system prompt 文字列。
        """
        system_prompt = self.role_prompt["response_generation"]
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        return system_prompt

    def _normalize_session_memory(self, session_memory: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """dataset 固有 role を chat template 互換の message に変換する。

        ``music`` role は track_id だけを持つため、応答生成で意味が通るよう
        track metadata 文字列へ展開する。
        """
        normalized = []
        for message in session_memory:
            role = message.get("role")
            content = message.get("content", "")
            if role == "music":
                role = "assistant"
                try:
                    content = self.item_db.id_to_metadata(content)
                except KeyError:
                    content = str(content)
            elif role not in {"user", "assistant", "system"}:
                role = "assistant"
                content = str(content)
            normalized.append({"role": role, "content": str(content)})
        return normalized

    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """1 turn 分の推薦候補取得と応答生成を実行する。

        Args:
            user_query: 最新のユーザ発話。
            user_id: personalization に使う user ID。

        Returns:
            user_id、user_query、retrieval_items、recommend_item、response を含む辞書。
        """
        self.session_memory.append({"role": "user", "content": user_query})
        normalized_session_memory = self._normalize_session_memory(self.session_memory)
        # 以降の retrieval/generation で同じ履歴解釈を使うため、先に role を正規化する。
        system_prompt = self._get_system_prompt(user_id)
        retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in normalized_session_memory])
        candidate_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=self.retrieval_topk)
        retrieval_items = self._rerank_items(
            candidate_items=candidate_items,
            retrieval_input=retrieval_input,
            user_id=user_id,
            session_memory=self.session_memory,
        )
        recommend_item = self.item_db.id_to_metadata(retrieval_items[0])
        # 応答では最終推薦 track の metadata を明示し、推薦内容との矛盾を抑える。
        response = self.lm.response_generation(
            system_prompt,
            normalized_session_memory,
            recommend_item,
            max_new_tokens=self.lm_max_new_tokens,
        )
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """複数 turn の推薦候補取得と応答生成を batch 実行する。

        Args:
            batch_data: user_query、user_id、session_memory を持つ入力辞書のリスト。

        Returns:
            各入力に対応する user_id、user_query、retrieval_items、recommend_item、response のリスト。
        """
        # batch 生成では prompt/retrieval 入力を先に揃え、後段でまとめて tokenization できるようにする。
        sys_prompts = []
        retrieval_inputs = []
        session_memories = []
        raw_session_memories = []
        user_ids = []

        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            raw_session_memory = list(data['session_memory'])
            session_memory = self._normalize_session_memory(raw_session_memory)
            session_memory.append({"role": "user", "content": user_query})

            sys_prompts.append(self._get_system_prompt(user_id))
            retrieval_input = "\n".join([f"{conversation['role']}: {conversation['content']}" for conversation in session_memory])
            retrieval_inputs.append(retrieval_input)
            session_memories.append(session_memory)
            raw_session_memories.append(raw_session_memory)
            user_ids.append(user_id)

        if hasattr(self.retrieval, 'batch_text_to_item_retrieval'):
            batch_candidate_items = self.retrieval.batch_text_to_item_retrieval(retrieval_inputs, topk=self.retrieval_topk)
        else:
            # 古い retrieval module との互換性を保つため、batch API がなければ逐次実行する。
            batch_candidate_items = [self.retrieval.text_to_item_retrieval(inp, topk=self.retrieval_topk) for inp in retrieval_inputs]

        batch_retrieval_items = self._batch_rerank_items(
            batch_candidate_items=batch_candidate_items,
            retrieval_inputs=retrieval_inputs,
            user_ids=user_ids,
            session_memories=raw_session_memories,
        )

        recommend_items = [self.item_db.id_to_metadata(items[0]) for items in batch_retrieval_items]

        if hasattr(self.lm, 'batch_response_generation'):
            responses = self.lm.batch_response_generation(
                sys_prompts,
                session_memories,
                recommend_items,
                max_new_tokens=self.lm_max_new_tokens,
            )
        else:
            # LM module 差し替え時に batch API がない場合でも推論を継続する。
            responses = [self.lm.response_generation(sys_prompts[i], session_memories[i], recommend_items[i])
                        for i in range(len(batch_data))]

        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get('user_id'),
                "user_query": data['user_query'],
                "retrieval_items": batch_retrieval_items[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })

        return results

    def _rerank_items(
        self,
        candidate_items: List[str],
        retrieval_input: str,
        user_id: Optional[str],
        session_memory: List[Dict[str, Any]],
    ) -> List[str]:
        """単一クエリの候補リストを reranker で並べ替える。"""
        if self.reranker is None:
            return candidate_items[:self.rerank_topk]
        return self.reranker.rerank(
            query=retrieval_input,
            candidate_track_ids=candidate_items,
            user_id=user_id,
            session_memory=session_memory,
            topk=self.rerank_topk,
        )

    def _batch_rerank_items(
        self,
        batch_candidate_items: List[List[str]],
        retrieval_inputs: List[str],
        user_ids: List[Optional[str]],
        session_memories: List[List[Dict[str, Any]]],
    ) -> List[List[str]]:
        """batch 候補リストを reranker で並べ替える。"""
        if self.reranker is None:
            return [candidate_items[:self.rerank_topk] for candidate_items in batch_candidate_items]
        if hasattr(self.reranker, "batch_rerank"):
            return self.reranker.batch_rerank(
                batch_candidate_track_ids=batch_candidate_items,
                user_ids=user_ids,
                session_memories=session_memories,
                topk=self.rerank_topk,
                queries=retrieval_inputs,
            )
        return [
            self.reranker.rerank(
                query=retrieval_inputs[index],
                candidate_track_ids=candidate_items,
                user_id=user_ids[index],
                session_memory=session_memories[index],
                topk=self.rerank_topk,
            )
            for index, candidate_items in enumerate(batch_candidate_items)
        ]
