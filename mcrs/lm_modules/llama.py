"""Transformers causal LM を Music-CRS 応答生成 API に合わせて扱う。"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLAMA_MODEL:
    """chat template 対応 causal LM による応答生成 wrapper。"""

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda", attn_implementation="eager", dtype=torch.bfloat16):
        """モデルと tokenizer を読み込み、推論用に初期化する。

        Args:
            model_name: Hugging Face causal LM のモデル名。
            device: 推論に使う torch device。
            attn_implementation: Transformers に渡す attention 実装名。
            dtype: モデル重みの dtype。
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.lm, self.tokenizer = self._load_model()
        self.lm.eval()
        self.lm.to(self.device).to(self.dtype)

    def _load_model(self):
        """Hugging Face Hub/local cache から tokenizer と causal LM を読み込む。"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(self.model_name, attn_implementation=self.attn_implementation, dtype=self.dtype)
        return lm, tokenizer

    def _format_chat_history(self, sys_prompt, chat_history: list, recommend_item: str):
        """system prompt、会話履歴、推薦 item を chat template 文字列へ整形する。"""
        response_contract = (
            "\n\nReturn a concise user-facing answer. "
            "The answer must start with FINAL_RESPONSE:. "
            "Only the text after FINAL_RESPONSE: will be shown to the user."
        )
        chat_data = [{"role": "system", "content": sys_prompt + response_contract}]
        chat_data += [
            {
                "role": message["role"] if message["role"] in {"user", "assistant", "system"} else "assistant",
                "content": str(message["content"]),
            }
            for message in chat_history
        ]
        chat_data += [{"role": "assistant", "content": recommend_item}]
        chat_template_kwargs = {}
        if self.model_name.startswith("Qwen/"):
            chat_template_kwargs["enable_thinking"] = False
        chat_template = self.tokenizer.apply_chat_template(
            chat_data,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        return chat_template

    def _clean_response(self, response: str) -> str:
        """生成テキストから thinking block、marker、特殊 token を取り除く。"""
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
        response = re.sub(r"^.*?</think>\s*", "", response, flags=re.DOTALL)
        if "FINAL_RESPONSE:" in response:
            response = response.rsplit("FINAL_RESPONSE:", 1)[-1]
        for special_token in self.tokenizer.all_special_tokens:
            response = response.replace(special_token, "")
        return response.strip()

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str,max_new_tokens=512, response_format=None):
        """単一入力に対する推薦応答を生成する。

        Args:
            sys_prompt: system prompt。
            chat_history: chat template に渡す会話履歴。
            recommend_item: LLM に提示する推薦 track metadata。
            max_new_tokens: 最大生成 token 数。
            response_format: 互換性のための未使用引数。

        Returns:
            表示用に整形済みの応答文字列。
        """
        chat_history = self._format_chat_history(sys_prompt, chat_history, recommend_item)
        token_inputs = self.tokenizer(chat_history, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.lm.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=False)[0]
        return self._clean_response(generated_text)

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=1024):
        """複数入力に対する推薦応答を batch 生成する。

        Args:
            sys_prompts: system prompt のリスト。
            chat_histories: 会話履歴リストのリスト。
            recommend_items: 推薦 track metadata のリスト。
            max_new_tokens: 最大生成 token 数。

        Returns:
            表示用に整形済みの応答文字列リスト。
        """
        # 先に chat template を適用して、padding 付き batch tokenization に渡す。
        formatted_chats = [
            self._format_chat_history(sys_prompt, chat_history, recommend_item)
            for sys_prompt, chat_history, recommend_item in zip(sys_prompts, chat_histories, recommend_items)
        ]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        token_inputs = self.tokenizer(formatted_chats, return_tensors="pt", padding=True, truncation=True)
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # prompt 部分を除き、新規生成された token だけを decode する。
        generated_texts = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=False)
        return [self._clean_response(text) for text in generated_texts]
