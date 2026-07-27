"""応答生成 module の loader。"""

from .llama import LLAMA_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    """モデル名から対応する LM wrapper を初期化する。

    Args:
        lm_type: Hugging Face モデル名。
        device: モデルを配置する torch device。
        attn_implementation: Transformers に渡す attention 実装名。
        dtype: モデル重みの dtype。

    Returns:
        response_generation API を持つ LM wrapper。

    Raises:
        ValueError: 未対応のモデル prefix が指定された場合。
    """
    supported_prefixes = (
        "meta-llama/",
        "Qwen/",
        "HuggingFaceTB/",
    )
    if lm_type.startswith(supported_prefixes):
        return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    else:
        raise ValueError(f"Unsupported LM type: {lm_type}")
