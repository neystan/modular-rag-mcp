"""Qwen LLM 实现。"""

from __future__ import annotations

import os

from libs.llm.openai_llm import OpenAICompatibleLLM


class QwenLLM(OpenAICompatibleLLM):
    """Qwen DashScope OpenAI-compatible API 实现。"""

    provider_name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, config: dict[str, object] | None = None) -> None:
        resolved_config = dict(config or {})
        if not resolved_config.get("api_key"):
            env_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("QWEN_API_KEY", "").strip()
            if env_api_key:
                resolved_config["api_key"] = env_api_key
        super().__init__(resolved_config)
