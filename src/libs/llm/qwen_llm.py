"""Qwen LLM 实现。"""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class QwenLLM(OpenAICompatibleLLM):
    """Qwen DashScope OpenAI-compatible API 实现。"""

    provider_name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
