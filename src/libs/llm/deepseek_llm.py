"""DeepSeek LLM 实现。"""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek OpenAI-compatible API 实现。"""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
