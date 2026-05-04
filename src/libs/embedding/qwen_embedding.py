"""Qwen Embedding 实现。"""

from __future__ import annotations

from libs.embedding.openai_embedding import OpenAIEmbedding


class QwenEmbedding(OpenAIEmbedding):
    """Qwen DashScope OpenAI-compatible Embeddings 实现。"""

    provider_name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
