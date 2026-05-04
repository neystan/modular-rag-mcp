"""Embedding 工厂。"""

from __future__ import annotations

from typing import Any, TypeVar

from core.settings import Settings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import OpenAIEmbedding
from libs.embedding.qwen_embedding import QwenEmbedding


class EmbeddingFactoryError(ValueError):
    """Embedding 工厂配置错误。"""


EmbeddingType = TypeVar("EmbeddingType", bound=BaseEmbedding)


class EmbeddingFactory:
    """按配置创建 Embedding Provider。"""

    _default_providers: dict[str, type[BaseEmbedding]] = {
        "openai": OpenAIEmbedding,
        "azure": AzureEmbedding,
        "qwen": QwenEmbedding,
    }
    _providers: dict[str, type[BaseEmbedding]] = dict(_default_providers)

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[EmbeddingType]) -> None:
        """注册一个 Embedding Provider 实现。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise EmbeddingFactoryError("Embedding provider 名称不能为空")
        if not issubclass(provider_cls, BaseEmbedding):
            raise EmbeddingFactoryError("Embedding provider 必须继承 BaseEmbedding")
        cls._providers[normalized_name] = provider_cls

    @classmethod
    def create(cls, settings: Settings | dict[str, Any]) -> BaseEmbedding:
        """根据 Settings 或 dict 配置创建 Embedding 实例。"""

        embedding_config = cls._extract_embedding_config(settings)
        provider = embedding_config.get("provider")
        if not provider:
            raise EmbeddingFactoryError("缺少配置项: embedding.provider")

        provider_name = str(provider).strip().lower()
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise EmbeddingFactoryError(
                f"未知 Embedding provider: {provider_name}; available providers: {available}"
            )

        return provider_cls(embedding_config)

    @classmethod
    def clear_providers(cls) -> None:
        """重置 Provider 注册表，保留默认实现。"""

        cls._providers = dict(cls._default_providers)

    @staticmethod
    def _extract_embedding_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.embedding)
        if not isinstance(settings, dict):
            raise EmbeddingFactoryError("settings 必须是 Settings 或 dict")

        embedding_config = settings.get("embedding", settings)
        if not isinstance(embedding_config, dict):
            raise EmbeddingFactoryError("配置项必须是 mapping/object: embedding")
        return dict(embedding_config)
