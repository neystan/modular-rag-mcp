"""Reranker 工厂。"""

from __future__ import annotations

from typing import Any, TypeVar

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactoryError(ValueError):
    """Reranker 工厂配置错误。"""


RerankerType = TypeVar("RerankerType", bound=BaseReranker)


class RerankerFactory:
    """按配置创建 Reranker Provider。"""

    _providers: dict[str, type[BaseReranker]] = {"none": NoneReranker}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[RerankerType]) -> None:
        """注册一个 Reranker Provider 实现。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise RerankerFactoryError("Reranker provider 名称不能为空")
        if not issubclass(provider_cls, BaseReranker):
            raise RerankerFactoryError("Reranker provider 必须继承 BaseReranker")
        cls._providers[normalized_name] = provider_cls

    @classmethod
    def create(cls, settings: Settings | dict[str, Any]) -> BaseReranker:
        """根据 Settings 或 dict 配置创建 Reranker 实例。"""

        rerank_config = cls._extract_rerank_config(settings)
        provider = rerank_config.get("provider", "none")
        if not provider:
            raise RerankerFactoryError("缺少配置项: rerank.provider")

        provider_name = str(provider).strip().lower()
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise RerankerFactoryError(
                f"未知 Reranker provider: {provider_name}; available providers: {available}"
            )

        return provider_cls(rerank_config)

    @classmethod
    def clear_providers(cls) -> None:
        """重置 Provider 注册表，保留 none 回退实现。"""

        cls._providers = {"none": NoneReranker}

    @staticmethod
    def _extract_rerank_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.rerank)
        if not isinstance(settings, dict):
            raise RerankerFactoryError("settings 必须是 Settings 或 dict")

        rerank_config = settings.get("rerank", settings)
        if not isinstance(rerank_config, dict):
            raise RerankerFactoryError("配置项必须是 mapping/object: rerank")
        return dict(rerank_config)
