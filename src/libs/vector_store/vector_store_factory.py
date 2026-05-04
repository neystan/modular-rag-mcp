"""VectorStore 工厂。"""

from __future__ import annotations

from typing import Any, TypeVar

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactoryError(ValueError):
    """VectorStore 工厂配置错误。"""


VectorStoreType = TypeVar("VectorStoreType", bound=BaseVectorStore)


class VectorStoreFactory:
    """按配置创建 VectorStore Provider。"""

    _providers: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[VectorStoreType]) -> None:
        """注册一个 VectorStore Provider 实现。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise VectorStoreFactoryError("VectorStore provider 名称不能为空")
        if not issubclass(provider_cls, BaseVectorStore):
            raise VectorStoreFactoryError("VectorStore provider 必须继承 BaseVectorStore")
        cls._providers[normalized_name] = provider_cls

    @classmethod
    def create(cls, settings: Settings | dict[str, Any]) -> BaseVectorStore:
        """根据 Settings 或 dict 配置创建 VectorStore 实例。"""

        vector_store_config = cls._extract_vector_store_config(settings)
        provider = vector_store_config.get("provider")
        if not provider:
            raise VectorStoreFactoryError("缺少配置项: vector_store.provider")

        provider_name = str(provider).strip().lower()
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise VectorStoreFactoryError(
                f"未知 VectorStore provider: {provider_name}; available providers: {available}"
            )

        return provider_cls(vector_store_config)

    @classmethod
    def clear_providers(cls) -> None:
        """清空 Provider 注册表，主要用于测试隔离。"""

        cls._providers.clear()

    @staticmethod
    def _extract_vector_store_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.vector_store)
        if not isinstance(settings, dict):
            raise VectorStoreFactoryError("settings 必须是 Settings 或 dict")

        vector_store_config = settings.get("vector_store", settings)
        if not isinstance(vector_store_config, dict):
            raise VectorStoreFactoryError("配置项必须是 mapping/object: vector_store")
        return dict(vector_store_config)
