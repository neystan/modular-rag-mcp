"""Splitter 工厂。"""

from __future__ import annotations

from typing import Any, TypeVar

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter


class SplitterFactoryError(ValueError):
    """Splitter 工厂配置错误。"""


SplitterType = TypeVar("SplitterType", bound=BaseSplitter)


class SplitterFactory:
    """按配置创建 Splitter Provider。"""

    _providers: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[SplitterType]) -> None:
        """注册一个 Splitter Provider 实现。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise SplitterFactoryError("Splitter provider 名称不能为空")
        if not issubclass(provider_cls, BaseSplitter):
            raise SplitterFactoryError("Splitter provider 必须继承 BaseSplitter")
        cls._providers[normalized_name] = provider_cls

    @classmethod
    def create(cls, settings: Settings | dict[str, Any]) -> BaseSplitter:
        """根据 Settings 或 dict 配置创建 Splitter 实例。"""

        splitter_config = cls._extract_splitter_config(settings)
        provider = splitter_config.get("provider")
        if not provider:
            raise SplitterFactoryError("缺少配置项: splitter.provider")

        provider_name = str(provider).strip().lower()
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise SplitterFactoryError(
                f"未知 Splitter provider: {provider_name}; available providers: {available}"
            )

        return provider_cls(splitter_config)

    @classmethod
    def clear_providers(cls) -> None:
        """清空 Provider 注册表，主要用于测试隔离。"""

        cls._providers.clear()

    @staticmethod
    def _extract_splitter_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.splitter)
        if not isinstance(settings, dict):
            raise SplitterFactoryError("settings 必须是 Settings 或 dict")

        splitter_config = settings.get("splitter", settings)
        if not isinstance(splitter_config, dict):
            raise SplitterFactoryError("配置项必须是 mapping/object: splitter")
        return dict(splitter_config)
