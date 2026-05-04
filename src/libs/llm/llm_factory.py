"""LLM 工厂。"""

from __future__ import annotations

from typing import Any, TypeVar

from core.settings import Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.openai_llm import OpenAILLM


class LLMFactoryError(ValueError):
    """LLM 工厂配置错误。"""


LLMType = TypeVar("LLMType", bound=BaseLLM)


class LLMFactory:
    """按配置创建 LLM Provider。"""

    _default_providers: dict[str, type[BaseLLM]] = {
        "openai": OpenAILLM,
        "azure": AzureLLM,
        "deepseek": DeepSeekLLM,
    }
    _providers: dict[str, type[BaseLLM]] = dict(_default_providers)

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[LLMType]) -> None:
        """注册一个 LLM Provider 实现。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise LLMFactoryError("LLM provider 名称不能为空")
        if not issubclass(provider_cls, BaseLLM):
            raise LLMFactoryError("LLM provider 必须继承 BaseLLM")
        cls._providers[normalized_name] = provider_cls

    @classmethod
    def create(cls, settings: Settings | dict[str, Any]) -> BaseLLM:
        """根据 Settings 或 dict 配置创建 LLM 实例。"""

        llm_config = cls._extract_llm_config(settings)
        provider = llm_config.get("provider")
        if not provider:
            raise LLMFactoryError("缺少配置项: llm.provider")

        provider_name = str(provider).strip().lower()
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise LLMFactoryError(
                f"未知 LLM provider: {provider_name}; available providers: {available}"
            )

        return provider_cls(llm_config)

    @classmethod
    def clear_providers(cls) -> None:
        """重置 Provider 注册表，保留默认实现。"""

        cls._providers = dict(cls._default_providers)

    @staticmethod
    def _extract_llm_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.llm)
        if not isinstance(settings, dict):
            raise LLMFactoryError("settings 必须是 Settings 或 dict")

        llm_config = settings.get("llm", settings)
        if not isinstance(llm_config, dict):
            raise LLMFactoryError("配置项必须是 mapping/object: llm")
        return dict(llm_config)
