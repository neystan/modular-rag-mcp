"""Evaluator 工厂。"""

from __future__ import annotations

from typing import Any, TypeVar

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.composite_evaluator import CompositeEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.ragas_evaluator import RagasEvaluator


class EvaluatorFactoryError(ValueError):
    """Evaluator 工厂配置错误。"""


EvaluatorType = TypeVar("EvaluatorType", bound=BaseEvaluator)


class EvaluatorFactory:
    """按配置创建 Evaluator Provider。"""

    _providers: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
        "ragas": RagasEvaluator,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[EvaluatorType]) -> None:
        """注册一个 Evaluator Provider 实现。"""

        normalized_name = name.strip().lower()
        if not normalized_name:
            raise EvaluatorFactoryError("Evaluator provider 名称不能为空")
        if not issubclass(provider_cls, BaseEvaluator):
            raise EvaluatorFactoryError("Evaluator provider 必须继承 BaseEvaluator")
        cls._providers[normalized_name] = provider_cls

    @classmethod
    def create(cls, settings: Settings | dict[str, Any]) -> BaseEvaluator:
        """根据 Settings 或 dict 配置创建 Evaluator 实例。"""

        evaluation_config = cls._extract_evaluation_config(settings)
        backends = evaluation_config.get("backends")
        if backends is not None:
            return CompositeEvaluator(cls._create_backends(backends, evaluation_config))

        provider = evaluation_config.get("provider", "custom")
        if not provider:
            raise EvaluatorFactoryError("缺少配置项: evaluation.provider")

        return cls._create_provider(str(provider), evaluation_config)

    @classmethod
    def _create_provider(cls, provider: str, evaluation_config: dict[str, Any]) -> BaseEvaluator:
        provider_name = str(provider).strip().lower()
        if not provider_name:
            raise EvaluatorFactoryError("缺少配置项: evaluation.provider")
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise EvaluatorFactoryError(
                f"未知 Evaluator provider: {provider_name}; available providers: {available}"
            )

        return provider_cls(evaluation_config)

    @classmethod
    def _create_backends(cls, backends: Any, evaluation_config: dict[str, Any]) -> list[BaseEvaluator]:
        if not isinstance(backends, list) or not backends:
            raise EvaluatorFactoryError("配置项 evaluation.backends 必须是非空 list")

        evaluators: list[BaseEvaluator] = []
        for index, backend in enumerate(backends):
            backend_config = cls._normalize_backend_config(backend, evaluation_config, index)
            provider = backend_config.get("provider")
            evaluators.append(cls._create_provider(str(provider), backend_config))
        return evaluators

    @staticmethod
    def _normalize_backend_config(
        backend: Any,
        evaluation_config: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        base_config = {key: value for key, value in evaluation_config.items() if key != "backends"}
        if isinstance(backend, str):
            provider = backend.strip()
            if not provider:
                raise EvaluatorFactoryError(f"配置项 evaluation.backends[{index}] provider 不能为空")
            return {**base_config, "provider": provider}
        if isinstance(backend, dict):
            provider = backend.get("provider")
            if not provider:
                raise EvaluatorFactoryError(f"缺少配置项: evaluation.backends[{index}].provider")
            return {**base_config, **backend}
        raise EvaluatorFactoryError(f"配置项 evaluation.backends[{index}] 必须是 string 或 mapping/object")

    @classmethod
    def clear_providers(cls) -> None:
        """重置 Provider 注册表，保留 custom 默认实现。"""

        cls._providers = {
            "custom": CustomEvaluator,
            "ragas": RagasEvaluator,
        }

    @staticmethod
    def _extract_evaluation_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            return dict(settings.evaluation)
        if not isinstance(settings, dict):
            raise EvaluatorFactoryError("settings 必须是 Settings 或 dict")

        evaluation_config = settings.get("evaluation", settings)
        if not isinstance(evaluation_config, dict):
            raise EvaluatorFactoryError("配置项必须是 mapping/object: evaluation")
        return dict(evaluation_config)
