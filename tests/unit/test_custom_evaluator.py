"""自定义 Evaluator 与工厂测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError


class ConstantEvaluator(BaseEvaluator):
    """测试用 Evaluator Provider。"""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        return {"hit_rate": 0.5, "mrr": 0.25}


class NotEvaluator:
    pass


@pytest.fixture(autouse=True)
def clear_evaluator_registry() -> None:
    EvaluatorFactory.clear_providers()


def make_settings(provider: str = "custom") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": provider},
        observability={"log_level": "INFO"},
    )


def test_custom_evaluator_returns_hit_and_mrr_for_first_rank_hit() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate(
        query="测试查询",
        retrieved_ids=["doc-a", "doc-b", "doc-c"],
        golden_ids=["doc-b"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


def test_custom_evaluator_returns_full_mrr_for_top_hit() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate("query", ["doc-a", "doc-b"], ["doc-a"])

    assert metrics == {"hit_rate": 1.0, "mrr": 1.0}


def test_custom_evaluator_returns_zero_without_hit() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate("query", ["doc-a"], ["doc-z"])

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_custom_evaluator_handles_empty_inputs() -> None:
    evaluator = CustomEvaluator()

    assert evaluator.evaluate("query", [], ["doc-a"]) == {"hit_rate": 0.0, "mrr": 0.0}
    assert evaluator.evaluate("query", ["doc-a"], []) == {"hit_rate": 0.0, "mrr": 0.0}


def test_factory_creates_custom_evaluator_by_default() -> None:
    evaluator = EvaluatorFactory.create(make_settings())

    assert isinstance(evaluator, CustomEvaluator)


def test_register_provider_and_create_from_settings() -> None:
    EvaluatorFactory.register_provider("constant", ConstantEvaluator)

    evaluator = EvaluatorFactory.create(make_settings(provider="constant"))

    assert isinstance(evaluator, ConstantEvaluator)
    assert evaluator.evaluate("query", [], []) == {"hit_rate": 0.5, "mrr": 0.25}


def test_create_from_dict_uses_evaluation_section() -> None:
    EvaluatorFactory.register_provider("constant", ConstantEvaluator)

    evaluator = EvaluatorFactory.create({"evaluation": {"provider": "constant"}})

    assert isinstance(evaluator, ConstantEvaluator)


def test_provider_name_is_case_insensitive() -> None:
    EvaluatorFactory.register_provider("Constant", ConstantEvaluator)

    evaluator = EvaluatorFactory.create(make_settings(provider="CONSTANT"))

    assert isinstance(evaluator, ConstantEvaluator)


def test_unknown_provider_reports_available_providers() -> None:
    with pytest.raises(EvaluatorFactoryError, match="未知 Evaluator provider: missing"):
        EvaluatorFactory.create(make_settings(provider="missing"))


def test_missing_provider_reports_config_path() -> None:
    with pytest.raises(EvaluatorFactoryError, match="evaluation.provider"):
        EvaluatorFactory.create({"evaluation": {"provider": ""}})


def test_register_provider_requires_baseevaluator_subclass() -> None:
    with pytest.raises(EvaluatorFactoryError, match="必须继承 BaseEvaluator"):
        EvaluatorFactory.register_provider("bad", NotEvaluator)  # type: ignore[arg-type]
