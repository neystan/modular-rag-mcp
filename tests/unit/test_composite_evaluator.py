"""Composite Evaluator 测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.composite_evaluator import CompositeEvaluator, CompositeEvaluatorError
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError
from observability.evaluation.composite_evaluator import CompositeEvaluator as ObservabilityCompositeEvaluator


class StaticEvaluator(BaseEvaluator):
    """测试用固定指标 Evaluator。"""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        return dict(self.config["metrics"])


class EchoEvaluator(BaseEvaluator):
    """验证入参透传的 Evaluator。"""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        assert query == "query"
        assert retrieved_ids == ["doc-a", "doc-b"]
        assert golden_ids == ["doc-b"]
        assert trace == {"answer": "answer"}
        return {"echo": 1.0}


@pytest.fixture(autouse=True)
def clear_evaluator_registry() -> None:
    EvaluatorFactory.clear_providers()


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={
            "backends": [
                {"provider": "static", "metrics": {"faithfulness": 0.91, "answer_relevancy": 0.82}},
                {"provider": "custom"},
            ]
        },
        observability={"log_level": "INFO"},
    )


def test_composite_evaluator_merges_metrics() -> None:
    evaluator = CompositeEvaluator(
        [
            StaticEvaluator({"metrics": {"faithfulness": 0.9, "answer_relevancy": 0.8}}),
            StaticEvaluator({"metrics": {"hit_rate": 1.0, "mrr": 0.5}}),
        ]
    )

    metrics = evaluator.evaluate("query", ["doc-a"], ["doc-a"])

    assert metrics == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "hit_rate": 1.0,
        "mrr": 0.5,
    }


def test_composite_evaluator_passes_inputs_to_children() -> None:
    evaluator = CompositeEvaluator([EchoEvaluator()])

    metrics = evaluator.evaluate("query", ["doc-a", "doc-b"], ["doc-b"], trace={"answer": "answer"})

    assert metrics == {"echo": 1.0}


def test_factory_creates_composite_from_backends() -> None:
    EvaluatorFactory.register_provider("static", StaticEvaluator)

    evaluator = EvaluatorFactory.create(make_settings())

    assert isinstance(evaluator, CompositeEvaluator)
    metrics = evaluator.evaluate("query", ["doc-b"], ["doc-b"])
    assert metrics["faithfulness"] == 0.91
    assert metrics["answer_relevancy"] == 0.82
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0


def test_factory_supports_string_backends() -> None:
    evaluator = EvaluatorFactory.create({"evaluation": {"backends": ["custom"]}})

    assert isinstance(evaluator, CompositeEvaluator)
    assert evaluator.evaluate("query", ["doc-a"], ["doc-a"]) == {"hit_rate": 1.0, "mrr": 1.0}


def test_observability_module_re_exports_composite_evaluator() -> None:
    assert ObservabilityCompositeEvaluator is CompositeEvaluator


def test_composite_rejects_empty_evaluator_list() -> None:
    with pytest.raises(CompositeEvaluatorError, match="non-empty list"):
        CompositeEvaluator([])


def test_factory_rejects_invalid_backends() -> None:
    with pytest.raises(EvaluatorFactoryError, match="evaluation.backends"):
        EvaluatorFactory.create({"evaluation": {"backends": []}})
