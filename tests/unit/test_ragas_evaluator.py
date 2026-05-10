"""Ragas Evaluator 测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.evaluator.ragas_evaluator import RagasEvaluator, RagasEvaluatorError
from observability.evaluation.ragas_evaluator import RagasEvaluator as ObservabilityRagasEvaluator


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "ragas", "answer": "答案", "reference": "标准答案"},
        observability={"log_level": "INFO"},
    )


def test_factory_creates_ragas_evaluator() -> None:
    evaluator = EvaluatorFactory.create(make_settings())

    assert isinstance(evaluator, RagasEvaluator)
    assert evaluator.config["llm_config"]["provider"] == "placeholder"
    assert evaluator.config["embedding_config"]["provider"] == "placeholder"


def test_observability_module_re_exports_ragas_evaluator() -> None:
    assert ObservabilityRagasEvaluator is RagasEvaluator


def test_ragas_evaluator_returns_metrics_from_injected_runner() -> None:
    captured: dict[str, Any] = {}

    def runner(
        query: str,
        answer: str,
        contexts: list[str],
        reference: str,
        metrics: list[str],
    ) -> dict[str, float]:
        captured["query"] = query
        captured["answer"] = answer
        captured["contexts"] = contexts
        captured["reference"] = reference
        captured["metrics"] = metrics
        return {
            "faithfulness": 0.91,
            "answer_relevancy": 0.82,
            "context_precision": 0.73,
            "context_recall": 0.64,
        }

    evaluator = RagasEvaluator({"runner": runner, "answer": "生成答案", "reference": "标准答案"})

    metrics = evaluator.evaluate(
        query="什么是 RAG?",
        retrieved_ids=["上下文 A", "上下文 B"],
        golden_ids=["标准答案"],
    )

    assert metrics == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.82,
        "context_precision": 0.73,
        "context_recall": 0.64,
    }
    assert captured == {
        "query": "什么是 RAG?",
        "answer": "生成答案",
        "contexts": ["上下文 A", "上下文 B"],
        "reference": "标准答案",
        "metrics": ["context_precision", "context_recall", "faithfulness", "answer_relevancy"],
    }


def test_ragas_evaluator_can_read_answer_from_trace() -> None:
    captured: dict[str, Any] = {}

    def runner(query: str, answer: str, contexts: list[str], reference: str, metrics: list[str]) -> dict[str, float]:
        captured["answer"] = answer
        captured["contexts"] = contexts
        captured["reference"] = reference
        return {"faithfulness": 1, "answer_relevancy": 0.9}

    evaluator = RagasEvaluator(
        {
            "runner": runner
        }
    )

    metrics = evaluator.evaluate(
        "query",
        ["context"],
        [],
        trace={"answer": "answer from trace", "contexts": ["context from trace"], "reference": "reference from trace"},
    )

    assert metrics == {"faithfulness": 1.0, "answer_relevancy": 0.9}
    assert captured == {
        "answer": "answer from trace",
        "contexts": ["context from trace"],
        "reference": "reference from trace",
    }


def test_ragas_evaluator_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(module_name: str) -> Any:
        raise ImportError(module_name)

    monkeypatch.setattr("libs.evaluator.ragas_evaluator.importlib.import_module", fake_import_module)
    evaluator = RagasEvaluator({"answer": "answer", "reference": "reference"})

    with pytest.raises(ImportError, match="install ragas and datasets"):
        evaluator.evaluate("query", ["context"], ["ground truth"])


def test_ragas_evaluator_requires_answer() -> None:
    evaluator = RagasEvaluator({"runner": lambda *args: {"faithfulness": 1.0}})

    with pytest.raises(RagasEvaluatorError, match="answer is required"):
        evaluator.evaluate("query", ["context"], ["ground truth"])


def test_ragas_evaluator_requires_reference() -> None:
    evaluator = RagasEvaluator({"runner": lambda *args: {"faithfulness": 1.0}})

    with pytest.raises(RagasEvaluatorError, match="reference is required"):
        evaluator.evaluate("query", ["context"], [], trace={"answer": "answer"})


def test_ragas_evaluator_ignores_non_metric_fields_in_response() -> None:
    evaluator = RagasEvaluator(
        {
            "runner": lambda *args: {
                "user_input": "question",
                "retrieved_contexts": ["context"],
                "context_precision": 0.75,
                "context_recall": 0.66,
                "faithfulness": 0.91,
                "answer_relevancy": 0.88,
            }
        }
    )

    metrics = evaluator.evaluate(
        "query",
        ["context"],
        [],
        trace={"answer": "answer", "reference": "reference", "contexts": ["context"]},
    )

    assert metrics == {
        "context_precision": 0.75,
        "context_recall": 0.66,
        "faithfulness": 0.91,
        "answer_relevancy": 0.88,
    }
