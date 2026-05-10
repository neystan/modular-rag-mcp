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
        evaluation={"provider": "ragas", "answer": "答案", "ground_truth": "标准答案"},
        observability={"log_level": "INFO"},
    )


def test_factory_creates_ragas_evaluator() -> None:
    evaluator = EvaluatorFactory.create(make_settings())

    assert isinstance(evaluator, RagasEvaluator)


def test_observability_module_re_exports_ragas_evaluator() -> None:
    assert ObservabilityRagasEvaluator is RagasEvaluator


def test_ragas_evaluator_returns_metrics_from_injected_runner() -> None:
    captured: dict[str, Any] = {}

    def runner(
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
        metrics: list[str],
    ) -> dict[str, float]:
        captured["query"] = query
        captured["answer"] = answer
        captured["contexts"] = contexts
        captured["ground_truth"] = ground_truth
        captured["metrics"] = metrics
        return {
            "faithfulness": 0.91,
            "answer_relevancy": 0.82,
            "context_precision": 0.73,
        }

    evaluator = RagasEvaluator({"runner": runner, "answer": "生成答案", "ground_truth": "标准答案"})

    metrics = evaluator.evaluate(
        query="什么是 RAG?",
        retrieved_ids=["上下文 A", "上下文 B"],
        golden_ids=["标准答案"],
    )

    assert metrics == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.82,
        "context_precision": 0.73,
    }
    assert captured == {
        "query": "什么是 RAG?",
        "answer": "生成答案",
        "contexts": ["上下文 A", "上下文 B"],
        "ground_truth": "标准答案",
        "metrics": ["faithfulness", "answer_relevancy", "context_precision"],
    }


def test_ragas_evaluator_can_read_answer_from_trace() -> None:
    evaluator = RagasEvaluator(
        {
            "runner": lambda query, answer, contexts, ground_truth, metrics: {
                "faithfulness": 1,
                "answer_relevancy": 0.9,
            }
        }
    )

    metrics = evaluator.evaluate(
        "query",
        ["context"],
        ["ground truth"],
        trace={"generated_answer": "answer from trace"},
    )

    assert metrics == {"faithfulness": 1.0, "answer_relevancy": 0.9}


def test_ragas_evaluator_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(module_name: str) -> Any:
        raise ImportError(module_name)

    monkeypatch.setattr("libs.evaluator.ragas_evaluator.importlib.import_module", fake_import_module)
    evaluator = RagasEvaluator({"answer": "answer", "ground_truth": "ground truth"})

    with pytest.raises(ImportError, match="install ragas and datasets"):
        evaluator.evaluate("query", ["context"], ["ground truth"])


def test_ragas_evaluator_requires_answer() -> None:
    evaluator = RagasEvaluator({"runner": lambda *args: {"faithfulness": 1.0}})

    with pytest.raises(RagasEvaluatorError, match="answer is required"):
        evaluator.evaluate("query", ["context"], ["ground truth"])
