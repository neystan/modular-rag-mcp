"""EvalRunner 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings
from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.llm.base_llm import BaseLLM
from observability.evaluation.eval_runner import EvalReport, EvalRunner, EvalRunnerError


class FakeHybridSearch:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k})
        return list(self.results_by_query.get(query, []))


class RecordingEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__({})
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        self.calls.append({"query": query, "retrieved_ids": retrieved_ids, "golden_ids": golden_ids, "trace": trace})
        first_hit_rank = next(
            (index for index, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in set(golden_ids)),
            None,
        )
        if first_hit_rank is None:
            return {"hit_rate": 0.0, "mrr": 0.0}
        return {"hit_rate": 1.0, "mrr": 1.0 / float(first_hit_rank)}


class FakeLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__({})
        self.messages: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[Any]) -> str:
        self.messages.append(messages)
        return "generated answer"


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 2},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_result(chunk_id: str, source_path: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        text=f"text for {chunk_id}",
        metadata={"source_path": source_path},
    )


def write_test_set(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "question": "q1",
                        "reference": "reference 1",
                        "expected_chunk_ids": ["chunk-b"],
                        "expected_sources": ["manual.pdf"],
                    },
                    {
                        "question": "q2",
                        "reference": "reference 2",
                        "expected_chunk_ids": ["chunk-z"],
                        "expected_sources": ["missing.pdf"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_eval_runner_runs_test_set_and_averages_metrics(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    write_test_set(test_set_path)
    search = FakeHybridSearch(
        {
            "q1": [make_result("chunk-a", "docs/a.pdf"), make_result("chunk-b", "docs/manual.pdf")],
            "q2": [make_result("chunk-c", "docs/c.pdf")],
        }
    )
    evaluator = RecordingEvaluator()
    runner = EvalRunner(make_settings(), search, evaluator)  # type: ignore[arg-type]

    report = runner.run(test_set_path)

    assert isinstance(report, EvalReport)
    assert report.total_cases == 2
    assert report.hit_rate == 0.5
    assert report.mrr == 0.25
    assert report.metrics == {"hit_rate": 0.5, "mrr": 0.25}
    assert [item.query for item in report.results] == ["q1", "q2"]
    assert search.calls == [{"query": "q1", "top_k": 2}, {"query": "q2", "top_k": 2}]
    assert evaluator.calls[0] == {
        "query": "q1",
        "retrieved_ids": ["chunk-a", "chunk-b"],
        "golden_ids": ["chunk-b"],
        "trace": {
            "answer": "text for chunk-a\n\ntext for chunk-b",
            "contexts": ["text for chunk-a", "text for chunk-b"],
            "reference": "reference 1",
        },
    }


def test_eval_runner_can_use_expected_sources_when_chunk_ids_are_absent(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    test_set_path.write_text(
        json.dumps({"test_cases": [{"question": "q", "reference": "reference", "expected_sources": ["manual.pdf"]}]}),
        encoding="utf-8",
    )
    search = FakeHybridSearch({"q": [make_result("chunk-a", "docs/manual.pdf")]})
    evaluator = RecordingEvaluator()
    runner = EvalRunner(make_settings(), search, evaluator)  # type: ignore[arg-type]

    report = runner.run(test_set_path)

    assert report.hit_rate == 1.0
    assert evaluator.calls[0]["golden_ids"] == ["chunk-a"]


def test_eval_runner_uses_llm_to_generate_answer_for_evaluators(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    test_set_path.write_text(
        json.dumps({"test_cases": [{"question": "q", "reference": "reference", "expected_chunk_ids": ["chunk-a"]}]}),
        encoding="utf-8",
    )
    search = FakeHybridSearch({"q": [make_result("chunk-a", "docs/manual.pdf")]})
    evaluator = RecordingEvaluator()
    llm = FakeLLM()
    runner = EvalRunner(make_settings(), search, evaluator, llm=llm)  # type: ignore[arg-type]

    report = runner.run(test_set_path)

    assert report.results[0].answer == "generated answer"
    assert evaluator.calls[0]["trace"] == {
        "answer": "generated answer",
        "contexts": ["text for chunk-a"],
        "reference": "reference",
    }
    assert llm.messages[0][1]["content"].startswith("问题：q")


def test_eval_runner_can_use_precomputed_ragas_sample_without_search(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    test_set_path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "question": "q",
                        "answer": "precomputed answer",
                        "contexts": ["precomputed context"],
                        "reference": "reference",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    search = FakeHybridSearch({})
    evaluator = RecordingEvaluator()
    runner = EvalRunner(make_settings(), search, evaluator)  # type: ignore[arg-type]

    report = runner.run(test_set_path)

    assert search.calls == []
    assert report.results[0].answer == "precomputed answer"
    assert report.results[0].contexts == ["precomputed context"]
    assert evaluator.calls[0]["trace"] == {
        "answer": "precomputed answer",
        "contexts": ["precomputed context"],
        "reference": "reference",
    }


def test_eval_runner_rejects_invalid_test_set(tmp_path: Path) -> None:
    test_set_path = tmp_path / "invalid.json"
    test_set_path.write_text(json.dumps({"test_cases": []}), encoding="utf-8")
    runner = EvalRunner(make_settings(), FakeHybridSearch({}), RecordingEvaluator())  # type: ignore[arg-type]

    with pytest.raises(EvalRunnerError, match="test_cases"):
        runner.run(test_set_path)


def test_eval_runner_reports_progress_per_case(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    write_test_set(test_set_path)
    search = FakeHybridSearch(
        {
            "q1": [make_result("chunk-a", "docs/a.pdf")],
            "q2": [make_result("chunk-b", "docs/b.pdf")],
        }
    )
    evaluator = RecordingEvaluator()
    runner = EvalRunner(make_settings(), search, evaluator)  # type: ignore[arg-type]
    updates: list[tuple[int, int, str]] = []

    runner.run_with_progress(test_set_path, progress_callback=lambda c, t, m: updates.append((c, t, m)))

    assert updates[0] == (0, 2, "准备开始评估")
    assert updates[1] == (0, 2, "评估中：q1")
    assert updates[2] == (1, 2, "已完成：q1")
    assert updates[3] == (1, 2, "评估中：q2")
    assert updates[4] == (2, 2, "已完成：q2")
