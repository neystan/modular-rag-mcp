"""Dashboard 评估面板测试。"""

from __future__ import annotations

import json
from pathlib import Path

from core.settings import Settings
from observability.dashboard.pages.evaluation_panel import (
    CONFIGURED_BACKEND,
    collect_evaluation_panel_data,
    load_test_set_summary,
    run_dashboard_evaluation,
)
from observability.evaluation.eval_runner import EvalCaseResult, EvalReport


class FakeRunner:
    def __init__(self, report: EvalReport) -> None:
        self.report = report
        self.calls: list[str] = []

    def run(self, test_set_path: str | Path) -> EvalReport:
        self.calls.append(str(test_set_path))
        return self.report


def make_settings(provider: str = "custom") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": provider, "enabled": True},
        observability={"log_level": "INFO"},
    )


def write_test_set(path: Path) -> None:
    path.write_text(
        json.dumps({"test_cases": [{"query": "q1", "expected_chunk_ids": ["chunk-a"]}]}),
        encoding="utf-8",
    )


def test_load_test_set_summary_reads_queries(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    write_test_set(test_set_path)

    summary = load_test_set_summary(test_set_path)

    assert summary["exists"] is True
    assert summary["case_count"] == 1
    assert summary["queries"] == ["q1"]
    assert summary["error"] is None


def test_collect_evaluation_panel_data_includes_config_and_test_set(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    write_test_set(test_set_path)

    payload = collect_evaluation_panel_data(
        settings_path=tmp_path / "settings.yaml",
        test_set_path=test_set_path,
        settings_loader=lambda _: make_settings(),
    )

    assert payload["evaluation"]["provider"] == "custom"
    assert payload["retrieval"]["top_k"] == 5
    assert payload["test_set"]["case_count"] == 1
    assert CONFIGURED_BACKEND in payload["backend_options"]


def test_run_dashboard_evaluation_can_override_backend(tmp_path: Path) -> None:
    test_set_path = tmp_path / "golden.json"
    write_test_set(test_set_path)
    report = EvalReport(
        total_cases=1,
        hit_rate=1.0,
        mrr=1.0,
        metrics={"hit_rate": 1.0, "mrr": 1.0},
        results=[
            EvalCaseResult(
                query="q1",
                retrieved_ids=["chunk-a"],
                expected_chunk_ids=["chunk-a"],
                expected_sources=[],
                metrics={"hit_rate": 1.0, "mrr": 1.0},
            )
        ],
    )
    fake_runner = FakeRunner(report)
    captured_provider: list[str] = []

    result = run_dashboard_evaluation(
        settings_path=tmp_path / "settings.yaml",
        test_set_path=test_set_path,
        backend="ragas",
        settings_loader=lambda _: make_settings(),
        runner_factory=lambda settings: captured_provider.append(settings.evaluation["provider"]) or fake_runner,
    )

    assert result.hit_rate == 1.0
    assert fake_runner.calls == [str(test_set_path)]
    assert captured_provider == ["ragas"]
