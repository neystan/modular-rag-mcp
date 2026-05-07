"""TraceContext 单元测试。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext
import core.trace.trace_context as trace_context_module


def test_trace_context_serializes_finished_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_values = iter([10.0, 10.25, 10.75])
    timestamps = iter(
        [
            datetime.fromisoformat("2026-05-07T10:00:00+00:00"),
            datetime.fromisoformat("2026-05-07T10:00:00.250000+00:00"),
            datetime.fromisoformat("2026-05-07T10:00:00.750000+00:00"),
        ]
    )
    monkeypatch.setattr(trace_context_module, "_perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(trace_context_module, "_utcnow", lambda: next(timestamps))

    trace = TraceContext(trace_type="ingestion", trace_id="trace-1")
    trace.record_stage("loader.load", {"status": "ok"})
    trace.finish()

    payload = trace.to_dict()

    assert payload["trace_id"] == "trace-1"
    assert payload["trace_type"] == "ingestion"
    assert payload["started_at"] == "2026-05-07T10:00:00+00:00"
    assert payload["finished_at"] == "2026-05-07T10:00:00.750000+00:00"
    assert payload["total_elapsed_ms"] == 750.0
    assert payload["stages"][0]["stage"] == "loader.load"
    assert payload["stages"][0]["payload"] == {"status": "ok"}
    assert payload["stages"][0]["elapsed_ms"] == 250.0
    assert trace.elapsed_ms("loader.load") == 250.0
    assert json.loads(json.dumps(payload))["trace_type"] == "ingestion"


def test_to_dict_finishes_trace_implicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_values = iter([1.0, 1.2, 1.6])
    timestamps = iter(
        [
            datetime.fromisoformat("2026-05-07T11:00:00+00:00"),
            datetime.fromisoformat("2026-05-07T11:00:00.200000+00:00"),
            datetime.fromisoformat("2026-05-07T11:00:00.600000+00:00"),
        ]
    )
    monkeypatch.setattr(trace_context_module, "_perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(trace_context_module, "_utcnow", lambda: next(timestamps))

    trace = TraceContext()
    trace.record_stage("query_processor.process", {"keyword_count": 2})

    payload = trace.to_dict()

    assert payload["trace_type"] == "query"
    assert payload["total_elapsed_ms"] == 600.0
    assert trace.finished_at == "2026-05-07T11:00:00.600000+00:00"


def test_trace_context_rejects_invalid_trace_type() -> None:
    with pytest.raises(ValueError, match="trace type"):
        TraceContext(trace_type="invalid")


def test_record_stage_after_finish_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_values = iter([2.0, 2.1])
    timestamps = iter(
        [
            datetime.fromisoformat("2026-05-07T12:00:00+00:00"),
            datetime.fromisoformat("2026-05-07T12:00:00.100000+00:00"),
        ]
    )
    monkeypatch.setattr(trace_context_module, "_perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(trace_context_module, "_utcnow", lambda: next(timestamps))

    trace = TraceContext()
    trace.finish()

    with pytest.raises(RuntimeError, match="finished"):
        trace.record_stage("late.stage")


def test_trace_collector_collects_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_values = iter([3.0, 3.2, 3.4])
    timestamps = iter(
        [
            datetime.fromisoformat("2026-05-07T13:00:00+00:00"),
            datetime.fromisoformat("2026-05-07T13:00:00.200000+00:00"),
            datetime.fromisoformat("2026-05-07T13:00:00.400000+00:00"),
        ]
    )
    monkeypatch.setattr(trace_context_module, "_perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(trace_context_module, "_utcnow", lambda: next(timestamps))

    collected: list[dict[str, object]] = []
    collector = TraceCollector(persister=collected.append)
    trace = TraceContext(trace_id="trace-collector")
    trace.record_stage("dense_retriever.retrieve", {"result_count": 3})

    collector.collect(trace)

    assert len(collector.traces) == 1
    assert len(collected) == 1
    assert collected[0]["trace_id"] == "trace-collector"
    assert collected[0]["stages"][0]["stage"] == "dense_retriever.retrieve"
