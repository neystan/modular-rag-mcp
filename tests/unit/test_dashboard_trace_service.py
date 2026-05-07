"""Dashboard TraceService 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.dashboard.pages.ingestion_traces import collect_ingestion_trace_data
from observability.dashboard.services.trace_service import TraceService


def test_trace_service_lists_ingestion_traces_sorted_desc(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    _write_trace_log(
        log_path,
        [
            {
                "trace_id": "query-1",
                "trace_type": "query",
                "started_at": "2026-05-07T09:59:00+00:00",
                "finished_at": "2026-05-07T09:59:01+00:00",
                "total_elapsed_ms": 900.0,
                "stages": [],
            },
            {
                "trace_id": "ing-1",
                "trace_type": "ingestion",
                "started_at": "2026-05-07T10:00:00+00:00",
                "finished_at": "2026-05-07T10:00:01+00:00",
                "total_elapsed_ms": 1000.0,
                "stages": [{"stage": "load", "elapsed_ms": 100.0}],
            },
            {
                "trace_id": "ing-2",
                "trace_type": "ingestion",
                "started_at": "2026-05-07T10:05:00+00:00",
                "finished_at": "2026-05-07T10:05:02+00:00",
                "total_elapsed_ms": 2000.0,
                "stages": [{"stage": "load", "elapsed_ms": 200.0}, {"stage": "split", "elapsed_ms": 400.0}],
            },
        ],
    )

    traces = TraceService(log_path).list_ingestion_traces()

    assert [item.trace_id for item in traces] == ["ing-2", "ing-1"]
    assert traces[0].stage_count == 2
    assert traces[1].total_elapsed_ms == 1000.0
    assert traces[0].source_name == ""


def test_trace_service_get_ingestion_trace_extracts_stage_durations(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    _write_trace_log(
        log_path,
        [
            {
                "trace_id": "ing-1",
                "trace_type": "ingestion",
                "started_at": "2026-05-07T10:00:00+00:00",
                "finished_at": "2026-05-07T10:00:02+00:00",
                "total_elapsed_ms": 2000.0,
                "stages": [
                    {"stage": "loader.load", "elapsed_ms": 80.0, "payload": {"status": "ok"}},
                    {"stage": "dashboard.upload", "elapsed_ms": 90.0, "payload": {"original_filename": "sample.pdf"}},
                    {"stage": "load", "elapsed_ms": 100.0, "payload": {"details": {"text_length": 10}}},
                    {"stage": "chunker.split", "elapsed_ms": 260.0, "payload": {"status": "ok"}},
                    {"stage": "split", "elapsed_ms": 300.0, "payload": {"details": {"chunk_count": 2}}},
                    {"stage": "transform", "elapsed_ms": 900.0, "payload": {"details": {"transform_count": 3}}},
                    {"stage": "embed", "elapsed_ms": 1400.0, "payload": {"details": {"dense_record_count": 2}}},
                    {"stage": "upsert", "elapsed_ms": 1800.0, "payload": {"details": {"collection": "manuals"}}},
                ],
            }
        ],
    )

    detail = TraceService(log_path).get_ingestion_trace("ing-1")

    assert detail.trace_id == "ing-1"
    assert detail.source_name == "sample.pdf"
    assert [item.stage for item in detail.stages] == ["load", "split", "transform", "embed", "upsert"]
    assert [item.duration_ms for item in detail.stages] == [100.0, 200.0, 600.0, 500.0, 400.0]
    assert detail.stages[0].payload["details"]["text_length"] == 10
    assert len(detail.raw_stages) == 8


def test_collect_ingestion_trace_data_selects_latest_trace(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    _write_trace_log(
        log_path,
        [
            {
                "trace_id": "ing-old",
                "trace_type": "ingestion",
                "started_at": "2026-05-07T10:00:00+00:00",
                "finished_at": "2026-05-07T10:00:01+00:00",
                "total_elapsed_ms": 1000.0,
                "stages": [],
            },
            {
                "trace_id": "ing-new",
                "trace_type": "ingestion",
                "started_at": "2026-05-07T10:02:00+00:00",
                "finished_at": "2026-05-07T10:02:02+00:00",
                "total_elapsed_ms": 2000.0,
                "stages": [{"stage": "dashboard.upload", "elapsed_ms": 50.0, "payload": {"original_filename": "new.pdf"}}, {"stage": "load", "elapsed_ms": 200.0}],
            },
        ],
    )

    payload = collect_ingestion_trace_data(TraceService(log_path))

    assert payload["selected_trace_id"] == "ing-new"
    assert payload["detail"] is not None
    assert payload["detail"].trace_id == "ing-new"
    assert payload["detail"].source_name == "new.pdf"


def test_trace_service_rejects_invalid_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid trace json"):
        TraceService(log_path).list_ingestion_traces()


def _write_trace_log(log_path: Path, traces: list[dict[str, object]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "message": "trace collected",
                    "trace_type": trace["trace_type"],
                    "trace_id": trace["trace_id"],
                    "trace": trace,
                },
                ensure_ascii=False,
            )
            for trace in traces
        )
        + "\n",
        encoding="utf-8",
    )
