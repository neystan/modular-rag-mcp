"""Dashboard Query TraceService 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from observability.dashboard.pages.query_traces import collect_query_trace_data
from observability.dashboard.services.trace_service import TraceService


def test_trace_service_lists_query_traces_and_filters_by_keyword(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    _write_trace_log(
        log_path,
        [
            _query_trace("trace-1", "How to configure Azure?", ["chunk-a"], ["chunk-a"], ["chunk-a"]),
            _query_trace("trace-2", "How to tune rerank?", ["chunk-b"], ["chunk-b"], ["chunk-b"]),
        ],
    )

    traces = TraceService(log_path).list_query_traces("azure")

    assert [item.trace_id for item in traces] == ["trace-1"]
    assert traces[0].query_text == "How to configure Azure?"
    assert traces[0].result_count == 1


def test_trace_service_get_query_trace_extracts_comparison_ids(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    _write_trace_log(
        log_path,
        [
            _query_trace(
                "trace-1",
                "How to configure Azure?",
                ["chunk-d1", "chunk-d2"],
                ["chunk-s1", "chunk-d2"],
                ["chunk-d2", "chunk-d1", "chunk-s1"],
                rerank_result_ids=["chunk-s1", "chunk-d2"],
            )
        ],
    )

    detail = TraceService(log_path).get_query_trace("trace-1")

    assert detail.query_text == "How to configure Azure?"
    assert detail.keywords == ["configure", "azure"]
    assert detail.dense_ids == ["chunk-d1", "chunk-d2"]
    assert detail.sparse_ids == ["chunk-s1", "chunk-d2"]
    assert detail.fusion_ids == ["chunk-d2", "chunk-d1", "chunk-s1"]
    assert detail.rerank_input_ids == ["chunk-d2", "chunk-d1", "chunk-s1"]
    assert detail.rerank_result_ids == ["chunk-s1", "chunk-d2"]
    assert detail.final_ids == ["chunk-s1", "chunk-d2"]


def test_collect_query_trace_data_selects_latest_trace(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    _write_trace_log(
        log_path,
        [
            _query_trace("trace-old", "old query", ["chunk-a"], ["chunk-a"], ["chunk-a"], started_at="2026-05-07T10:00:00+00:00"),
            _query_trace("trace-new", "new query", ["chunk-b"], ["chunk-b"], ["chunk-b"], started_at="2026-05-07T10:05:00+00:00"),
        ],
    )

    payload = collect_query_trace_data(TraceService(log_path))

    assert payload["selected_trace_id"] == "trace-new"
    assert payload["detail"] is not None
    assert payload["detail"].query_text == "new query"


def _query_trace(
    trace_id: str,
    query_text: str,
    dense_ids: list[str],
    sparse_ids: list[str],
    fusion_ids: list[str],
    *,
    rerank_result_ids: list[str] | None = None,
    started_at: str = "2026-05-07T10:00:00+00:00",
) -> dict[str, object]:
    final_ids = rerank_result_ids or fusion_ids
    return {
        "trace_id": trace_id,
        "trace_type": "query",
        "started_at": started_at,
        "finished_at": "2026-05-07T10:00:02+00:00",
        "total_elapsed_ms": 2000.0,
        "stages": [
            {
                "stage": "query_processing",
                "elapsed_ms": 50.0,
                "payload": {
                    "details": {
                        "query_text": query_text,
                        "keywords": ["configure", "azure"],
                        "filters": {"collection": "manuals"},
                    }
                },
            },
            {"stage": "dense_retrieval", "elapsed_ms": 300.0, "payload": {"details": {"chunk_ids": dense_ids}}},
            {"stage": "sparse_retrieval", "elapsed_ms": 500.0, "payload": {"details": {"chunk_ids": sparse_ids}}},
            {"stage": "fusion", "elapsed_ms": 700.0, "payload": {"details": {"chunk_ids": fusion_ids}}},
            {
                "stage": "rerank",
                "elapsed_ms": 900.0,
                "payload": {"details": {"input_ids": fusion_ids, "result_ids": rerank_result_ids or fusion_ids}},
            },
            {
                "stage": "query.execution",
                "elapsed_ms": 950.0,
                "payload": {
                    "query_text": query_text,
                    "collection": "manuals",
                    "top_k": 5,
                    "rerank_enabled": True,
                    "rerank_applied": rerank_result_ids is not None,
                    "final_ids": final_ids,
                    "result_count": len(final_ids),
                },
            },
        ],
    }


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
