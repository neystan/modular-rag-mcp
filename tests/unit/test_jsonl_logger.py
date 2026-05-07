"""JSON Lines logger 单元测试。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from observability.logger import JSONFormatter, get_trace_logger, write_trace


def _cleanup_logger(name: str) -> None:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_json_formatter_outputs_valid_json_line() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="tests.trace",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="trace collected",
        args=(),
        exc_info=None,
    )
    record.trace_type = "query"
    record.trace_id = "trace-123"

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "tests.trace"
    assert payload["message"] == "trace collected"
    assert payload["trace_type"] == "query"
    assert payload["trace_id"] == "trace-123"


def test_get_trace_logger_reuses_same_file_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    logger_name = "tests.trace.reuse"
    _cleanup_logger(logger_name)

    logger_a = get_trace_logger(name=logger_name, log_path=log_path)
    logger_b = get_trace_logger(name=logger_name, log_path=log_path)

    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1
    assert isinstance(logger_a.handlers[0], logging.FileHandler)

    _cleanup_logger(logger_name)


def test_write_trace_appends_jsonl_record(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "traces.jsonl"
    logger_name = "modular_rag_mcp.trace"
    _cleanup_logger(logger_name)
    trace_payload = {
        "trace_id": "trace-1",
        "trace_type": "ingestion",
        "started_at": "2026-05-07T10:00:00+00:00",
        "finished_at": "2026-05-07T10:00:01+00:00",
        "total_elapsed_ms": 1000.0,
        "stages": [{"stage": "loader.load", "payload": {"status": "ok"}, "elapsed_ms": 100.0}],
    }

    write_trace(trace_payload, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "trace collected"
    assert payload["trace_type"] == "ingestion"
    assert payload["trace_id"] == "trace-1"
    assert payload["trace"]["total_elapsed_ms"] == 1000.0

    _cleanup_logger(logger_name)


def test_write_trace_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="trace payload must be dict"):
        write_trace(["invalid"], log_path=tmp_path / "logs" / "traces.jsonl")  # type: ignore[arg-type]
