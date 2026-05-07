"""项目日志与 Trace JSON Lines 输出。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


DEFAULT_TRACE_LOG_PATH = Path("logs/traces.jsonl")
_STANDARD_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """将 LogRecord 序列化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_")
        }
        payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "modular_rag_mcp") -> logging.Logger:
    """返回写入 stderr 的项目 logger。"""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_trace_logger(
    name: str = "modular_rag_mcp.trace",
    log_path: str | Path = DEFAULT_TRACE_LOG_PATH,
) -> logging.Logger:
    """返回写入 JSON Lines trace 文件的 logger。"""

    target_path = Path(log_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_target = str(target_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == target_path.resolve():
            return logger

    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    logger._trace_log_path = resolved_target  # type: ignore[attr-defined]
    return logger


def write_trace(trace_dict: dict[str, Any], log_path: str | Path = DEFAULT_TRACE_LOG_PATH) -> None:
    """将 trace 字典写入 JSON Lines 文件。"""

    if not isinstance(trace_dict, dict):
        raise TypeError("trace payload must be dict")

    logger = get_trace_logger(log_path=log_path)
    logger.info("trace collected", extra={"trace": trace_dict, **trace_dict})
