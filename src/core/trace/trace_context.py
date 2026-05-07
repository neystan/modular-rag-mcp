"""TraceContext 请求级追踪上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import time


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _perf_counter() -> float:
    return time.perf_counter()


@dataclass(slots=True)
class TraceContext:
    """记录阶段性处理信息并汇总总耗时。"""

    trace_type: str = "query"
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(init=False)
    finished_at: str | None = field(default=None, init=False)
    total_elapsed_ms: float | None = field(default=None, init=False)
    _started_perf: float = field(init=False, repr=False)
    _finished_perf: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trace_type not in {"query", "ingestion"}:
            raise ValueError("trace type must be 'query' or 'ingestion'")
        self.started_at = _utcnow().isoformat()
        self._started_perf = _perf_counter()

    def record_stage(self, stage: str, payload: dict[str, Any] | None = None) -> None:
        if self.finished_at is not None:
            raise RuntimeError("cannot record stage after trace is finished")
        self.stages.append(
            {
                "stage": stage,
                "payload": dict(payload or {}),
                "recorded_at": _utcnow().isoformat(),
                "elapsed_ms": self.elapsed_ms(),
            }
        )

    def finish(self) -> None:
        if self.finished_at is not None:
            return
        self._finished_perf = _perf_counter()
        self.finished_at = _utcnow().isoformat()
        self.total_elapsed_ms = self.elapsed_ms()

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        if stage_name is None:
            end_perf = self._finished_perf if self._finished_perf is not None else _perf_counter()
            return round(max(end_perf - self._started_perf, 0.0) * 1000, 3)

        for item in reversed(self.stages):
            if item["stage"] == stage_name:
                return float(item["elapsed_ms"])
        raise KeyError(f"unknown stage: {stage_name}")

    def to_dict(self) -> dict[str, Any]:
        if self.finished_at is None:
            self.finish()
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "stages": [
                {
                    "stage": item["stage"],
                    "payload": dict(item["payload"]),
                    "recorded_at": item["recorded_at"],
                    "elapsed_ms": item["elapsed_ms"],
                }
                for item in self.stages
            ],
        }
