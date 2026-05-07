"""Trace 收集器。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.trace.trace_context import TraceContext


TracePersister = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class TraceCollector:
    """收集 trace，并将序列化结果交给外部持久化回调。"""

    persister: TracePersister | None = None
    traces: list[dict[str, Any]] = field(default_factory=list)

    def collect(self, trace: TraceContext) -> None:
        if not isinstance(trace, TraceContext):
            raise TypeError("trace collector expects TraceContext")

        payload = trace.to_dict()
        self.traces.append(payload)
        if self.persister is not None:
            self.persister(payload)
