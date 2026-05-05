"""最小 TraceContext 实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceContext:
    """记录阶段性处理信息的轻量上下文。"""

    trace_id: str = field(default_factory=lambda: uuid4().hex)
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record_stage(self, stage: str, payload: dict[str, Any] | None = None) -> None:
        self.stages.append(
            {
                "stage": stage,
                "payload": dict(payload or {}),
            }
        )
