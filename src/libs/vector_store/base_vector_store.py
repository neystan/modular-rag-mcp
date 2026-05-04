"""VectorStore 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    """向量存储写入记录。"""

    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorQueryResult:
    """向量检索结果。"""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """所有 VectorStore Provider 必须实现的最小接口。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        """写入或更新向量记录，返回写入数量。"""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        """按向量查询 Top-K 结果。"""
