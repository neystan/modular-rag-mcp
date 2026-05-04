"""Embedding 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """所有 Embedding Provider 必须实现的最小接口。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        """批量生成文本向量。"""
