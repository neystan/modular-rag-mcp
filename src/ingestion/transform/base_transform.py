"""Transform 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import Chunk


class BaseTransform(ABC):
    """所有 Chunk 增强器必须实现的最小接口。"""

    @abstractmethod
    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        """对 chunk 列表做变换并返回新列表。"""
