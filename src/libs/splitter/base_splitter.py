"""Splitter 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """所有 Splitter Provider 必须实现的最小接口。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        """将文本切分为若干片段。"""
