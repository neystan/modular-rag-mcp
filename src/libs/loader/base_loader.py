"""Loader 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import Document


class LoaderError(ValueError):
    """文档加载错误。"""


class BaseLoader(ABC):
    """所有文档 Loader 必须实现的最小接口。"""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """加载文件并返回统一 Document。"""

    def _resolve_file(self, path: str | Path) -> Path:
        file_path = Path(path)
        if not file_path.is_file():
            raise LoaderError(f"file not found: {file_path}")
        return file_path
