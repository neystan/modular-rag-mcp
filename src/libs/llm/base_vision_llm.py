"""Vision LLM 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisionChatResponse:
    """统一的图像理解响应结构。"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVisionLLM(ABC):
    """所有 Vision LLM Provider 必须实现的最小接口。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> VisionChatResponse:
        """根据文本和图片输入返回图像理解结果。"""

    def preprocess_image(self, image_path: str | bytes) -> str | bytes:
        """为子类预留图片压缩/格式转换扩展点。"""

        return image_path
