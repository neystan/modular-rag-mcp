"""LLM 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    """统一的聊天消息结构。"""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """所有 LLM Provider 必须实现的最小接口。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def chat(self, messages: list[ChatMessage] | list[dict[str, Any]]) -> str:
        """根据消息列表返回模型文本响应。"""
