"""Ollama LLM 本地后端实现。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.openai_llm import LLMProviderError


class OllamaLLM(BaseLLM):
    """通过 Ollama 本地 HTTP API 调用聊天模型。"""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"

    def chat(self, messages: list[ChatMessage] | list[dict[str, Any]]) -> str:
        payload = self._build_payload(messages)
        request = Request(
            self._chat_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=float(self.config.get("timeout", 30))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMProviderError(f"{self.provider_name} HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise LLMProviderError(f"{self.provider_name} network error: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"{self.provider_name} response error: {type(exc).__name__}") from exc

        return self._parse_response(data)

    def _build_payload(self, messages: list[ChatMessage] | list[dict[str, Any]]) -> dict[str, Any]:
        model = self.config.get("model")
        if not model:
            raise LLMProviderError(f"{self.provider_name} config error: model is required")

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._normalize_messages(messages),
            "stream": False,
        }
        options = self._options()
        if options:
            payload["options"] = options
        return payload

    def _normalize_messages(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        if not isinstance(messages, list) or not messages:
            raise LLMProviderError(f"{self.provider_name} input error: messages must be a non-empty list")

        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if isinstance(message, ChatMessage):
                role = message.role
                content = message.content
            elif isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            else:
                raise LLMProviderError(
                    f"{self.provider_name} input error: messages[{index}] must be ChatMessage or dict"
                )

            if not isinstance(role, str) or not role:
                raise LLMProviderError(f"{self.provider_name} input error: messages[{index}].role is required")
            if not isinstance(content, str) or not content:
                raise LLMProviderError(f"{self.provider_name} input error: messages[{index}].content is required")
            normalized.append({"role": role, "content": content})
        return normalized

    def _chat_endpoint(self) -> str:
        base_url = str(self.config.get("base_url", self.default_base_url)).rstrip("/")
        return f"{base_url}/api/chat"

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        # Ollama 将采样参数放在 options 中，避免污染顶层请求字段。
        for key in ("temperature", "top_p", "num_predict"):
            if key in self.config:
                options[key] = self.config[key]
        return options

    def _parse_response(self, data: dict[str, Any]) -> str:
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(f"{self.provider_name} response error: missing message.content") from exc
        if not isinstance(content, str):
            raise LLMProviderError(f"{self.provider_name} response error: content must be string")
        return content
