"""OpenAI-Compatible LLM 实现。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.llm.base_llm import BaseLLM, ChatMessage


class LLMProviderError(RuntimeError):
    """LLM Provider 调用错误。"""


class OpenAICompatibleLLM(BaseLLM):
    """兼容 OpenAI Chat Completions 协议的基础实现。"""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def chat(self, messages: list[ChatMessage] | list[dict[str, Any]]) -> str:
        payload = self._build_payload(messages)
        endpoint = self._chat_endpoint()
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
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
        normalized_messages = self._normalize_messages(messages)
        model = self.config.get("model")
        if not model:
            raise LLMProviderError(f"{self.provider_name} config error: model is required")

        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
        }
        for key in ("temperature", "max_tokens", "top_p"):
            if key in self.config:
                payload[key] = self.config[key]
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
        return f"{base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _parse_response(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"{self.provider_name} response error: missing choices[0].message.content") from exc
        if not isinstance(content, str):
            raise LLMProviderError(f"{self.provider_name} response error: content must be string")
        return content


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI 官方 API 实现。"""

    provider_name = "openai"
