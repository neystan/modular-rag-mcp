"""OpenAI-compatible LLM Provider 冒烟测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.settings import Settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import ChatMessage
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import LLMProviderError, OpenAILLM


class FakeHTTPResponse:
    """测试用 HTTP 响应。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def make_settings(provider: str) -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={
            "provider": provider,
            "model": "test-model",
            "api_key": "secret",
            "base_url": "https://example.test/v1",
            "endpoint": "https://azure.example.test",
            "deployment": "test-deployment",
        },
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_factory_routes_openai_compatible_providers() -> None:
    assert isinstance(LLMFactory.create(make_settings("openai")), OpenAILLM)
    assert isinstance(LLMFactory.create(make_settings("azure")), AzureLLM)
    assert isinstance(LLMFactory.create(make_settings("deepseek")), DeepSeekLLM)


def test_openai_chat_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse({"choices": [{"message": {"content": "回答"}}]})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = OpenAILLM({"model": "gpt-test", "api_key": "secret", "base_url": "https://api.test/v1"})

    result = llm.chat([ChatMessage(role="user", content="你好")])

    assert result == "回答"
    assert captured["url"] == "https://api.test/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "你好"}]
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_azure_uses_deployment_endpoint_and_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        return FakeHTTPResponse({"choices": [{"message": {"content": "azure answer"}}]})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = AzureLLM(
        {
            "model": "ignored-by-endpoint",
            "endpoint": "https://azure.test",
            "deployment": "dep",
            "api_key": "secret",
            "api_version": "2024-01-01",
        }
    )

    assert llm.chat([{"role": "user", "content": "hi"}]) == "azure answer"
    assert captured["url"] == (
        "https://azure.test/openai/deployments/dep/chat/completions?api-version=2024-01-01"
    )
    assert captured["headers"]["Api-key"] == "secret"


def test_input_shape_error_mentions_provider() -> None:
    llm = OpenAILLM({"model": "gpt-test"})

    with pytest.raises(LLMProviderError, match="openai input error"):
        llm.chat([])


def test_missing_model_error_is_readable() -> None:
    llm = DeepSeekLLM({})

    with pytest.raises(LLMProviderError, match="deepseek config error: model is required"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_response_shape_error_mentions_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"choices": []})

    monkeypatch.setattr("libs.llm.openai_llm.urlopen", fake_urlopen)
    llm = OpenAILLM({"model": "gpt-test"})

    with pytest.raises(LLMProviderError, match="openai response error"):
        llm.chat([{"role": "user", "content": "hi"}])
