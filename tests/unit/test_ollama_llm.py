"""Ollama LLM 本地后端测试。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError

import pytest

from core.settings import Settings
from libs.llm.base_llm import ChatMessage
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import LLMProviderError


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


def make_settings(provider: str = "ollama") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": provider, "model": "llama3.1", "base_url": "http://ollama.test"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_factory_creates_ollama_provider() -> None:
    llm = LLMFactory.create(make_settings())

    assert isinstance(llm, OllamaLLM)


def test_ollama_chat_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse({"message": {"content": "本地模型回答"}})

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = OllamaLLM(
        {
            "model": "llama3.1",
            "base_url": "http://ollama.test",
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 128,
            "timeout": 3,
        }
    )

    result = llm.chat([ChatMessage(role="user", content="你好")])

    assert result == "本地模型回答"
    assert captured["url"] == "http://ollama.test/api/chat"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["payload"] == {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 128},
    }
    assert captured["timeout"] == 3


def test_ollama_uses_default_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        return FakeHTTPResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = OllamaLLM({"model": "llama3.1"})

    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"


def test_ollama_missing_model_error_is_readable() -> None:
    llm = OllamaLLM({})

    with pytest.raises(LLMProviderError, match="ollama config error: model is required"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_ollama_network_error_does_not_leak_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = OllamaLLM({"model": "llama3.1", "base_url": "http://secret-host.local", "timeout": 1})

    with pytest.raises(LLMProviderError) as exc_info:
        llm.chat([{"role": "user", "content": "hi"}])

    error_text = str(exc_info.value)
    assert "ollama network error" in error_text
    assert "secret-host" not in error_text


def test_ollama_timeout_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = OllamaLLM({"model": "llama3.1"})

    with pytest.raises(LLMProviderError, match="ollama response error: TimeoutError"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_ollama_response_shape_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"done": True})

    monkeypatch.setattr("libs.llm.ollama_llm.urlopen", fake_urlopen)
    llm = OllamaLLM({"model": "llama3.1"})

    with pytest.raises(LLMProviderError, match="ollama response error: missing message.content"):
        llm.chat([{"role": "user", "content": "hi"}])
