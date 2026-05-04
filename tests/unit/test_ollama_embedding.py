"""Ollama Embedding 本地后端测试。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError

import pytest

from core.settings import Settings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.openai_embedding import EmbeddingProviderError


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
        llm={"provider": "placeholder"},
        embedding={"provider": provider, "model": "nomic-embed-text", "base_url": "http://ollama.test"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_factory_creates_ollama_provider() -> None:
    embedding = EmbeddingFactory.create(make_settings())

    assert isinstance(embedding, OllamaEmbedding)


def test_ollama_embed_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding(
        {
            "model": "nomic-embed-text",
            "base_url": "http://ollama.test",
            "truncate": True,
            "timeout": 3,
        }
    )

    result = embedding.embed(["第一段", "第二段"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "http://ollama.test/api/embed"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["payload"] == {
        "model": "nomic-embed-text",
        "input": ["第一段", "第二段"],
        "truncate": True,
    }
    assert captured["timeout"] == 3


def test_ollama_uses_default_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        return FakeHTTPResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding({"model": "nomic-embed-text"})

    assert embedding.embed(["hi"]) == [[0.1, 0.2]]
    assert captured["url"] == "http://localhost:11434/api/embed"


def test_ollama_missing_model_error_is_readable() -> None:
    embedding = OllamaEmbedding({})

    with pytest.raises(EmbeddingProviderError, match="ollama config error: model is required"):
        embedding.embed(["hi"])


def test_ollama_network_error_does_not_leak_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise URLError("connection refused")

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding({"model": "nomic-embed-text", "base_url": "http://secret-host.local", "timeout": 1})

    with pytest.raises(EmbeddingProviderError) as exc_info:
        embedding.embed(["hi"])

    error_text = str(exc_info.value)
    assert "ollama network error" in error_text
    assert "secret-host" not in error_text


def test_ollama_timeout_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding({"model": "nomic-embed-text"})

    with pytest.raises(EmbeddingProviderError, match="ollama response error: TimeoutError"):
        embedding.embed(["hi"])


def test_ollama_response_shape_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"done": True})

    monkeypatch.setattr("libs.embedding.ollama_embedding.urlopen", fake_urlopen)
    embedding = OllamaEmbedding({"model": "nomic-embed-text"})

    with pytest.raises(EmbeddingProviderError, match="ollama response error: missing embeddings"):
        embedding.embed(["hi"])


def test_ollama_too_long_input_can_truncate() -> None:
    embedding = OllamaEmbedding(
        {
            "model": "nomic-embed-text",
            "max_input_length": 4,
            "too_long_strategy": "truncate",
        }
    )

    assert embedding._normalize_texts(["abcdef"]) == ["abcd"]
