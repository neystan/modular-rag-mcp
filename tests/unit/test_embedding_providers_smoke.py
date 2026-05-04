"""OpenAI/Azure/Qwen Embedding Provider 冒烟测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.settings import Settings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import EmbeddingProviderError, OpenAIEmbedding
from libs.embedding.qwen_embedding import QwenEmbedding


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
        llm={"provider": "placeholder"},
        embedding={
            "provider": provider,
            "model": "text-embedding-3-small",
            "api_key": "secret",
            "base_url": "https://api.example.test/v1",
            "endpoint": "https://azure.example.test",
            "deployment": "embed-deployment",
        },
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_factory_routes_openai_compatible_embedding_providers() -> None:
    assert isinstance(EmbeddingFactory.create(make_settings("openai")), OpenAIEmbedding)
    assert isinstance(EmbeddingFactory.create(make_settings("azure")), AzureEmbedding)
    assert isinstance(EmbeddingFactory.create(make_settings("qwen")), QwenEmbedding)


def test_openai_embed_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse({"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = OpenAIEmbedding(
        {
            "model": "text-embedding-3-small",
            "api_key": "secret",
            "base_url": "https://api.test/v1",
            "dimensions": 2,
        }
    )

    result = embedding.embed(["第一段", "第二段"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://api.test/v1/embeddings"
    assert captured["payload"]["model"] == "text-embedding-3-small"
    assert captured["payload"]["input"] == ["第一段", "第二段"]
    assert captured["payload"]["dimensions"] == 2
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_azure_uses_deployment_endpoint_and_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"data": [{"embedding": [1, 2, 3]}]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = AzureEmbedding(
        {
            "model": "text-embedding-ada-002",
            "endpoint": "https://azure.test",
            "deployment": "embed-dep",
            "api_key": "secret",
            "api_version": "2024-01-01",
        }
    )

    assert embedding.embed(["hello"]) == [[1.0, 2.0, 3.0]]
    assert captured["url"] == "https://azure.test/openai/deployments/embed-dep/embeddings?api-version=2024-01-01"
    assert captured["headers"]["Api-key"] == "secret"
    assert captured["payload"]["input"] == ["hello"]


def test_qwen_uses_dashscope_compatible_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"data": [{"embedding": [0.9, 0.8]}]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = QwenEmbedding({"model": "text-embedding-v4", "api_key": "dashscope-secret"})

    assert embedding.embed(["你好"]) == [[0.9, 0.8]]
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer dashscope-secret"
    assert captured["payload"]["model"] == "text-embedding-v4"
    assert captured["payload"]["input"] == ["你好"]


def test_empty_input_error_mentions_provider() -> None:
    embedding = OpenAIEmbedding({"model": "text-embedding-3-small"})

    with pytest.raises(EmbeddingProviderError, match="openai input error"):
        embedding.embed([])


def test_too_long_input_can_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"data": [{"embedding": [0.5, 0.6]}]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = OpenAIEmbedding(
        {
            "model": "text-embedding-3-small",
            "max_input_length": 4,
            "too_long_strategy": "truncate",
        }
    )

    assert embedding.embed(["abcdef"]) == [[0.5, 0.6]]
    assert captured["payload"]["input"] == ["abcd"]


def test_response_shape_error_mentions_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"data": [{}]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)
    embedding = OpenAIEmbedding({"model": "text-embedding-3-small"})

    with pytest.raises(EmbeddingProviderError, match="openai response error"):
        embedding.embed(["hello"])
