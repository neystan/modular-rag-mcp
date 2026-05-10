"""Qwen Reranker 测试。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError

import pytest

from core.settings import Settings
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.qwen_reranker import QwenReranker, QwenRerankerError
from libs.reranker.reranker_factory import RerankerFactory


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


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "qwen", "model": "qwen3-rerank", "top_k": 2, "api_key": "dashscope-secret"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="Python testing guide", score=0.1),
        RerankCandidate(id="b", text="Database storage basics", score=0.2),
        RerankCandidate(id="c", text="Python unit test patterns", score=0.3),
    ]


def test_factory_creates_qwen_reranker() -> None:
    reranker = RerankerFactory.create(make_settings())

    assert isinstance(reranker, QwenReranker)


def test_qwen_reranker_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.73},
                ]
            }
        )

    monkeypatch.setattr("libs.reranker.qwen_reranker.urlopen", fake_urlopen)
    reranker = QwenReranker({"model": "qwen3-rerank", "api_key": "dashscope-secret", "top_k": 2})

    ranked = reranker.rerank("python testing", make_candidates())

    assert [item.id for item in ranked] == ["c", "a"]
    assert [item.score for item in ranked] == [0.95, 0.73]
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    assert captured["headers"]["Authorization"] == "Bearer dashscope-secret"
    assert captured["payload"] == {
        "model": "qwen3-rerank",
        "query": "python testing",
        "documents": ["Python testing guide", "Database storage basics", "Python unit test patterns"],
        "top_n": 2,
    }


def test_qwen_reranker_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["headers"] = dict(request.header_items())
        return FakeHTTPResponse({"results": [{"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr("libs.reranker.qwen_reranker.urlopen", fake_urlopen)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-from-env")
    reranker = QwenReranker({"model": "qwen3-rerank"})

    ranked = reranker.rerank("python", [RerankCandidate(id="a", text="Python", score=0.1)])

    assert [item.id for item in ranked] == ["a"]
    assert captured["headers"]["Authorization"] == "Bearer dashscope-from-env"


def test_qwen_reranker_http_error_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHTTPError(HTTPError):
        def read(self) -> bytes:
            return json.dumps({"error": {"code": "InvalidApiKey", "message": "Invalid DashScope API key."}}).encode(
                "utf-8"
            )

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise FakeHTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("libs.reranker.qwen_reranker.urlopen", fake_urlopen)
    reranker = QwenReranker({"model": "qwen3-rerank", "api_key": "bad-key"})

    with pytest.raises(
        QwenRerankerError,
        match="qwen reranker HTTP error: 401; code=InvalidApiKey; Invalid DashScope API key.",
    ):
        reranker.rerank("python", [RerankCandidate(id="a", text="Python", score=0.1)])


def test_qwen_reranker_rejects_invalid_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"results": [{"index": "0", "relevance_score": 0.8}]})

    monkeypatch.setattr("libs.reranker.qwen_reranker.urlopen", fake_urlopen)
    reranker = QwenReranker({"model": "qwen3-rerank", "api_key": "dashscope-secret"})

    with pytest.raises(QwenRerankerError, match="results\\[0\\]\\.index must be int"):
        reranker.rerank("python", [RerankCandidate(id="a", text="Python", score=0.1)])
