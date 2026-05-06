"""Core Reranker 回退测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine.reranker import Reranker, RerankerError
from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class FakeSuccessReranker(BaseReranker):
    """测试用成功重排器。"""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        assert query == "hybrid search"
        return [
            RerankCandidate(id="chunk-b", text="text b", score=0.99, metadata={"source_path": "docs/b.pdf"}),
            RerankCandidate(id="chunk-a", text="text a", score=0.88, metadata={"source_path": "docs/a.pdf"}),
        ]


class FakeFailureReranker(BaseReranker):
    """测试用失败重排器。"""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        raise TimeoutError("backend timeout")


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none", "top_k": 2},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_candidates() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-a",
            score=0.41,
            text="text a",
            metadata={"source_path": "docs/a.pdf"},
        ),
        RetrievalResult(
            chunk_id="chunk-b",
            score=0.39,
            text="text b",
            metadata={"source_path": "docs/b.pdf"},
        ),
        RetrievalResult(
            chunk_id="chunk-c",
            score=0.21,
            text="text c",
            metadata={"source_path": "docs/c.pdf"},
        ),
    ]


def test_reranker_reorders_results_when_backend_succeeds() -> None:
    reranker = Reranker(make_settings(), reranker_backend=FakeSuccessReranker({}))

    results = reranker.rerank("hybrid search", make_candidates())

    assert [item.chunk_id for item in results] == ["chunk-b", "chunk-a"]
    assert [item.score for item in results] == [0.99, 0.88]
    assert all(item.metadata["rerank_fallback"] is False for item in results)


def test_reranker_falls_back_to_fusion_order_on_backend_failure() -> None:
    reranker = Reranker(make_settings(), reranker_backend=FakeFailureReranker({}))

    results = reranker.rerank("hybrid search", make_candidates())

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-b"]
    assert all(item.metadata["rerank_fallback"] is True for item in results)
    assert all(item.metadata["rerank_fallback_reason"] == "backend timeout" for item in results)


def test_reranker_records_trace_stage() -> None:
    reranker = Reranker(make_settings(), reranker_backend=FakeFailureReranker({}))
    trace = TraceContext()

    reranker.rerank("hybrid search", make_candidates(), trace=trace)

    stage = next(item for item in trace.stages if item["stage"] == "query_reranker.rerank")
    assert stage["payload"]["fallback"] is True
    assert stage["payload"]["result_count"] == 2


def test_reranker_rejects_blank_query() -> None:
    reranker = Reranker(make_settings(), reranker_backend=FakeSuccessReranker({}))

    with pytest.raises(RerankerError, match="query is required"):
        reranker.rerank("   ", make_candidates())
