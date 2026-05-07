"""HybridSearch 集成测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine.hybrid_search import HybridSearch, HybridSearchError
from core.query_engine.query_processor import ProcessedQuery
from core.query_engine.reranker import Reranker
from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class FakeQueryProcessor:
    def __init__(self, processed: ProcessedQuery) -> None:
        self.processed = processed
        self.calls: list[str] = []

    def process(self, query: str, trace: Any | None = None) -> ProcessedQuery:
        self.calls.append(query)
        return self.processed


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        if self.error is not None:
            raise self.error
        return list(self.results)


class FakeSparseRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, keywords: list[str], top_k: int, trace: Any | None = None) -> list[RetrievalResult]:
        self.calls.append({"keywords": list(keywords), "top_k": top_k})
        if self.error is not None:
            raise self.error
        return list(self.results)


class FakeFusion:
    def __init__(self, fused: list[RetrievalResult]) -> None:
        self.fused = fused
        self.calls: list[dict[str, Any]] = []

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "dense_ids": [item.chunk_id for item in dense_results],
                "sparse_ids": [item.chunk_id for item in sparse_results],
                "top_k": top_k,
            }
        )
        return list(self.fused)


class FakeRerankerBackend(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        assert query == "query"
        reordered = list(reversed(candidates))
        return [
            RerankCandidate(id=item.id, text=item.text, score=1.0 - index * 0.1, metadata=dict(item.metadata))
            for index, item in enumerate(reordered)
        ]


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_result(chunk_id: str, *, collection: str = "manuals", doc_type: str = "guide") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=1.0,
        text=f"text for {chunk_id}",
        metadata={
            "source_path": f"docs/{chunk_id}.pdf",
            "collection": collection,
            "doc_type": doc_type,
        },
    )


def test_hybrid_search_orchestrates_query_dense_sparse_and_fusion() -> None:
    query_processor = FakeQueryProcessor(
        ProcessedQuery(
            original_query="How to configure Azure?",
            normalized_query="How to configure Azure?",
            keywords=["configure", "azure"],
            filters={},
        )
    )
    dense = FakeDenseRetriever([make_result("chunk-a"), make_result("chunk-b")])
    sparse = FakeSparseRetriever([make_result("chunk-b"), make_result("chunk-c")])
    fusion = FakeFusion([make_result("chunk-b"), make_result("chunk-a"), make_result("chunk-c")])
    search = HybridSearch(
        make_settings(),
        query_processor=query_processor,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )

    results = search.search("How to configure Azure?", top_k=2)

    assert query_processor.calls == ["How to configure Azure?"]
    assert dense.calls == [{"query": "How to configure Azure?", "top_k": 2, "filters": {}}]
    assert sparse.calls == [{"keywords": ["configure", "azure"], "top_k": 2}]
    assert fusion.calls == [{"dense_ids": ["chunk-a", "chunk-b"], "sparse_ids": ["chunk-b", "chunk-c"], "top_k": 2}]
    assert [item.chunk_id for item in results] == ["chunk-b", "chunk-a"]


def test_hybrid_search_applies_metadata_filters_as_fallback() -> None:
    query_processor = FakeQueryProcessor(
        ProcessedQuery(
            original_query="query",
            normalized_query="query",
            keywords=["query"],
            filters={},
        )
    )
    dense = FakeDenseRetriever([make_result("chunk-a", collection="manuals"), make_result("chunk-b", collection="faq")])
    sparse = FakeSparseRetriever([])
    fusion = FakeFusion([])
    search = HybridSearch(
        make_settings(),
        query_processor=query_processor,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )

    results = search.search("query", top_k=5, filters={"collection": "manuals"})

    assert [item.chunk_id for item in results] == ["chunk-a"]


def test_hybrid_search_falls_back_to_sparse_when_dense_fails() -> None:
    query_processor = FakeQueryProcessor(
        ProcessedQuery(original_query="query", normalized_query="query", keywords=["query"], filters={})
    )
    dense = FakeDenseRetriever(error=RuntimeError("dense boom"))
    sparse = FakeSparseRetriever([make_result("chunk-s")])
    fusion = FakeFusion([])
    search = HybridSearch(
        make_settings(),
        query_processor=query_processor,
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )
    trace = TraceContext()

    results = search.search("query", top_k=3, trace=trace)

    assert [item.chunk_id for item in results] == ["chunk-s"]
    assert any(stage["stage"] == "hybrid_search.search" for stage in trace.stages)


def test_query_trace_contains_query_dense_sparse_fusion_and_rerank_stages() -> None:
    search = HybridSearch(
        make_settings(),
        query_processor=FakeQueryProcessor(
            ProcessedQuery(original_query="query", normalized_query="query", keywords=["query"], filters={})
        ),
        dense_retriever=FakeDenseRetriever([make_result("chunk-a"), make_result("chunk-b")]),
        sparse_retriever=FakeSparseRetriever([make_result("chunk-b"), make_result("chunk-c")]),
        fusion=FakeFusion([make_result("chunk-b"), make_result("chunk-a"), make_result("chunk-c")]),
    )
    reranker = Reranker(make_settings(), reranker_backend=FakeRerankerBackend({}))
    trace = TraceContext(trace_type="query")

    search_results = search.search("query", top_k=3, trace=trace)
    reranked = reranker.rerank("query", search_results, top_k=2, trace=trace)
    payload = trace.to_dict()

    assert payload["trace_type"] == "query"
    assert [item.chunk_id for item in reranked] == ["chunk-c", "chunk-a"]

    expected_stages = {"query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank"}
    stage_map = {stage["stage"]: stage for stage in payload["stages"]}
    assert expected_stages.issubset(stage_map)
    for stage_name in expected_stages:
        stage = stage_map[stage_name]
        assert stage["elapsed_ms"] >= 0
        assert stage["payload"]["method"]

    assert stage_map["dense_retrieval"]["payload"]["provider"] == "FakeDenseRetriever"
    assert stage_map["sparse_retrieval"]["payload"]["provider"] == "FakeSparseRetriever"
    assert stage_map["fusion"]["payload"]["method"] == "rrf"
    assert stage_map["rerank"]["payload"]["provider"] == "FakeRerankerBackend"


def test_hybrid_search_raises_when_both_routes_fail() -> None:
    search = HybridSearch(
        make_settings(),
        query_processor=FakeQueryProcessor(
            ProcessedQuery(original_query="query", normalized_query="query", keywords=["query"], filters={})
        ),
        dense_retriever=FakeDenseRetriever(error=RuntimeError("dense boom")),
        sparse_retriever=FakeSparseRetriever(error=RuntimeError("sparse boom")),
        fusion=FakeFusion([]),
    )

    with pytest.raises(HybridSearchError, match="both failed"):
        search.search("query", top_k=2)


def test_hybrid_search_rejects_invalid_top_k() -> None:
    search = HybridSearch(
        make_settings(),
        query_processor=FakeQueryProcessor(
            ProcessedQuery(original_query="query", normalized_query="query", keywords=["query"], filters={})
        ),
        dense_retriever=FakeDenseRetriever([]),
        sparse_retriever=FakeSparseRetriever([]),
        fusion=FakeFusion([]),
    )

    with pytest.raises(HybridSearchError, match="top_k must be positive int"):
        search.search("query", top_k=0)
