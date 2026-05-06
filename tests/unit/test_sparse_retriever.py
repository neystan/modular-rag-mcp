"""SparseRetriever 单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from core.query_engine.sparse_retriever import SparseRetriever, SparseRetrieverError
from ingestion.storage.bm25_indexer import BM25QueryResult
from libs.vector_store.base_vector_store import BaseVectorStore, VectorQueryResult, VectorRecord


class FakeBM25Indexer:
    def __init__(self, results: list[BM25QueryResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def query(self, query: str | list[str], top_k: int = 5) -> list[BM25QueryResult]:
        self.calls.append({"query": query, "top_k": top_k})
        return list(self.results)


class FakeVectorStore(BaseVectorStore):
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        super().__init__({})
        self.payloads = payloads
        self.calls: list[list[str]] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        return []

    def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[dict[str, Any]]:
        self.calls.append(list(ids))
        by_id = {str(item["id"]): item for item in self.payloads}
        return [by_id[item_id] for item_id in ids if item_id in by_id]


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


def test_retrieve_merges_bm25_scores_with_vector_payloads() -> None:
    bm25 = FakeBM25Indexer(
        [
            BM25QueryResult(chunk_id="chunk-b", score=1.8),
            BM25QueryResult(chunk_id="chunk-a", score=1.2),
        ]
    )
    vector_store = FakeVectorStore(
        [
            {"id": "chunk-a", "text": "A text", "metadata": {"source_path": "docs/a.pdf"}},
            {"id": "chunk-b", "text": "B text", "metadata": {"source_path": "docs/b.pdf"}},
        ]
    )
    retriever = SparseRetriever(make_settings(), bm25_indexer=bm25, vector_store=vector_store)

    results = retriever.retrieve(["alpha", "beta"], top_k=2)

    assert bm25.calls == [{"query": ["alpha", "beta"], "top_k": 2}]
    assert vector_store.calls == [["chunk-b", "chunk-a"]]
    assert [item.chunk_id for item in results] == ["chunk-b", "chunk-a"]
    assert isinstance(results[0], RetrievalResult)
    assert results[0].score == 1.8
    assert results[0].text == "B text"


def test_retrieve_skips_missing_payloads() -> None:
    bm25 = FakeBM25Indexer(
        [
            BM25QueryResult(chunk_id="chunk-a", score=1.0),
            BM25QueryResult(chunk_id="chunk-missing", score=0.7),
        ]
    )
    vector_store = FakeVectorStore(
        [{"id": "chunk-a", "text": "A text", "metadata": {"source_path": "docs/a.pdf"}}]
    )
    retriever = SparseRetriever(make_settings(), bm25_indexer=bm25, vector_store=vector_store)

    results = retriever.retrieve(["alpha"], top_k=5)

    assert [item.chunk_id for item in results] == ["chunk-a"]


def test_retrieve_records_trace_stage() -> None:
    bm25 = FakeBM25Indexer([BM25QueryResult(chunk_id="chunk-a", score=1.0)])
    vector_store = FakeVectorStore(
        [{"id": "chunk-a", "text": "A text", "metadata": {"source_path": "docs/a.pdf"}}]
    )
    retriever = SparseRetriever(make_settings(), bm25_indexer=bm25, vector_store=vector_store)
    trace = TraceContext()

    retriever.retrieve(["alpha"], top_k=1, trace=trace)

    assert any(stage["stage"] == "sparse_retriever.retrieve" for stage in trace.stages)


def test_retrieve_rejects_empty_keywords() -> None:
    retriever = SparseRetriever(
        make_settings(),
        bm25_indexer=FakeBM25Indexer([]),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(SparseRetrieverError, match="keywords must be non-empty list"):
        retriever.retrieve([], top_k=1)


def test_retrieve_rejects_invalid_top_k() -> None:
    retriever = SparseRetriever(
        make_settings(),
        bm25_indexer=FakeBM25Indexer([]),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(SparseRetrieverError, match="top_k must be positive int"):
        retriever.retrieve(["alpha"], top_k=0)


def test_retrieve_requires_text_in_payload() -> None:
    retriever = SparseRetriever(
        make_settings(),
        bm25_indexer=FakeBM25Indexer([BM25QueryResult(chunk_id="chunk-a", score=1.0)]),
        vector_store=FakeVectorStore([{"id": "chunk-a", "text": "", "metadata": {"source_path": "docs/a.pdf"}}]),
    )

    with pytest.raises(SparseRetrieverError, match="missing text"):
        retriever.retrieve(["alpha"], top_k=1)
