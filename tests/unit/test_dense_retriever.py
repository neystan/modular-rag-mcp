"""DenseRetriever 单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from core.query_engine.dense_retriever import DenseRetriever, DenseRetrieverError
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore, VectorQueryResult, VectorRecord


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]]) -> None:
        super().__init__({})
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.vectors


class FakeVectorStore(BaseVectorStore):
    def __init__(self, results: list[VectorQueryResult]) -> None:
        super().__init__({})
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        self.calls.append({"vector": list(vector), "top_k": top_k, "filters": filters})
        return list(self.results)


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


def test_retrieve_embeds_query_and_queries_vector_store() -> None:
    embedding = FakeEmbedding([[0.1, 0.2]])
    vector_store = FakeVectorStore(
        [
            VectorQueryResult(
                id="chunk-a",
                score=0.91,
                text="Dense retrieval result",
                metadata={"source_path": "docs/a.pdf", "collection": "manuals"},
            )
        ]
    )
    retriever = DenseRetriever(make_settings(), embedding_client=embedding, vector_store=vector_store)

    results = retriever.retrieve("dense retrieval", top_k=3, filters={"collection": "manuals"})

    assert embedding.calls == [["dense retrieval"]]
    assert vector_store.calls == [{"vector": [0.1, 0.2], "top_k": 3, "filters": {"collection": "manuals"}}]
    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk_id == "chunk-a"
    assert results[0].text == "Dense retrieval result"


def test_retrieve_records_trace_stage() -> None:
    embedding = FakeEmbedding([[0.3, 0.4]])
    vector_store = FakeVectorStore(
        [
            VectorQueryResult(
                id="chunk-trace",
                score=0.5,
                text="trace result",
                metadata={"source_path": "docs/trace.pdf"},
            )
        ]
    )
    retriever = DenseRetriever(make_settings(), embedding_client=embedding, vector_store=vector_store)
    trace = TraceContext()

    retriever.retrieve("trace query", top_k=1, trace=trace)

    assert any(stage["stage"] == "dense_retriever.retrieve" for stage in trace.stages)


def test_retrieve_rejects_blank_query() -> None:
    retriever = DenseRetriever(
        make_settings(),
        embedding_client=FakeEmbedding([[0.1]]),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(DenseRetrieverError, match="query is required"):
        retriever.retrieve("   ", top_k=1)


def test_retrieve_rejects_invalid_top_k() -> None:
    retriever = DenseRetriever(
        make_settings(),
        embedding_client=FakeEmbedding([[0.1]]),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(DenseRetrieverError, match="top_k must be positive int"):
        retriever.retrieve("query", top_k=0)


def test_retrieve_requires_single_query_vector() -> None:
    retriever = DenseRetriever(
        make_settings(),
        embedding_client=FakeEmbedding([[0.1], [0.2]]),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(DenseRetrieverError, match="expected 1 query vector, got 2"):
        retriever.retrieve("query", top_k=1)


def test_retrieve_requires_text_in_vector_store_results() -> None:
    retriever = DenseRetriever(
        make_settings(),
        embedding_client=FakeEmbedding([[0.1]]),
        vector_store=FakeVectorStore(
            [VectorQueryResult(id="chunk-x", score=0.4, text="", metadata={"source_path": "docs/x.pdf"})]
        ),
    )

    with pytest.raises(DenseRetrieverError, match="missing text"):
        retriever.retrieve("query", top_k=1)
