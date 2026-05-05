"""DenseEncoder 单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder, DenseEncoderError
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        super().__init__({})
        self.vectors = vectors or []
        self.calls: list[dict[str, Any]] = []

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.calls.append({"texts": texts, "trace": trace})
        if self.vectors:
            return self.vectors
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder", "model": "fake-model"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="chunk-1",
            text="first body",
            metadata={"source_path": "docs/sample.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=10,
            source_ref="doc-1",
        ),
        Chunk(
            id="chunk-2",
            text="second body",
            metadata={"source_path": "docs/sample.pdf", "chunk_index": 1},
            start_offset=11,
            end_offset=22,
            source_ref="doc-1",
        ),
    ]


def test_encode_returns_chunk_records_with_dense_vectors() -> None:
    embedding = FakeEmbedding(vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    encoder = DenseEncoder(make_settings(), embedding=embedding)
    trace = TraceContext()

    records = encoder.encode(make_chunks(), trace=trace)

    assert [record.id for record in records] == ["chunk-1", "chunk-2"]
    assert records[0].dense_vector == [0.1, 0.2, 0.3]
    assert records[1].dense_vector == [0.4, 0.5, 0.6]
    assert records[0].metadata["chunk_index"] == 0
    assert embedding.calls[0]["texts"] == ["first body", "second body"]
    assert any(stage["stage"] == "dense_encoder.success" for stage in trace.stages)


def test_encode_returns_empty_list_for_empty_chunks() -> None:
    embedding = FakeEmbedding()
    encoder = DenseEncoder(make_settings(), embedding=embedding)

    records = encoder.encode([])

    assert records == []
    assert embedding.calls == []


def test_encode_raises_when_provider_vector_count_mismatches() -> None:
    encoder = DenseEncoder(make_settings(), embedding=FakeEmbedding(vectors=[[0.1, 0.2]]))

    with pytest.raises(DenseEncoderError, match="expected 2 vectors, got 1"):
        encoder.encode(make_chunks())


def test_encode_raises_when_vector_dimensions_are_inconsistent() -> None:
    encoder = DenseEncoder(
        make_settings(),
        embedding=FakeEmbedding(vectors=[[0.1, 0.2], [0.3, 0.4, 0.5]]),
    )

    with pytest.raises(DenseEncoderError, match="inconsistent vector dimensions"):
        encoder.encode(make_chunks())
