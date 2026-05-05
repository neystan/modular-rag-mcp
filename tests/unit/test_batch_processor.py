"""BatchProcessor 单元测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.trace import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor, BatchProcessorError


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[float(index), float(len(chunk.text))],
            )
            for index, chunk in enumerate(chunks)
        ]


class FakeSparseEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                sparse_vector={chunk.id: 1.0},
            )
            for chunk in chunks
        ]


def make_chunks(count: int = 5) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{index}",
            text=f"body {index}",
            metadata={"source_path": "docs/sample.pdf", "chunk_index": index},
            start_offset=index * 10,
            end_offset=index * 10 + 6,
            source_ref="doc-1",
        )
        for index in range(count)
    ]


def test_process_splits_into_stable_batches() -> None:
    dense_encoder = FakeDenseEncoder()
    sparse_encoder = FakeSparseEncoder()
    processor = BatchProcessor(dense_encoder=dense_encoder, sparse_encoder=sparse_encoder, batch_size=2)
    trace = TraceContext()

    result = processor.process(make_chunks(5), trace=trace)

    assert dense_encoder.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]
    assert sparse_encoder.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]
    assert [record.id for record in result.dense_records] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
    ]
    assert [record.id for record in result.sparse_records] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
    ]
    assert any(stage["stage"] == "batch_processor.success" for stage in trace.stages)


def test_process_returns_empty_results_for_empty_input() -> None:
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2,
    )

    result = processor.process([])

    assert result.dense_records == []
    assert result.sparse_records == []


def test_batch_size_must_be_positive_int() -> None:
    with pytest.raises(BatchProcessorError, match="batch_size must be positive int"):
        BatchProcessor(
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder(),
            batch_size=0,
        )
