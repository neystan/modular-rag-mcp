"""VectorUpserter 单元测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import ChunkRecord
from ingestion.storage.vector_upserter import VectorUpserter, VectorUpserterError
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorQueryResult


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        super().__init__({})
        self.records: dict[str, VectorRecord] = {}
        self.calls: list[list[str]] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        self.calls.append([record.id for record in records])
        for record in records:
            self.records[record.id] = record
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        return []


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder", "collection": "test"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_record(text: str = "body", chunk_index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id=f"temp-{chunk_index}",
        text=text,
        metadata={"source_path": "docs/sample.pdf", "chunk_index": chunk_index},
        dense_vector=[0.1, 0.2, 0.3],
    )


def test_same_chunk_twice_produces_same_id(tmp_path: Any = None) -> None:
    vector_store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=vector_store)

    first = upserter.upsert([make_record("same text", 1)])[0]
    second = upserter.upsert([make_record("same text", 1)])[0]

    assert first.id == second.id
    assert len(vector_store.records) == 1


def test_content_change_produces_different_id() -> None:
    vector_store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=vector_store)

    first = upserter.upsert([make_record("body a", 1)])[0]
    second = upserter.upsert([make_record("body b", 1)])[0]

    assert first.id != second.id


def test_batch_upsert_preserves_order_and_records_trace() -> None:
    vector_store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=vector_store)
    trace = TraceContext()

    records = upserter.upsert([make_record("a", 0), make_record("b", 1), make_record("c", 2)], trace=trace)

    assert [record.metadata["chunk_index"] for record in records] == [0, 1, 2]
    assert [record.id for record in records] == vector_store.calls[0]
    assert all(record.metadata["chunk_id"] == record.id for record in records)
    assert any(stage["stage"] == "vector_upserter.success" for stage in trace.stages)


def test_missing_dense_vector_raises_error() -> None:
    vector_store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=vector_store)
    record = ChunkRecord(
        id="temp-1",
        text="body",
        metadata={"source_path": "docs/sample.pdf", "chunk_index": 1},
        dense_vector=None,
    )

    with pytest.raises(VectorUpserterError, match="missing dense_vector"):
        upserter.upsert([record])


def test_complex_metadata_is_sanitized_for_vector_store_only() -> None:
    vector_store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=vector_store)
    record = ChunkRecord(
        id="temp-complex",
        text="body",
        metadata={
            "source_path": "docs/sample.pdf",
            "chunk_index": 3,
            "image_refs": ["img-1", "img-2"],
            "images": [
                {
                    "id": "img-1",
                    "path": "data/images/img-1.png",
                    "page": 1,
                    "text_offset": 0,
                    "text_length": 14,
                    "position": {"x": 12},
                }
            ],
            "image_captions": [{"image_id": "img-1", "caption": "流程图"}],
            "extra": {"owner": "pipeline"},
        },
        dense_vector=[0.1, 0.2, 0.3],
    )

    result = upserter.upsert([record])[0]
    stored = next(iter(vector_store.records.values()))

    assert result.metadata["images"][0]["id"] == "img-1"
    assert result.metadata["image_captions"][0]["caption"] == "流程图"
    assert result.metadata["extra"]["owner"] == "pipeline"
    assert stored.metadata["image_refs"] == ["img-1", "img-2"]
    assert json.loads(stored.metadata["images"])[0]["id"] == "img-1"
    assert json.loads(stored.metadata["image_captions"])[0]["caption"] == "流程图"
    assert json.loads(stored.metadata["extra"])["owner"] == "pipeline"
