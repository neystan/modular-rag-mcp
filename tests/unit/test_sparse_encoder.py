"""SparseEncoder 单元测试。"""

from __future__ import annotations

import pytest

from core.trace import TraceContext
from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder


def make_chunk(text: str, chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_encode_builds_sparse_vector_and_metadata() -> None:
    encoder = SparseEncoder()
    trace = TraceContext()

    records = encoder.encode([make_chunk("BM25 retrieval retrieval test")], trace=trace)

    assert len(records) == 1
    assert records[0].sparse_vector is not None
    assert set(records[0].sparse_vector) == {"bm25", "retrieval", "test"}
    assert records[0].sparse_vector["retrieval"] > records[0].sparse_vector["bm25"]
    assert records[0].metadata["sparse_doc_length"] == 4
    assert records[0].metadata["sparse_unique_terms"] == 3
    assert any(stage["stage"] == "sparse_encoder.success" for stage in trace.stages)


def test_encode_handles_empty_text_with_empty_sparse_vector() -> None:
    encoder = SparseEncoder()

    records = encoder.encode([make_chunk("   ")])

    assert records[0].sparse_vector == {}
    assert records[0].metadata["sparse_doc_length"] == 0
    assert records[0].metadata["sparse_unique_terms"] == 0


def test_encode_supports_mixed_chinese_and_english_tokens() -> None:
    encoder = SparseEncoder()

    records = encoder.encode([make_chunk("Qwen 中文 检索 Qwen")])

    assert records[0].sparse_vector is not None
    assert "qwen" in records[0].sparse_vector
    assert "中文" in records[0].sparse_vector
    assert "检索" in records[0].sparse_vector


def test_encode_expands_long_chinese_runs_for_bm25_matching() -> None:
    encoder = SparseEncoder()

    records = encoder.encode([make_chunk("作者叫什么名字")])

    assert records[0].sparse_vector is not None
    assert "作者叫什么名字" in records[0].sparse_vector
    assert "作者" in records[0].sparse_vector
    assert "名字" in records[0].sparse_vector


def test_encode_requires_string_text() -> None:
    encoder = SparseEncoder()
    chunk = make_chunk("ok")
    chunk.text = None  # type: ignore[assignment]

    with pytest.raises(TypeError, match="text must be string"):
        encoder.encode([chunk])
