"""BM25Indexer 单元测试。"""

from __future__ import annotations

import math
from pathlib import Path

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer


def make_record(
    chunk_id: str,
    source_path: str,
    sparse_vector: dict[str, float],
    doc_length: int,
    collection: str = "",
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        text="placeholder text",
        metadata={"source_path": source_path, "sparse_doc_length": doc_length, "collection": collection},
        sparse_vector=sparse_vector,
    )


def test_build_load_and_query_roundtrip_returns_stable_top_ids(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    records = [
        make_record("chunk-a", "docs/a.pdf", {"alpha": 0.6, "beta": 0.2}, 3),
        make_record("chunk-b", "docs/b.pdf", {"alpha": 0.2, "gamma": 0.9}, 4),
        make_record("chunk-c", "docs/c.pdf", {"gamma": 0.5}, 2),
    ]

    indexer.build(records, rebuild=True)

    loaded = BM25Indexer(tmp_path)
    loaded.load()
    results = loaded.query("gamma alpha", top_k=3)

    assert [result.chunk_id for result in results] == ["chunk-b", "chunk-a", "chunk-c"]
    assert results[0].score >= results[1].score >= results[2].score


def test_idf_matches_spec_formula(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    records = [
        make_record("chunk-a", "docs/a.pdf", {"alpha": 0.6}, 3),
        make_record("chunk-b", "docs/b.pdf", {"alpha": 0.2}, 4),
        make_record("chunk-c", "docs/c.pdf", {"gamma": 0.5}, 2),
    ]

    indexer.build(records, rebuild=True)
    alpha_idf = indexer.index["alpha"]["idf"]
    expected = round(math.log((3 - 2 + 0.5) / (2 + 0.5)), 8)

    assert alpha_idf == expected


def test_build_supports_incremental_update_and_rebuild(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(
        [
            make_record("chunk-a", "docs/a.pdf", {"alpha": 0.6}, 3),
            make_record("chunk-b", "docs/b.pdf", {"beta": 0.4}, 2),
        ],
        rebuild=True,
    )
    indexer.build([make_record("chunk-c", "docs/c.pdf", {"alpha": 0.8}, 5)], rebuild=False)

    loaded = BM25Indexer(tmp_path)
    loaded.load()
    alpha_results = loaded.query("alpha", top_k=5)

    assert [result.chunk_id for result in alpha_results] == ["chunk-c", "chunk-a"]

    loaded.build([make_record("chunk-only", "docs/only.pdf", {"omega": 1.0}, 1)], rebuild=True)
    reloaded = BM25Indexer(tmp_path)
    reloaded.load()
    rebuilt_results = reloaded.query("omega", top_k=5)

    assert [result.chunk_id for result in rebuilt_results] == ["chunk-only"]
    assert reloaded.query("alpha", top_k=5) == []


def test_remove_document_rebuilds_index(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(
        [
            make_record("chunk-a", "docs/a.pdf", {"alpha": 0.6}, 3),
            make_record("chunk-b", "docs/b.pdf", {"beta": 0.4}, 2),
        ],
        rebuild=True,
    )

    indexer.remove_document("docs/a.pdf")
    reloaded = BM25Indexer(tmp_path)
    reloaded.load()

    assert reloaded.query("alpha", top_k=5) == []
    assert [result.chunk_id for result in reloaded.query("beta", top_k=5)] == ["chunk-b"]


def test_query_supports_metadata_filters(tmp_path: Path) -> None:
    indexer = BM25Indexer(tmp_path)
    indexer.build(
        [
            make_record("chunk-a", "docs/a.pdf", {"pdf": 0.6, "fixture": 0.2}, 3, collection="alpha"),
            make_record("chunk-b", "docs/b.pdf", {"pdf": 0.8, "fixture": 0.4}, 4, collection="beta"),
        ],
        rebuild=True,
    )

    loaded = BM25Indexer(tmp_path)
    loaded.load()

    alpha_results = loaded.query("pdf fixture", top_k=5, filters={"collection": "alpha"})
    beta_results = loaded.query("pdf fixture", top_k=5, filters={"collection": "beta"})

    assert [result.chunk_id for result in alpha_results] == ["chunk-a"]
    assert [result.chunk_id for result in beta_results] == ["chunk-b"]
