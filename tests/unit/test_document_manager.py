"""DocumentManager 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from core.types import ChunkRecord
from ingestion.document_manager import CollectionStats, DocumentDetail, DocumentInfo, DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x18\x8f"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def build_manager(tmp_path: Path) -> tuple[DocumentManager, SQLiteIntegrityChecker]:
    collection = "manuals"
    source_path = "docs/sample.pdf"
    file_hash = "a" * 64

    chroma = ChromaStore({"collection": collection, "persist_path": str(tmp_path / "chroma")})
    bm25 = BM25Indexer(tmp_path / "bm25")
    image_storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "image_index.db")
    integrity = SQLiteIntegrityChecker(tmp_path / "db" / "history.db")
    upserter = VectorUpserter(
        {
            "vector_store": {"provider": "chroma", "collection": collection, "persist_path": str(tmp_path / "chroma")},
        },
        vector_store=chroma,
    )
    records = [
        ChunkRecord(
            id="chunk-0",
            text="first chunk",
            metadata={
                "source_path": source_path,
                "collection": collection,
                "chunk_index": 0,
                "images": [
                    {
                        "id": f"{file_hash}_1_0",
                        "path": str(tmp_path / "raw-1.png"),
                        "page": 1,
                        "text_offset": 0,
                        "text_length": 14,
                        "position": {},
                    }
                ],
            },
            dense_vector=[1.0, 0.0],
            sparse_vector={"first": 0.7},
        ),
        ChunkRecord(
            id="chunk-1",
            text="second chunk",
            metadata={
                "source_path": source_path,
                "collection": collection,
                "chunk_index": 1,
            },
            dense_vector=[0.8, 0.2],
            sparse_vector={"second": 0.6},
        ),
    ]
    upserter.upsert(records)
    bm25.build(records, rebuild=True)
    image_storage.save_image(f"{file_hash}_1_0", PNG_BYTES, collection=collection, doc_hash=file_hash, page_num=1)
    integrity.mark_success(
        file_hash,
        source_path,
        collection=collection,
        chunk_count=2,
        image_count=1,
    )
    manager = DocumentManager(chroma, bm25, image_storage, integrity)
    return manager, integrity


def test_list_documents_returns_integrity_backed_document_rows(tmp_path: Path) -> None:
    manager, _ = build_manager(tmp_path)

    documents = manager.list_documents("manuals")

    assert len(documents) == 1
    assert isinstance(documents[0], DocumentInfo)
    assert documents[0].source_path == "docs/sample.pdf"
    assert documents[0].chunk_count == 2
    assert documents[0].image_count == 1


def test_get_document_detail_returns_chunks_and_images(tmp_path: Path) -> None:
    manager, _ = build_manager(tmp_path)

    detail = manager.get_document_detail("docs/sample.pdf")

    assert isinstance(detail, DocumentDetail)
    assert detail.collection == "manuals"
    assert detail.chunk_count == 2
    assert len(detail.chunks) == 2
    assert detail.chunks[0]["metadata"]["chunk_index"] == 0
    assert len(detail.images) == 1


def test_delete_document_removes_cross_store_artifacts(tmp_path: Path) -> None:
    manager, integrity = build_manager(tmp_path)

    result = manager.delete_document("docs/sample.pdf", "manuals")

    assert result.deleted_chunks == 2
    assert result.deleted_images == 1
    assert result.deleted_integrity_record is True
    assert manager.list_documents("manuals") == []
    assert manager.chroma_store.get_by_metadata({"source_path": "docs/sample.pdf", "collection": "manuals"}) == []
    reloaded_bm25 = BM25Indexer(tmp_path / "bm25")
    reloaded_bm25.load()
    assert reloaded_bm25.query("first", top_k=5) == []
    assert integrity.list_processed() == []


def test_get_collection_stats_aggregates_document_counts(tmp_path: Path) -> None:
    manager, _ = build_manager(tmp_path)

    stats = manager.get_collection_stats("manuals")

    assert isinstance(stats, CollectionStats)
    assert stats.collection == "manuals"
    assert stats.document_count == 1
    assert stats.chunk_count == 2
    assert stats.image_count == 1
