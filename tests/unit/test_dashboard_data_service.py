"""Dashboard 数据浏览服务测试。"""

from __future__ import annotations

from pathlib import Path

from core.settings import Settings
from core.types import ChunkRecord
from ingestion.document_manager import DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.pages.data_browser import collect_browser_data
from observability.dashboard.services.data_service import DataService


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x18\x8f"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp", "environment": "local"},
        llm={"provider": "placeholder"},
        vision_llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "chroma", "collection": "manuals", "persist_path": str(tmp_path / "chroma")},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
        ingestion={},
    )


def build_manager(tmp_path: Path) -> DocumentManager:
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
    return DocumentManager(chroma, bm25, image_storage, integrity)


def test_data_service_lists_collections_and_documents(tmp_path: Path) -> None:
    service = DataService(
        settings_path=tmp_path / "settings.yaml",
        document_manager=build_manager(tmp_path),
        settings_loader=lambda _: make_settings(tmp_path),
    )

    collections = service.list_collections()
    documents = service.list_documents("manuals")

    assert collections == ["manuals"]
    assert len(documents) == 1
    assert documents[0].source_path == "docs/sample.pdf"


def test_collect_browser_data_includes_selected_document_detail(tmp_path: Path) -> None:
    service = DataService(
        settings_path=tmp_path / "settings.yaml",
        document_manager=build_manager(tmp_path),
        settings_loader=lambda _: make_settings(tmp_path),
    )

    payload = collect_browser_data(service, collection="manuals")

    assert payload["collections"] == ["manuals"]
    assert payload["selected_doc_id"] == "docs/sample.pdf"
    assert payload["detail"] is not None
    assert payload["detail"].chunk_count == 2
    assert len(payload["detail"].images) == 1
