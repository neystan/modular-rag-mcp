"""Dashboard Ingestion 管理页测试。"""

from __future__ import annotations

from pathlib import Path

from core.settings import Settings
from core.types import ChunkRecord
from ingestion.document_manager import DocumentManager
from ingestion.pipeline import IngestionPipelineResult
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.pages.ingestion_manager import collect_ingestion_data, ingest_uploaded_file
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


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        trace: object | None = None,
        on_progress: object | None = None,
    ) -> IngestionPipelineResult:
        temp_path = Path(path)
        self.calls.append(
            {
                "path": temp_path,
                "collection": collection,
                "force": force,
                "exists_during_run": temp_path.exists(),
                "trace": trace,
            }
        )
        if callable(on_progress):
            on_progress("load", 1, 5)
            on_progress("upsert", 5, 5)
        return IngestionPipelineResult(
            file_hash="f" * 64,
            collection=collection,
            document=None,
            chunks=[],
            dense_records=[],
            sparse_records=[],
            vector_records=[],
            image_count=0,
            skipped=False,
        )


def test_collect_ingestion_data_includes_stats(tmp_path: Path) -> None:
    service = DataService(
        settings_path=tmp_path / "settings.yaml",
        document_manager=build_manager(tmp_path),
        settings_loader=lambda _: make_settings(tmp_path),
    )

    payload = collect_ingestion_data(service, collection="manuals")

    assert payload["collections"] == ["manuals"]
    assert len(payload["documents"]) == 1
    assert payload["stats"].document_count == 1
    assert payload["stats"].chunk_count == 2


def test_ingest_uploaded_file_passes_progress_and_cleans_temp_file() -> None:
    pipeline = FakePipeline()
    progress_calls: list[tuple[str, int, int]] = []
    persisted_traces: list[dict[str, object]] = []

    result = ingest_uploaded_file(
        "sample.pdf",
        b"%PDF-1.4 sample",
        "manuals",
        pipeline=pipeline,
        force=True,
        progress_callback=lambda stage, current, total: progress_calls.append((stage, current, total)),
        trace_persister=persisted_traces.append,
    )

    assert result.collection == "manuals"
    assert progress_calls == [("load", 1, 5), ("upsert", 5, 5)]
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["collection"] == "manuals"
    assert pipeline.calls[0]["force"] is True
    assert pipeline.calls[0]["exists_during_run"] is True
    assert pipeline.calls[0]["trace"] is not None
    assert not Path(pipeline.calls[0]["path"]).exists()
    assert len(persisted_traces) == 1
    assert persisted_traces[0]["trace_type"] == "ingestion"
    assert any(
        stage["stage"] == "dashboard.upload" and stage["payload"]["original_filename"] == "sample.pdf"
        for stage in persisted_traces[0]["stages"]
    )


def test_ingest_uploaded_file_persists_error_trace(tmp_path: Path) -> None:
    class FailingPipeline(FakePipeline):
        def run(
            self,
            path: str | Path,
            *,
            collection: str = "default",
            force: bool = False,
            trace: object | None = None,
            on_progress: object | None = None,
        ) -> IngestionPipelineResult:
            raise RuntimeError("boom")

    persisted_traces: list[dict[str, object]] = []

    try:
        ingest_uploaded_file(
            "broken.pdf",
            b"%PDF-1.4 broken",
            "manuals",
            pipeline=FailingPipeline(),
            trace_persister=persisted_traces.append,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert len(persisted_traces) == 1
    assert any(stage["stage"] == "dashboard.error" for stage in persisted_traces[0]["stages"])


def test_data_service_delete_document_updates_listing(tmp_path: Path) -> None:
    service = DataService(
        settings_path=tmp_path / "settings.yaml",
        document_manager=build_manager(tmp_path),
        settings_loader=lambda _: make_settings(tmp_path),
    )

    result = service.delete_document("docs/sample.pdf", "manuals")

    assert result.deleted_chunks == 2
    assert service.list_documents("manuals") == []
