"""IngestionPipeline 集成测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk, ChunkRecord, Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.loader.pdf_loader import PdfLoader
from libs.splitter.base_splitter import BaseSplitter
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorQueryResult


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x18\x8f"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeIntegrityChecker:
    def __init__(self) -> None:
        self.success_calls: list[dict[str, Any]] = []
        self.failed_calls: list[dict[str, Any]] = []

    def compute_sha256(self, path: str | Path) -> str:
        return "hash-123"

    def should_skip(self, file_hash: str) -> bool:
        return False

    def mark_success(self, file_hash: str, file_path: str | Path, **metadata: Any) -> None:
        self.success_calls.append({"file_hash": file_hash, "file_path": str(file_path), "metadata": metadata})

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        self.failed_calls.append({"file_hash": file_hash, "error_msg": error_msg})


class FakeLoader:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    def load(self, path: str | Path) -> Document:
        return Document(
            id="doc-abc",
            text="Architecture overview [IMAGE: img-1]\n\nDetailed retrieval flow.",
            metadata={
                "source_path": str(path),
                "images": [
                    {
                        "id": "img-1",
                        "path": str(self.image_path),
                        "page": 1,
                        "text_offset": 22,
                        "text_length": 14,
                        "position": {},
                    }
                ],
            },
        )


class FakeChunker:
    def split_document(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                id="chunk-0",
                text="Architecture overview [IMAGE: img-1]",
                metadata={
                    "source_path": document.metadata["source_path"],
                    "chunk_index": 0,
                    "images": [document.metadata["images"][0]],
                    "image_refs": ["img-1"],
                },
                start_offset=0,
                end_offset=35,
                source_ref=document.id,
            ),
            Chunk(
                id="chunk-1",
                text="Detailed retrieval flow.",
                metadata={
                    "source_path": document.metadata["source_path"],
                    "chunk_index": 1,
                },
                start_offset=37,
                end_offset=61,
                source_ref=document.id,
            ),
        ]


class PassThroughSplitter(BaseSplitter):
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        stripped = text.strip()
        return [stripped] if stripped else []


class IdentityTransform:
    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        return chunks


class FakeBatchProcessor:
    def process(self, chunks: list[Chunk], trace: Any | None = None) -> Any:
        dense_records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[float(index + 1), 0.5],
            )
            for index, chunk in enumerate(chunks)
        ]
        sparse_records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                sparse_vector={"architecture": 0.5} if index == 0 else {"retrieval": 0.8},
            )
            for index, chunk in enumerate(chunks)
        ]

        class _Result:
            def __init__(self, dense: list[ChunkRecord], sparse: list[ChunkRecord]) -> None:
                self.dense_records = dense
                self.sparse_records = sparse

        return _Result(dense_records, sparse_records)


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        super().__init__({})
        self.records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        self.records = list(records)
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
        vision_llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder", "collection": "default"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
        ingestion={},
    )


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample_documents"


def test_pipeline_runs_end_to_end_with_injected_dependencies(tmp_path: Path) -> None:
    source_pdf = tmp_path / "simple.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 fake")
    raw_image = tmp_path / "raw.png"
    raw_image.write_bytes(PNG_BYTES)

    integrity = FakeIntegrityChecker()
    vector_store = FakeVectorStore()
    image_storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "image_index.db")
    pipeline = IngestionPipeline(
        make_settings(),
        integrity_checker=integrity,
        loader=FakeLoader(raw_image),
        chunker=FakeChunker(),
        transforms=[IdentityTransform()],
        batch_processor=FakeBatchProcessor(),
        bm25_indexer=BM25Indexer(tmp_path / "db" / "bm25"),
        vector_upserter=VectorUpserter(make_settings(), vector_store=vector_store),
        image_storage=image_storage,
    )
    trace = TraceContext(trace_type="ingestion")

    result = pipeline.run(source_pdf, collection="manuals", trace=trace)

    assert result.skipped is False
    assert len(result.chunks) == 2
    assert len(result.vector_records) == 2
    assert (tmp_path / "db" / "bm25" / "bm25_index.pkl").exists()
    indexed_path = image_storage.get_image_path("img-1")
    assert indexed_path is not None and indexed_path.exists()
    assert len(vector_store.records) == 2
    assert vector_store.records[0].metadata["collection"] == "manuals"
    assert integrity.success_calls[0]["metadata"]["chunk_count"] == 2
    assert any(stage["stage"] == "pipeline.success" for stage in trace.stages)
    payload = trace.to_dict()
    assert payload["trace_type"] == "ingestion"
    stage_map = {stage["stage"]: stage for stage in payload["stages"]}
    for stage_name in ("load", "split", "transform", "embed", "upsert"):
        assert stage_name in stage_map
        assert stage_map[stage_name]["elapsed_ms"] >= 0
        assert stage_map[stage_name]["payload"]["method"]
    assert stage_map["load"]["payload"]["provider"] == "FakeLoader"
    assert stage_map["split"]["payload"]["method"] == "placeholder"
    assert stage_map["upsert"]["payload"]["method"] == "placeholder"


def test_pipeline_wraps_stage_failures_and_marks_failed(tmp_path: Path) -> None:
    class BrokenLoader:
        def load(self, path: str | Path) -> Document:
            raise RuntimeError("loader boom")

    source_pdf = tmp_path / "broken.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 fake")
    integrity = FakeIntegrityChecker()
    pipeline = IngestionPipeline(
        make_settings(),
        integrity_checker=integrity,
        loader=BrokenLoader(),
        chunker=FakeChunker(),
        transforms=[IdentityTransform()],
        batch_processor=FakeBatchProcessor(),
        bm25_indexer=BM25Indexer(tmp_path / "db" / "bm25"),
        vector_upserter=VectorUpserter(make_settings(), vector_store=FakeVectorStore()),
        image_storage=ImageStorage(tmp_path / "images", tmp_path / "db" / "image_index.db"),
    )

    with pytest.raises(IngestionPipelineError, match="loader.load failed: loader boom"):
        pipeline.run(source_pdf, collection="manuals")

    assert integrity.failed_calls[0]["file_hash"] == "hash-123"
    assert "loader.load failed: loader boom" in integrity.failed_calls[0]["error_msg"]


def test_pipeline_processes_repository_complex_fixture_with_real_pdf_loader(tmp_path: Path) -> None:
    source_pdf = FIXTURE_DIR / "complex_technical_doc.pdf"
    integrity = FakeIntegrityChecker()
    vector_store = FakeVectorStore()
    image_storage = ImageStorage(tmp_path / "images", tmp_path / "db" / "image_index.db")
    pipeline = IngestionPipeline(
        make_settings(),
        integrity_checker=integrity,
        loader=PdfLoader(image_root=tmp_path / "loader-images"),
        chunker=DocumentChunker(make_settings(), splitter=PassThroughSplitter({})),
        transforms=[IdentityTransform()],
        batch_processor=FakeBatchProcessor(),
        bm25_indexer=BM25Indexer(tmp_path / "db" / "bm25"),
        vector_upserter=VectorUpserter(make_settings(), vector_store=vector_store),
        image_storage=image_storage,
    )

    result = pipeline.run(source_pdf, collection="fixture-tech-docs", trace=TraceContext())

    assert result.skipped is False
    assert result.document is not None
    assert "## 1. System Overview" in result.document.text
    assert result.image_count == 3
    assert len(result.chunks) == 1
    assert len(result.vector_records) == 1
    stored_images = image_storage.list_images("fixture-tech-docs", result.document.id)
    assert len(stored_images) == 3
    assert all(Path(item["file_path"]).exists() for item in stored_images)
    assert (tmp_path / "db" / "bm25" / "bm25_index.pkl").exists()
