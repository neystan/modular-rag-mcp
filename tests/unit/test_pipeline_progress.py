"""IngestionPipeline 进度回调测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.settings import Settings
from core.types import Chunk, ChunkRecord, Document
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, VectorQueryResult


class FakeIntegrityChecker:
    def compute_sha256(self, path: str | Path) -> str:
        return "hash-123"

    def should_skip(self, file_hash: str) -> bool:
        return False

    def mark_success(self, file_hash: str, file_path: str | Path, **metadata: Any) -> None:
        return None

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        return None


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
                text=document.text,
                metadata={
                    "source_path": document.metadata["source_path"],
                    "chunk_index": 0,
                    "images": document.metadata["images"],
                },
                start_offset=0,
                end_offset=len(document.text),
                source_ref=document.id,
            )
        ]


class IdentityTransform:
    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        return chunks


class FakeBatchProcessor:
    def process(self, chunks: list[Chunk], trace: Any | None = None) -> Any:
        dense_records = [
            ChunkRecord(id=chunk.id, text=chunk.text, metadata=dict(chunk.metadata), dense_vector=[1.0, 0.5])
            for chunk in chunks
        ]
        sparse_records = [
            ChunkRecord(id=chunk.id, text=chunk.text, metadata=dict(chunk.metadata), sparse_vector={"architecture": 0.5})
            for chunk in chunks
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


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x18\x8f"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def build_pipeline(tmp_path: Path) -> IngestionPipeline:
    vector_store = FakeVectorStore()
    return IngestionPipeline(
        make_settings(),
        integrity_checker=FakeIntegrityChecker(),
        loader=FakeLoader(tmp_path / "raw.png"),
        chunker=FakeChunker(),
        transforms=[IdentityTransform()],
        batch_processor=FakeBatchProcessor(),
        bm25_indexer=BM25Indexer(tmp_path / "db" / "bm25"),
        vector_upserter=VectorUpserter(make_settings(), vector_store=vector_store),
        image_storage=ImageStorage(tmp_path / "images", tmp_path / "db" / "image_index.db"),
    )


def test_pipeline_calls_progress_callback_for_each_stage(tmp_path: Path) -> None:
    source_pdf = tmp_path / "simple.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 fake")
    raw_image = tmp_path / "raw.png"
    raw_image.write_bytes(PNG_BYTES)
    pipeline = build_pipeline(tmp_path)
    progress_calls: list[tuple[str, int, int]] = []

    pipeline.run(
        source_pdf,
        collection="manuals",
        on_progress=lambda stage_name, current, total: progress_calls.append((stage_name, current, total)),
    )

    assert progress_calls == [
        ("load", 1, 5),
        ("split", 2, 5),
        ("transform", 3, 5),
        ("embed", 4, 5),
        ("upsert", 5, 5),
    ]


def test_pipeline_allows_missing_progress_callback(tmp_path: Path) -> None:
    source_pdf = tmp_path / "simple.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 fake")
    raw_image = tmp_path / "raw.png"
    raw_image.write_bytes(PNG_BYTES)
    pipeline = build_pipeline(tmp_path)

    result = pipeline.run(source_pdf, collection="manuals", on_progress=None)

    assert result.skipped is False
