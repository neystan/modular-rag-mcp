"""文档生命周期管理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    source_path: str
    collection: str
    file_hash: str
    chunk_count: int
    image_count: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class DocumentDetail:
    source_path: str
    collection: str
    file_hash: str
    chunk_count: int
    image_count: int
    updated_at: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DeleteResult:
    source_path: str
    collection: str
    file_hash: str
    deleted_chunks: int
    deleted_images: int
    deleted_integrity_record: bool


@dataclass(frozen=True, slots=True)
class CollectionStats:
    collection: str
    document_count: int
    chunk_count: int
    image_count: int


class DocumentManager:
    """协调 Chroma、BM25、ImageStorage、FileIntegrity 的文档管理。"""

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityChecker,
    ) -> None:
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        records = self._list_integrity_records(collection)
        documents = [
            DocumentInfo(
                source_path=str(record.get("file_path", "")),
                collection=str(record.get("metadata", {}).get("collection", collection or "default")),
                file_hash=str(record.get("file_hash", "")),
                chunk_count=int(record.get("metadata", {}).get("chunk_count", 0)),
                image_count=int(record.get("metadata", {}).get("image_count", 0)),
                updated_at=str(record.get("updated_at", "")),
            )
            for record in records
            if str(record.get("file_path", "")).strip()
        ]
        return sorted(documents, key=lambda item: (item.collection, item.source_path))

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        source_path = self._require_non_empty_str(doc_id, "doc_id")
        record = self._find_integrity_record(source_path)
        if record is None:
            raise ValueError(f"document not found: {source_path}")

        metadata = dict(record.get("metadata", {}))
        collection = str(metadata.get("collection", "default"))
        file_hash = str(record.get("file_hash", ""))
        chunks = self.chroma_store.get_by_metadata({"source_path": source_path, "collection": collection})
        images = self.image_storage.list_images(collection, file_hash) if file_hash else []

        return DocumentDetail(
            source_path=source_path,
            collection=collection,
            file_hash=file_hash,
            chunk_count=len(chunks) or int(metadata.get("chunk_count", 0)),
            image_count=len(images) or int(metadata.get("image_count", 0)),
            updated_at=str(record.get("updated_at", "")),
            chunks=sorted(chunks, key=lambda item: int(item["metadata"].get("chunk_index", 0))),
            images=images,
        )

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        normalized_source = self._require_non_empty_str(source_path, "source_path")
        normalized_collection = self._require_non_empty_str(collection, "collection")
        record = self._find_integrity_record(normalized_source, normalized_collection)
        if record is None:
            raise ValueError(f"document not found: {normalized_source}")

        file_hash = str(record.get("file_hash", ""))
        deleted_chunks = self.chroma_store.delete_by_metadata(
            {"source_path": normalized_source, "collection": normalized_collection}
        )
        self.bm25_indexer.remove_document(normalized_source)
        deleted_images = self.image_storage.delete_images(normalized_collection, file_hash or None)
        deleted_integrity_record = False
        if isinstance(self.file_integrity, SQLiteIntegrityChecker) and file_hash:
            deleted_integrity_record = self.file_integrity.remove_record(file_hash)

        return DeleteResult(
            source_path=normalized_source,
            collection=normalized_collection,
            file_hash=file_hash,
            deleted_chunks=deleted_chunks,
            deleted_images=deleted_images,
            deleted_integrity_record=deleted_integrity_record,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        documents = self.list_documents(collection)
        return CollectionStats(
            collection=collection or "all",
            document_count=len(documents),
            chunk_count=sum(item.chunk_count for item in documents),
            image_count=sum(item.image_count for item in documents),
        )

    def _list_integrity_records(self, collection: str | None = None) -> list[dict[str, Any]]:
        if not isinstance(self.file_integrity, SQLiteIntegrityChecker):
            return []
        records = self.file_integrity.list_processed()
        if collection is None:
            return records
        return [
            record
            for record in records
            if str(record.get("metadata", {}).get("collection", "")).strip() == collection
        ]

    def _find_integrity_record(self, source_path: str, collection: str | None = None) -> dict[str, Any] | None:
        for record in self._list_integrity_records(collection):
            if str(record.get("file_path", "")).strip() == source_path:
                return record
        return None

    @staticmethod
    def _require_non_empty_str(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} is required")
        return value.strip()
