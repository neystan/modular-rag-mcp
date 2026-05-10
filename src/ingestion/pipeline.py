"""数据摄取 Pipeline。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk, Document, ChunkRecord
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchEncodingResult, BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader import BaseLoader, PdfLoader, SQLiteIntegrityChecker
from libs.loader.file_integrity import FileIntegrityChecker


class IngestionPipelineError(RuntimeError):
    """IngestionPipeline 可读错误。"""


@dataclass(slots=True)
class IngestionPipelineResult:
    """Pipeline 运行结果。"""

    file_hash: str
    collection: str
    document: Document | None
    chunks: list[Chunk]
    dense_records: list[ChunkRecord]
    sparse_records: list[ChunkRecord]
    vector_records: list[ChunkRecord]
    image_count: int
    skipped: bool = False


class IngestionPipeline:
    """串行执行 integrity -> load -> split -> transform -> encode -> store。"""

    progress_stages = ("load", "split", "transform", "embed", "upsert")

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        *,
        integrity_checker: FileIntegrityChecker | None = None,
        loader: BaseLoader | None = None,
        chunker: DocumentChunker | None = None,
        transforms: list[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        bm25_indexer: BM25Indexer | None = None,
        vector_upserter: VectorUpserter | None = None,
        image_storage: ImageStorage | None = None,
    ) -> None:
        self.settings = settings
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self.loader = loader or PdfLoader()
        self.chunker = chunker or DocumentChunker(settings)
        self.transforms = transforms or [
            ChunkRefiner(settings),
            MetadataEnricher(settings),
            ImageCaptioner(settings),
        ]
        self.batch_processor = batch_processor or BatchProcessor(
            dense_encoder=DenseEncoder(settings),
            sparse_encoder=SparseEncoder(),
            batch_size=self._batch_size(settings),
        )
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.image_storage = image_storage or ImageStorage()

    def run(
        self,
        path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        trace: Any | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> IngestionPipelineResult:
        file_path = Path(path)
        trace_context = trace if isinstance(trace, TraceContext) else TraceContext(trace_type="ingestion")
        total_stages = len(self.progress_stages)
        file_hash = self._run_stage(
            "integrity.compute_hash",
            lambda: self.integrity_checker.compute_sha256(file_path),
            trace_context,
        )

        if not force and self.integrity_checker.should_skip(file_hash):
            trace_context.record_stage("pipeline.skip", {"file_hash": file_hash, "path": str(file_path)})
            return IngestionPipelineResult(
                file_hash=file_hash,
                collection=collection,
                document=None,
                chunks=[],
                dense_records=[],
                sparse_records=[],
                vector_records=[],
                image_count=0,
                skipped=True,
            )

        try:
            document = self._run_stage("loader.load", lambda: self.loader.load(file_path), trace_context)
            document = self._attach_collection(document, collection)
            self._record_pipeline_stage(
                trace_context,
                stage="load",
                method=self.loader.__class__.__name__.removesuffix("Loader").lower() or self.loader.__class__.__name__,
                provider=self.loader.__class__.__name__,
                details={
                    "document_id": document.id,
                    "text_length": len(document.text),
                    "image_count": len(document.metadata.get("images", [])),
                },
            )
            self._emit_progress(on_progress, "load", 1, total_stages)
            chunks = self._run_stage("chunker.split", lambda: self.chunker.split_document(document), trace_context)
            self._record_pipeline_stage(
                trace_context,
                stage="split",
                method=self._resolve_split_method(),
                provider=self.chunker.__class__.__name__,
                details={
                    "chunk_count": len(chunks),
                    "document_id": document.id,
                },
            )
            self._emit_progress(on_progress, "split", 2, total_stages)
            chunks = self._run_stage("transform.apply", lambda: self._apply_transforms(chunks, trace_context), trace_context)
            self._record_pipeline_stage(
                trace_context,
                stage="transform",
                method=self._resolve_transform_method(),
                provider=self._resolve_transform_provider(),
                details={
                    "chunk_count": len(chunks),
                    "transform_count": len(self.transforms),
                },
            )
            self._emit_progress(on_progress, "transform", 3, total_stages)
            chunks, image_count = self._run_stage(
                "image_storage.index",
                lambda: self._persist_images(chunks, collection, document.id),
                trace_context,
            )
            batch_result = self._run_stage(
                "embedding.batch",
                lambda: self.batch_processor.process(chunks, trace=trace_context),
                trace_context,
            )
            self._record_pipeline_stage(
                trace_context,
                stage="embed",
                method="batch_encode",
                provider=self.batch_processor.__class__.__name__,
                details={
                    "chunk_count": len(chunks),
                    "dense_record_count": len(batch_result.dense_records),
                    "sparse_record_count": len(batch_result.sparse_records),
                },
            )
            self._emit_progress(on_progress, "embed", 4, total_stages)
            vector_records = self._run_stage(
                "vector.upsert",
                lambda: self.vector_upserter.upsert(batch_result.dense_records, trace=trace_context),
                trace_context,
            )
            bm25_records = self._merge_sparse_vectors(
                vector_records, batch_result.sparse_records
            )
            self._run_stage(
                "bm25.build",
                lambda: self.bm25_indexer.build(bm25_records, rebuild=False),
                trace_context,
            )
            self._record_pipeline_stage(
                trace_context,
                stage="upsert",
                method=self._resolve_upsert_method(),
                provider=self.vector_upserter.__class__.__name__,
                details={
                    "vector_record_count": len(vector_records),
                    "image_count": image_count,
                    "collection": collection,
                },
            )
            self._emit_progress(on_progress, "upsert", 5, total_stages)
            self.integrity_checker.mark_success(
                file_hash,
                file_path,
                collection=collection,
                chunk_count=len(chunks),
                image_count=image_count,
            )
            trace_context.record_stage(
                "pipeline.success",
                {"chunks": len(chunks), "dense_records": len(batch_result.dense_records), "images": image_count},
            )
            return IngestionPipelineResult(
                file_hash=file_hash,
                collection=collection,
                document=document,
                chunks=chunks,
                dense_records=batch_result.dense_records,
                sparse_records=batch_result.sparse_records,
                vector_records=vector_records,
                image_count=image_count,
                skipped=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.integrity_checker.mark_failed(file_hash, str(exc))
            raise

    def _apply_transforms(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]:
        current = chunks
        for transform in self.transforms:
            current = transform.transform(current, trace=trace)
        return current

    def _persist_images(
        self,
        chunks: list[Chunk],
        collection: str,
        doc_hash: str,
    ) -> tuple[list[Chunk], int]:
        image_path_map: dict[str, str] = {}
        image_count = 0

        for chunk in chunks:
            for image in chunk.metadata.get("images", []):
                image_id = str(image.get("id", "")).strip()
                image_path = str(image.get("path", "")).strip()
                if not image_id or not image_path or image_id in image_path_map:
                    continue
                source_path = Path(image_path)
                if not source_path.exists():
                    raise IngestionPipelineError(f"pipeline image_storage.index failed: file not found: {source_path}")
                saved_path = self.image_storage.save_image(
                    image_id,
                    source_path.read_bytes(),
                    collection=collection,
                    doc_hash=doc_hash,
                    page_num=image.get("page"),
                    suffix=source_path.suffix or ".png",
                )
                image_path_map[image_id] = str(saved_path)
                image_count += 1

        updated_chunks: list[Chunk] = []
        for chunk in chunks:
            metadata = copy.deepcopy(chunk.metadata)
            metadata["collection"] = collection
            if "images" in metadata:
                for image in metadata["images"]:
                    image_id = str(image.get("id", "")).strip()
                    if image_id in image_path_map:
                        image["path"] = image_path_map[image_id]
            updated_chunks.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            )
        return updated_chunks, image_count

    @staticmethod
    def _attach_collection(document: Document, collection: str) -> Document:
        metadata = copy.deepcopy(document.metadata)
        metadata["collection"] = collection
        return Document(id=document.id, text=document.text, metadata=metadata)

    def _run_stage(self, stage: str, fn: Any, trace: TraceContext) -> Any:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            trace.record_stage(stage, {"status": "failed", "error": str(exc)})
            raise IngestionPipelineError(f"{stage} failed: {exc}") from exc
        trace.record_stage(stage, {"status": "ok"})
        return result

    def _resolve_split_method(self) -> str:
        if isinstance(self.settings, Settings):
            splitter = self.settings.splitter
        else:
            splitter = self.settings.get("splitter", {})
        if isinstance(splitter, dict):
            provider = splitter.get("provider")
            if isinstance(provider, str) and provider.strip():
                return provider.strip()
        return self.chunker.__class__.__name__.lower()

    def _resolve_transform_method(self) -> str:
        if not self.transforms:
            return "none"
        return "+".join(transform.__class__.__name__.lower() for transform in self.transforms)

    def _resolve_transform_provider(self) -> str:
        if not self.transforms:
            return "none"
        return ",".join(transform.__class__.__name__ for transform in self.transforms)

    def _resolve_upsert_method(self) -> str:
        if isinstance(self.settings, Settings):
            vector_store = self.settings.vector_store
        else:
            vector_store = self.settings.get("vector_store", {})
        if isinstance(vector_store, dict):
            provider = vector_store.get("provider")
            if isinstance(provider, str) and provider.strip():
                return provider.strip()
        return self.vector_upserter.__class__.__name__.lower()

    @staticmethod
    def _merge_sparse_vectors(
        vector_records: list[ChunkRecord],
        sparse_records: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        """将 vector_upsert 后的新 ID 与 sparse_records 的 sparse_vector 合并，供 BM25 构建。"""
        # dense_records 和 sparse_records 来自同一批 chunks，顺序一致，按索引对齐
        merged: list[ChunkRecord] = []
        for index, vr in enumerate(vector_records):
            sr = sparse_records[index] if index < len(sparse_records) else None
            merged.append(
                ChunkRecord(
                    id=vr.id,
                    text=vr.text,
                    metadata=copy.deepcopy(vr.metadata),
                    dense_vector=vr.dense_vector,
                    sparse_vector=copy.deepcopy(sr.sparse_vector) if sr and sr.sparse_vector else None,
                )
            )
        return merged

    @staticmethod
    def _record_pipeline_stage(
        trace: TraceContext,
        *,
        stage: str,
        method: str,
        provider: str,
        details: dict[str, Any],
    ) -> None:
        trace.record_stage(
            stage,
            {
                "method": method,
                "provider": provider,
                "details": details,
            },
        )

    @staticmethod
    def _emit_progress(
        on_progress: Callable[[str, int, int], None] | None,
        stage_name: str,
        current: int,
        total: int,
    ) -> None:
        if on_progress is None:
            return
        on_progress(stage_name, current, total)

    @staticmethod
    def _batch_size(settings: Settings | dict[str, Any]) -> int:
        if isinstance(settings, Settings):
            ingestion = settings.ingestion
        else:
            ingestion = settings.get("ingestion", {})
        value = ingestion.get("batch_processor", {}).get("batch_size", 8) if isinstance(ingestion, dict) else 8
        return value if isinstance(value, int) and value > 0 else 8
