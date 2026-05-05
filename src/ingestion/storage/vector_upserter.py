"""向量存储写入与稳定 ID 生成。"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserterError(RuntimeError):
    """VectorUpserter 可读错误。"""


class VectorUpserter:
    """将 DenseEncoder 输出写入向量库，并生成稳定 chunk id。"""

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store or self._resolve_vector_store(settings)

    def upsert(self, records: list[ChunkRecord], trace: Any | None = None) -> list[ChunkRecord]:
        if not records:
            return []

        trace_context = trace if isinstance(trace, TraceContext) else None
        vector_records: list[VectorRecord] = []
        updated_records: list[ChunkRecord] = []

        for record in records:
            stable_id = self._generate_stable_id(record)
            metadata = copy.deepcopy(record.metadata)
            metadata["chunk_id"] = stable_id

            dense_vector = record.dense_vector
            if not dense_vector:
                raise VectorUpserterError(f"vector upserter input error: record {record.id} missing dense_vector")

            vector_records.append(
                VectorRecord(
                    id=stable_id,
                    vector=dense_vector,
                    text=record.text,
                    metadata=metadata,
                )
            )
            updated_records.append(
                ChunkRecord(
                    id=stable_id,
                    text=record.text,
                    metadata=metadata,
                    dense_vector=dense_vector,
                    sparse_vector=copy.deepcopy(record.sparse_vector),
                )
            )

        upserted = self.vector_store.upsert(vector_records, trace=trace_context)
        if upserted != len(vector_records):
            raise VectorUpserterError(
                f"vector upserter provider error: expected {len(vector_records)} upserts, got {upserted}"
            )

        if trace_context is not None:
            trace_context.record_stage("vector_upserter.success", {"count": upserted})
        return updated_records

    @staticmethod
    def _generate_stable_id(record: ChunkRecord) -> str:
        source_path = str(record.metadata.get("source_path", "")).strip()
        if not source_path:
            raise VectorUpserterError("vector upserter input error: metadata.source_path is required")

        chunk_index = record.metadata.get("chunk_index")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise VectorUpserterError("vector upserter input error: metadata.chunk_index must be non-negative int")

        content_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:8]
        raw = f"{source_path}:{chunk_index}:{content_hash}"
        stable_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"chunk_{stable_hash}"

    @staticmethod
    def _resolve_vector_store(settings: Settings | dict[str, Any]) -> BaseVectorStore:
        return VectorStoreFactory.create(settings)
