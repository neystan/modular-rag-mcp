"""Dense/Sparse 编码批处理编排。"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from core.trace import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessorError(ValueError):
    """BatchProcessor 可读错误。"""


@dataclass(slots=True)
class BatchEncodingResult:
    """批处理后的双路编码结果。"""

    dense_records: list[ChunkRecord]
    sparse_records: list[ChunkRecord]


class BatchProcessor:
    """按批次驱动 dense/sparse 编码，保持输出顺序稳定。"""

    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 8,
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise BatchProcessorError("batch processor config error: batch_size must be positive int")
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(self, chunks: list[Chunk], trace: Any | None = None) -> BatchEncodingResult:
        if not chunks:
            return BatchEncodingResult(dense_records=[], sparse_records=[])

        trace_context = trace if isinstance(trace, TraceContext) else None
        dense_records: list[ChunkRecord] = []
        sparse_records: list[ChunkRecord] = []
        batch_count = ceil(len(chunks) / self.batch_size)

        for batch_index, start in enumerate(range(0, len(chunks), self.batch_size)):
            end = min(start + self.batch_size, len(chunks))
            batch = chunks[start:end]

            if trace_context is not None:
                trace_context.record_stage(
                    "batch_processor.batch_start",
                    {"batch_index": batch_index, "start": start, "end": end, "size": len(batch)},
                )

            dense_records.extend(self.dense_encoder.encode(batch, trace=trace_context))
            sparse_records.extend(self.sparse_encoder.encode(batch, trace=trace_context))

            if trace_context is not None:
                trace_context.record_stage(
                    "batch_processor.batch_end",
                    {"batch_index": batch_index, "start": start, "end": end, "size": len(batch)},
                )

        if trace_context is not None:
            trace_context.record_stage(
                "batch_processor.success",
                {"batch_count": batch_count, "count": len(chunks), "batch_size": self.batch_size},
            )

        return BatchEncodingResult(
            dense_records=dense_records,
            sparse_records=sparse_records,
        )
