"""Chunk 稠密向量编码。"""

from __future__ import annotations

import copy
from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoderError(RuntimeError):
    """DenseEncoder 可读错误。"""


class DenseEncoder:
    """调用 embedding provider，为 chunks 生成 ChunkRecord。"""

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        embedding: BaseEmbedding | None = None,
    ) -> None:
        self.settings = settings
        self.embedding = embedding or self._resolve_embedding(settings)

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        if not chunks:
            return []

        trace_context = trace if isinstance(trace, TraceContext) else None
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding.embed(texts, trace=trace_context)

        if len(vectors) != len(chunks):
            raise DenseEncoderError(
                f"dense encoder provider error: expected {len(chunks)} vectors, got {len(vectors)}"
            )

        dimension = len(vectors[0]) if vectors else 0
        records: list[ChunkRecord] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if len(vector) != dimension:
                raise DenseEncoderError(
                    "dense encoder provider error: inconsistent vector dimensions "
                    f"at index {index}: expected {dimension}, got {len(vector)}"
                )

            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=copy.deepcopy(chunk.metadata),
                    dense_vector=vector,
                )
            )

        if trace_context is not None:
            trace_context.record_stage(
                "dense_encoder.success",
                {"count": len(records), "dimension": dimension},
            )
        return records

    @staticmethod
    def _resolve_embedding(settings: Settings | dict[str, Any]) -> BaseEmbedding:
        return EmbeddingFactory.create(settings)
