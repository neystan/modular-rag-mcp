"""Chunk 稀疏向量编码。"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any

from core.trace import TraceContext
from core.types import Chunk, ChunkRecord


class SparseEncoder:
    """将 chunk 文本转换为 BM25 可消费的 term weight 结构。"""

    token_pattern = re.compile(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9][A-Za-z0-9_-]*")

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        trace_context = trace if isinstance(trace, TraceContext) else None

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            term_counts = Counter(tokens)
            sparse_vector = self._build_sparse_vector(term_counts)
            metadata = copy.deepcopy(chunk.metadata)
            metadata["sparse_doc_length"] = len(tokens)
            metadata["sparse_unique_terms"] = len(term_counts)

            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    sparse_vector=sparse_vector,
                )
            )

        if trace_context is not None:
            trace_context.record_stage(
                "sparse_encoder.success",
                {"count": len(records), "non_empty": sum(1 for record in records if record.sparse_vector)},
            )
        return records

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("sparse encoder input error: text must be string")
        return [token.lower() for token in cls.token_pattern.findall(text)]

    @staticmethod
    def _build_sparse_vector(term_counts: Counter[str]) -> dict[str, float]:
        if not term_counts:
            return {}

        total_terms = sum(term_counts.values())
        weights: dict[str, float] = {}
        for term, count in term_counts.items():
            # 使用子线性 TF，便于后续 BM25Indexer 直接消费或再映射。
            weights[term] = round((1.0 + math.log(count)) / total_terms, 8)
        return weights
