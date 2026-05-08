"""稀疏检索。"""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class SparseRetrieverError(RuntimeError):
    """SparseRetriever 可读错误。"""


class SparseRetriever:
    """基于 BM25 命中结果回查向量库中的 text/metadata。"""

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        keywords: list[str],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(keywords, list) or not [str(item).strip() for item in keywords if str(item).strip()]:
            raise SparseRetrieverError("sparse retriever input error: keywords must be non-empty list[str]")
        if not isinstance(top_k, int) or top_k <= 0:
            raise SparseRetrieverError("sparse retriever input error: top_k must be positive int")

        trace_context = trace if isinstance(trace, TraceContext) else None
        postings = self.bm25_indexer.query(keywords, top_k=top_k, filters=filters)
        if not postings:
            if trace_context is not None:
                trace_context.record_stage("sparse_retriever.retrieve", {"top_k": top_k, "result_count": 0})
            return []

        chunk_ids = [item.chunk_id for item in postings]
        payloads = self.vector_store.get_by_ids(chunk_ids, trace=trace_context)
        payload_by_id = {str(item.get("id", "")): item for item in payloads}

        results: list[RetrievalResult] = []
        for posting in postings:
            payload = payload_by_id.get(posting.chunk_id)
            if payload is None:
                continue

            text = payload.get("text")
            metadata = payload.get("metadata")
            if not isinstance(text, str) or not text:
                raise SparseRetrieverError(f"sparse retriever provider error: missing text for chunk {posting.chunk_id}")
            if not isinstance(metadata, dict):
                raise SparseRetrieverError(
                    f"sparse retriever provider error: invalid metadata for chunk {posting.chunk_id}"
                )

            results.append(
                RetrievalResult(
                    chunk_id=posting.chunk_id,
                    score=posting.score,
                    text=text,
                    metadata=metadata,
                )
            )

        if trace_context is not None:
            trace_context.record_stage(
                "sparse_retriever.retrieve",
                {"top_k": top_k, "result_count": len(results)},
            )
        return results
