"""稠密向量检索。"""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore, VectorQueryResult
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetrieverError(RuntimeError):
    """DenseRetriever 可读错误。"""


class DenseRetriever:
    """将 query 向量化后调用 VectorStore 执行语义召回。"""

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingFactory.create(settings)
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise DenseRetrieverError("dense retriever input error: query is required")
        if not isinstance(top_k, int) or top_k <= 0:
            raise DenseRetrieverError("dense retriever input error: top_k must be positive int")

        trace_context = trace if isinstance(trace, TraceContext) else None
        vectors = self.embedding_client.embed([query.strip()], trace=trace_context)
        if len(vectors) != 1:
            raise DenseRetrieverError(
                f"dense retriever provider error: expected 1 query vector, got {len(vectors)}"
            )

        results = self.vector_store.query(vectors[0], top_k=top_k, filters=filters, trace=trace_context)
        normalized = [self._to_retrieval_result(item) for item in results]

        if trace_context is not None:
            trace_context.record_stage(
                "dense_retriever.retrieve",
                {"top_k": top_k, "result_count": len(normalized)},
            )
        return normalized

    @staticmethod
    def _to_retrieval_result(item: VectorQueryResult) -> RetrievalResult:
        if not isinstance(item.text, str) or not item.text:
            raise DenseRetrieverError(f"dense retriever provider error: missing text for chunk {item.id}")
        if not isinstance(item.metadata, dict):
            raise DenseRetrieverError(f"dense retriever provider error: invalid metadata for chunk {item.id}")

        return RetrievalResult(
            chunk_id=str(item.id),
            score=float(item.score),
            text=item.text,
            metadata=item.metadata,
        )
