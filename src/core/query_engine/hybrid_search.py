"""混合检索编排。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult


class HybridSearchError(RuntimeError):
    """HybridSearch 可读错误。"""


class HybridSearch:
    """编排 query 预处理、双路召回、融合与过滤。"""

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFFusion | None = None,
    ) -> None:
        self.settings = settings
        self.query_processor = query_processor or QueryProcessor(settings)
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever(settings)
        self.fusion = fusion or RRFFusion(settings)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(top_k, int) or top_k <= 0:
            raise HybridSearchError("hybrid search input error: top_k must be positive int")

        trace_context = trace if isinstance(trace, TraceContext) else None
        processed = self.query_processor.process(query, trace=trace_context)
        merged_filters = self._merge_filters(processed.filters, filters)

        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievers(
            processed.normalized_query,
            processed.keywords,
            top_k,
            merged_filters,
            trace_context,
        )

        if dense_error and sparse_error:
            raise HybridSearchError(
                "hybrid search failed: dense and sparse retrievers both failed: "
                f"dense={dense_error}; sparse={sparse_error}"
            )

        if dense_results and sparse_results:
            fused = self.fusion.fuse(dense_results, sparse_results, top_k=top_k, trace=trace_context)
        else:
            fused = dense_results or sparse_results

        filtered = self._apply_metadata_filters(fused, merged_filters)[:top_k]
        if trace_context is not None:
            trace_context.record_stage(
                "hybrid_search.search",
                {
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(filtered),
                    "dense_fallback": bool(dense_error),
                    "sparse_fallback": bool(sparse_error),
                },
            )
        return filtered

    def _run_retrievers(
        self,
        normalized_query: str,
        keywords: list[str],
        top_k: int,
        filters: dict[str, Any],
        trace: TraceContext | None,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], str | None, str | None]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.dense_retriever.retrieve,
                normalized_query,
                top_k,
                filters,
                trace,
            )
            sparse_future = executor.submit(
                self.sparse_retriever.retrieve,
                keywords,
                top_k,
                trace,
            )

            dense_results, dense_error = self._resolve_future(dense_future)
            sparse_results, sparse_error = self._resolve_future(sparse_future)
        return dense_results, sparse_results, dense_error, sparse_error

    @staticmethod
    def _resolve_future(future: Any) -> tuple[list[RetrievalResult], str | None]:
        try:
            return future.result(), None
        except Exception as exc:  # noqa: BLE001
            return [], str(exc)

    @staticmethod
    def _merge_filters(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(base)
        if isinstance(override, dict):
            merged.update(override)
        return merged

    @staticmethod
    def _apply_metadata_filters(
        candidates: list[RetrievalResult],
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        if not filters:
            return list(candidates)

        filtered: list[RetrievalResult] = []
        for item in candidates:
            if all(item.metadata.get(key) == value for key, value in filters.items()):
                filtered.append(item)
        return filtered
