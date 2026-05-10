"""查询编排共享服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from core.query_engine.reranker import Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings, load_settings
from core.trace import TraceCollector, TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.logger import DEFAULT_TRACE_LOG_PATH, write_trace


@dataclass(slots=True)
class QueryExecution:
    """查询执行结果。"""

    processed_query: ProcessedQuery
    dense_results: list[RetrievalResult] = field(default_factory=list)
    sparse_results: list[RetrievalResult] = field(default_factory=list)
    fusion_results: list[RetrievalResult] = field(default_factory=list)
    final_results: list[RetrievalResult] = field(default_factory=list)
    rerank_enabled: bool = True
    rerank_applied: bool = False


@dataclass(slots=True)
class QueryComponents:
    """查询所需组件集合。"""

    query_processor: QueryProcessor
    dense_retriever: DenseRetriever
    sparse_retriever: SparseRetriever
    fusion: RRFFusion
    hybrid_search: HybridSearch
    reranker: Reranker


def run_query(
    query: str,
    *,
    top_k: int | None = None,
    collection: str | None = None,
    verbose: bool = False,
    no_rerank: bool = False,
    settings_path: str | Path = "config/settings.yaml",
    settings: Settings | None = None,
    components: QueryComponents | None = None,
) -> QueryExecution:
    """执行一次在线查询。"""

    normalized_query = _normalize_query(query)
    normalized_collection = _normalize_collection(collection)
    active_settings = settings or load_settings(settings_path)
    normalized_top_k = _resolve_top_k(top_k, active_settings)
    active_components = components or build_components(active_settings)

    filters = {"collection": normalized_collection} if normalized_collection else None
    trace = TraceContext(trace_type="query")
    merged_filters = filters or {}

    try:
        if verbose:
            processed = active_components.query_processor.process(normalized_query, trace=trace)
            merged_filters = dict(processed.filters)
            if filters:
                merged_filters.update(filters)
            dense_results = active_components.dense_retriever.retrieve(
                processed.normalized_query,
                normalized_top_k,
                filters=merged_filters,
                trace=trace,
            )
            sparse_results = active_components.sparse_retriever.retrieve(
                processed.keywords,
                normalized_top_k,
                filters=merged_filters,
                trace=trace,
            )
            fusion_results = active_components.fusion.fuse(
                dense_results,
                sparse_results,
                top_k=normalized_top_k,
                trace=trace,
            )
            fusion_results = active_components.hybrid_search._apply_metadata_filters(
                fusion_results,
                merged_filters,
            )[:normalized_top_k]
            trace.record_stage(
                "hybrid_search.search",
                {
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(fusion_results),
                    "chunk_ids": [item.chunk_id for item in fusion_results],
                    "dense_fallback": False,
                    "sparse_fallback": False,
                },
            )
        else:
            dense_results = []
            sparse_results = []
            fusion_results = active_components.hybrid_search.search(
                normalized_query,
                top_k=normalized_top_k,
                filters=merged_filters or None,
                trace=trace,
            )
            processed = _processed_query_from_trace(trace, normalized_query)

        final_results = fusion_results
        rerank_applied = False
        if not no_rerank:
            final_results = active_components.reranker.rerank(
                normalized_query,
                fusion_results,
                top_k=normalized_top_k,
                trace=trace,
            )
            rerank_applied = True

        _record_query_execution(
            trace,
            query_text=normalized_query,
            collection=normalized_collection,
            top_k=normalized_top_k,
            rerank_enabled=not no_rerank,
            rerank_applied=rerank_applied,
            final_results=final_results,
        )
        TraceCollector(persister=_persist_query_trace).collect(trace)
        return QueryExecution(
            processed_query=processed,
            dense_results=dense_results,
            sparse_results=sparse_results,
            fusion_results=fusion_results,
            final_results=final_results,
            rerank_enabled=not no_rerank,
            rerank_applied=rerank_applied,
        )
    except Exception:
        _record_query_error(trace, query_text=normalized_query, collection=normalized_collection, top_k=normalized_top_k)
        TraceCollector(persister=_persist_query_trace).collect(trace)
        raise


def build_components(settings: Settings) -> QueryComponents:
    """基于配置创建查询链路组件。"""

    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings)
    bm25_indexer = BM25Indexer()
    try:
        bm25_indexer.load()
    except BM25IndexerError as exc:
        if "file not found" not in str(exc):
            raise

    query_processor = QueryProcessor(settings)
    dense_retriever = DenseRetriever(settings, embedding_client=embedding_client, vector_store=vector_store)
    sparse_retriever = SparseRetriever(settings, bm25_indexer=bm25_indexer, vector_store=vector_store)
    fusion = RRFFusion(settings)
    hybrid_search = HybridSearch(
        settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )
    reranker = Reranker(settings)
    return QueryComponents(
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
        hybrid_search=hybrid_search,
        reranker=reranker,
    )


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    return query.strip()


def _normalize_top_k(top_k: int) -> int:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be positive int")
    return top_k


def _resolve_top_k(top_k: int | None, settings: Settings) -> int:
    if top_k is not None:
        return _normalize_top_k(top_k)
    return _normalize_top_k(settings.retrieval.get("top_k"))


def _normalize_collection(collection: str | None) -> str | None:
    if collection is None:
        return None
    normalized = str(collection).strip()
    return normalized or None


def _processed_query_from_trace(trace: TraceContext, query_text: str) -> ProcessedQuery:
    for item in reversed(trace.stages):
        if item["stage"] != "query_processing":
            continue
        payload = item.get("payload", {})
        details = payload.get("details", {}) if isinstance(payload, dict) else {}
        keywords = details.get("keywords", []) if isinstance(details, dict) else []
        filters = details.get("filters", {}) if isinstance(details, dict) else {}
        return ProcessedQuery(
            original_query=query_text,
            normalized_query=str(details.get("query_text", query_text)),
            keywords=[str(keyword) for keyword in keywords if str(keyword).strip()],
            filters=dict(filters) if isinstance(filters, dict) else {},
        )
    return ProcessedQuery(original_query=query_text, normalized_query=query_text, keywords=[], filters={})


def _record_query_execution(
    trace: TraceContext,
    *,
    query_text: str,
    collection: str | None,
    top_k: int,
    rerank_enabled: bool,
    rerank_applied: bool,
    final_results: list[RetrievalResult],
) -> None:
    trace.record_stage(
        "query.execution",
        {
            "query_text": query_text,
            "collection": collection,
            "top_k": top_k,
            "rerank_enabled": rerank_enabled,
            "rerank_applied": rerank_applied,
            "final_ids": [item.chunk_id for item in final_results],
            "result_count": len(final_results),
        },
    )


def _record_query_error(
    trace: TraceContext,
    *,
    query_text: str,
    collection: str | None,
    top_k: int,
) -> None:
    trace.record_stage(
        "query.error",
        {
            "query_text": query_text,
            "collection": collection,
            "top_k": top_k,
        },
    )


def _persist_query_trace(trace_dict: dict[str, Any]) -> None:
    write_trace(trace_dict, log_path=DEFAULT_TRACE_LOG_PATH)
