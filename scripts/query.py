"""在线查询入口。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearch, HybridSearchError
from core.query_engine.query_processor import ProcessedQuery, QueryProcessor
from core.query_engine.reranker import Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings, load_settings
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory


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


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="在线查询本地知识库")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数量，默认 10")
    parser.add_argument("--collection", help="按 collection 过滤结果")
    parser.add_argument("--verbose", action="store_true", help="显示各阶段中间结果")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 reranker 阶段")
    parser.add_argument("--settings", default="config/settings.yaml", help="配置文件路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        execution = run_query(
            args.query,
            top_k=args.top_k,
            collection=args.collection,
            verbose=args.verbose,
            no_rerank=args.no_rerank,
            settings_path=args.settings,
        )
    except (ValueError, HybridSearchError, BM25IndexerError) as exc:
        print(f"query failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"query failed: unexpected error: {exc}", file=sys.stderr)
        return 1

    print(render_execution(execution, verbose=args.verbose))
    return 0


def run_query(
    query: str,
    *,
    top_k: int = 10,
    collection: str | None = None,
    verbose: bool = False,
    no_rerank: bool = False,
    settings_path: str | Path = "config/settings.yaml",
    settings: Settings | None = None,
    components: QueryComponents | None = None,
) -> QueryExecution:
    """执行一次在线查询。"""

    normalized_query = _normalize_query(query)
    normalized_top_k = _normalize_top_k(top_k)
    normalized_collection = _normalize_collection(collection)
    active_settings = settings or load_settings(settings_path)
    active_components = components or build_components(active_settings)

    filters = {"collection": normalized_collection} if normalized_collection else None
    processed = active_components.query_processor.process(normalized_query)

    if verbose:
        dense_results = active_components.dense_retriever.retrieve(
            processed.normalized_query,
            normalized_top_k,
            filters=processed.filters,
        )
        sparse_results = active_components.sparse_retriever.retrieve(processed.keywords, normalized_top_k)
        fusion_results = active_components.fusion.fuse(dense_results, sparse_results, top_k=normalized_top_k)
        fusion_results = active_components.hybrid_search._apply_metadata_filters(fusion_results, filters)[:normalized_top_k]
    else:
        dense_results = []
        sparse_results = []
        fusion_results = active_components.hybrid_search.search(
            normalized_query,
            top_k=normalized_top_k,
            filters=filters,
        )

    final_results = fusion_results
    rerank_applied = False
    if not no_rerank:
        final_results = active_components.reranker.rerank(
            normalized_query,
            fusion_results,
            top_k=normalized_top_k,
        )
        rerank_applied = True

    return QueryExecution(
        processed_query=processed,
        dense_results=dense_results,
        sparse_results=sparse_results,
        fusion_results=fusion_results,
        final_results=final_results,
        rerank_enabled=not no_rerank,
        rerank_applied=rerank_applied,
    )


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


def render_execution(execution: QueryExecution, *, verbose: bool = False) -> str:
    """将执行结果渲染为 CLI 文本。"""

    if not execution.final_results:
        return "未找到相关文档，请先运行 ingest.py 摄取数据。"

    lines: list[str] = []
    if verbose:
        lines.append(f"query: {execution.processed_query.normalized_query}")
        lines.append(f"keywords: {', '.join(execution.processed_query.keywords) or '<none>'}")
        lines.append("")
        lines.append("Dense Results:")
        lines.extend(_format_results(execution.dense_results))
        lines.append("")
        lines.append("Sparse Results:")
        lines.extend(_format_results(execution.sparse_results))
        lines.append("")
        lines.append("Fusion Results:")
        lines.extend(_format_results(execution.fusion_results))
        lines.append("")
        lines.append("Rerank Results:" if execution.rerank_enabled else "Final Results (No Rerank):")
        lines.extend(_format_results(execution.final_results))
        return "\n".join(lines)

    return "\n".join(_format_results(execution.final_results))


def _format_results(results: list[RetrievalResult]) -> list[str]:
    if not results:
        return ["<empty>"]

    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        source = str(item.metadata.get("source_path", "<unknown>"))
        page = item.metadata.get("page")
        page_text = f" page={page}" if page is not None else ""
        snippet = _summarize_text(item.text)
        lines.append(f"{index}. score={item.score:.4f} source={source}{page_text}")
        lines.append(f"   {snippet}")
    return lines


def _summarize_text(text: str, limit: int = 120) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 1)].rstrip() + "..."


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    return query.strip()


def _normalize_top_k(top_k: int) -> int:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be positive int")
    return top_k


def _normalize_collection(collection: str | None) -> str | None:
    if collection is None:
        return None
    normalized = str(collection).strip()
    return normalized or None


if __name__ == "__main__":
    raise SystemExit(main())
