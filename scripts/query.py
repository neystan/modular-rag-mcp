"""在线查询入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import core.query_service as query_service
from core.query_engine.hybrid_search import HybridSearch, HybridSearchError
from core.query_engine.query_processor import ProcessedQuery
from core.query_service import QueryComponents, QueryExecution, build_components
from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25IndexerError


_DEFAULT_PERSIST_QUERY_TRACE = query_service._persist_query_trace


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="在线查询本地知识库")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=None, help="返回结果数量，默认读取配置 retrieval.top_k")
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
    top_k: int | None = None,
    collection: str | None = None,
    verbose: bool = False,
    no_rerank: bool = False,
    settings_path: str | Path = "config/settings.yaml",
    settings: Settings | None = None,
    components: QueryComponents | None = None,
) -> QueryExecution:
    query_service._persist_query_trace = _persist_query_trace
    return query_service.run_query(
        query,
        top_k=top_k,
        collection=collection,
        verbose=verbose,
        no_rerank=no_rerank,
        settings_path=settings_path,
        settings=settings,
        components=components,
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
    return query_service._normalize_top_k(top_k)


def _resolve_top_k(top_k: int | None, settings: Settings) -> int:
    return query_service._resolve_top_k(top_k, settings)


def _normalize_collection(collection: str | None) -> str | None:
    return query_service._normalize_collection(collection)


def _processed_query_from_trace(trace: TraceContext, query_text: str) -> ProcessedQuery:
    return query_service._processed_query_from_trace(trace, query_text)


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
    query_service._record_query_execution(
        trace,
        query_text=query_text,
        collection=collection,
        top_k=top_k,
        rerank_enabled=rerank_enabled,
        rerank_applied=rerank_applied,
        final_results=final_results,
    )


def _record_query_error(
    trace: TraceContext,
    *,
    query_text: str,
    collection: str | None,
    top_k: int,
) -> None:
    query_service._record_query_error(trace, query_text=query_text, collection=collection, top_k=top_k)


def _persist_query_trace(trace_dict: dict[str, Any]) -> None:
    _DEFAULT_PERSIST_QUERY_TRACE(trace_dict)


if __name__ == "__main__":
    raise SystemExit(main())
