"""知识库查询工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.query_engine.hybrid_search import HybridSearch
from core.response.multimodal_assembler import MultimodalAssembler
from core.query_engine.reranker import Reranker
from core.response.response_builder import ResponseBuilder
from core.settings import Settings, load_settings
from core.types import RetrievalResult
from mcp_server.protocol_handler import ProtocolHandlerError, ToolDefinition


ToolExecutor = Callable[[str, int, str | None], list[RetrievalResult]]


def build_query_knowledge_hub_tool(
    executor: ToolExecutor | None = None,
    response_builder: ResponseBuilder | None = None,
    multimodal_assembler: MultimodalAssembler | None = None,
    settings_path: str | Path = "config/settings.yaml",
) -> ToolDefinition:
    """构建 query_knowledge_hub 的工具定义。"""

    def handler(arguments: dict[str, Any]) -> dict[str, object]:
        return query_knowledge_hub(
            arguments.get("query"),
            top_k=arguments.get("top_k"),
            collection=arguments.get("collection"),
            executor=executor,
            response_builder=response_builder,
            multimodal_assembler=multimodal_assembler,
            settings_path=settings_path,
        )

    return ToolDefinition(
        name="query_knowledge_hub",
        description="查询知识库并返回带引用的 Markdown 结果",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户查询"},
                "top_k": {"type": "integer", "minimum": 1, "description": "返回结果数量"},
                "collection": {"type": "string", "description": "可选 collection 过滤"},
            },
            "required": ["query"],
        },
        handler=handler,
    )


def query_knowledge_hub(
    query: Any,
    *,
    top_k: Any = None,
    collection: Any = None,
    executor: ToolExecutor | None = None,
    response_builder: ResponseBuilder | None = None,
    multimodal_assembler: MultimodalAssembler | None = None,
    settings_path: str | Path = "config/settings.yaml",
) -> dict[str, object]:
    """执行知识库查询并返回 MCP Tool 结果。"""

    normalized_query = _normalize_query(query)
    normalized_collection = _normalize_collection(collection)
    settings = load_settings(settings_path)
    normalized_top_k = _resolve_top_k(top_k, settings)
    active_builder = response_builder or ResponseBuilder()
    active_assembler = multimodal_assembler or MultimodalAssembler()
    active_executor = executor or _build_default_executor(settings)

    retrieval_results = active_executor(normalized_query, normalized_top_k, normalized_collection)
    payload = active_builder.build(retrieval_results, normalized_query)
    image_contents = active_assembler.assemble(retrieval_results)
    if image_contents:
        payload["content"].extend(image_contents)
    return payload


def _build_default_executor(settings: Settings) -> ToolExecutor:
    hybrid_search = HybridSearch(settings)
    reranker = Reranker(settings)

    def execute(query: str, top_k: int, collection: str | None) -> list[RetrievalResult]:
        filters = {"collection": collection} if collection else None
        results = hybrid_search.search(query, top_k=top_k, filters=filters)
        return reranker.rerank(query, results, top_k=top_k)

    return execute


def _normalize_query(query: Any) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ProtocolHandlerError("query is required")
    return query.strip()


def _normalize_top_k(top_k: Any) -> int:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ProtocolHandlerError("top_k must be positive int")
    return top_k


def _resolve_top_k(top_k: Any, settings: Settings) -> int:
    if top_k is not None:
        return _normalize_top_k(top_k)
    return _normalize_top_k(settings.retrieval.get("top_k"))


def _normalize_collection(collection: Any) -> str | None:
    if collection is None:
        return None
    if not isinstance(collection, str):
        raise ProtocolHandlerError("collection must be string")
    normalized = collection.strip()
    return normalized or None
