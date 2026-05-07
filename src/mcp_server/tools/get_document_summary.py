"""文档摘要工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import chromadb

from core.settings import load_settings
from mcp_server.protocol_handler import ProtocolHandlerError, ToolDefinition


DocumentSummaryResolver = Callable[[str], dict[str, object] | None]


def build_get_document_summary_tool(
    resolver: DocumentSummaryResolver | None = None,
    settings_path: str | Path = "config/settings.yaml",
) -> ToolDefinition:
    """构建 get_document_summary 工具定义。"""

    def handler(arguments: dict[str, Any]) -> dict[str, object]:
        return get_document_summary(
            arguments.get("doc_id"),
            resolver=resolver,
            settings_path=settings_path,
        )

    return ToolDefinition(
        name="get_document_summary",
        description="按文档 ID 返回 title、summary、tags 等结构化摘要",
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "文档 ID；当前默认使用 source_path"},
            },
            "required": ["doc_id"],
        },
        handler=handler,
    )


def get_document_summary(
    doc_id: Any,
    *,
    resolver: DocumentSummaryResolver | None = None,
    settings_path: str | Path = "config/settings.yaml",
) -> dict[str, object]:
    """返回单个文档的结构化摘要。"""

    normalized_doc_id = _normalize_doc_id(doc_id)
    active_resolver = resolver or _build_default_resolver(settings_path)
    summary = active_resolver(normalized_doc_id)
    if summary is None:
        raise ProtocolHandlerError(f"document not found: {normalized_doc_id}")

    return {
        "content": [
            {
                "type": "text",
                "text": _build_markdown(summary),
            }
        ],
        "structuredContent": summary,
    }


def _build_default_resolver(settings_path: str | Path) -> DocumentSummaryResolver:
    settings = load_settings(settings_path)
    vector_store = settings.vector_store
    provider = str(vector_store.get("provider", "")).strip().lower()
    if provider != "chroma":
        raise ProtocolHandlerError(f"unsupported vector store provider for get_document_summary: {provider}")

    persist_path = str(vector_store.get("persist_path", "data/db/chroma"))
    collection_name = str(vector_store.get("collection", "default")).strip() or "default"
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(name=collection_name)

    def resolve(doc_id: str) -> dict[str, object] | None:
        payload = collection.get(
            where={"source_path": doc_id},
            limit=1,
            include=["documents", "metadatas"],
        )
        ids = payload.get("ids", [])
        if not ids:
            return None

        metadata = (payload.get("metadatas") or [{}])[0] or {}
        return {
            "doc_id": doc_id,
            "title": str(metadata.get("title", "")).strip() or Path(doc_id).stem,
            "summary": str(metadata.get("summary", "")).strip() or str((payload.get("documents") or [""])[0]).strip(),
            "tags": _normalize_tags(metadata.get("tags")),
            "source_path": str(metadata.get("source_path", doc_id)),
        }

    return resolve


def _build_markdown(summary: dict[str, object]) -> str:
    tags = summary.get("tags", [])
    tag_text = ", ".join(str(item) for item in tags) if isinstance(tags, list) and tags else "无"
    return "\n".join(
        [
            f"文档：{summary['doc_id']}",
            f"标题：{summary['title']}",
            f"摘要：{summary['summary']}",
            f"标签：{tag_text}",
        ]
    )


def _normalize_doc_id(doc_id: Any) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ProtocolHandlerError("doc_id is required")
    return doc_id.strip()


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
