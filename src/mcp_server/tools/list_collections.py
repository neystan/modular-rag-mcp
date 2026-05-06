"""集合列表工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_server.protocol_handler import ToolDefinition


DEFAULT_DOCUMENTS_ROOT = Path("data/documents")


def build_list_collections_tool(documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT) -> ToolDefinition:
    """构建 list_collections 工具定义。"""

    def handler(arguments: dict[str, Any]) -> dict[str, object]:
        _ = arguments
        return list_collections(documents_root=documents_root)

    return ToolDefinition(
        name="list_collections",
        description="列出知识库中可用的文档集合",
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=handler,
    )


def list_collections(documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT) -> dict[str, object]:
    """扫描本地目录并返回集合列表。"""

    root = Path(documents_root)
    collections = _scan_collections(root)
    if not collections:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "当前没有可用集合，请先运行 ingest.py 摄取文档。",
                }
            ],
            "structuredContent": {
                "collections": [],
                "count": 0,
            },
        }

    lines = ["可用集合：", ""]
    for index, item in enumerate(collections, start=1):
        lines.append(f"{index}. {item['name']} ({item['documentCount']} documents)")

    return {
        "content": [
            {
                "type": "text",
                "text": "\n".join(lines),
            }
        ],
        "structuredContent": {
            "collections": collections,
            "count": len(collections),
        },
    }


def _scan_collections(root: Path) -> list[dict[str, object]]:
    if not root.exists() or not root.is_dir():
        return []

    collections: list[dict[str, object]] = []
    for item in sorted(root.iterdir(), key=lambda path: path.name):
        if not item.is_dir():
            continue
        document_count = sum(1 for child in item.rglob("*") if child.is_file())
        collections.append(
            {
                "name": item.name,
                "path": str(item),
                "documentCount": document_count,
            }
        )
    return collections
