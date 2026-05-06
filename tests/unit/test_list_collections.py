"""list_collections 工具测试。"""

from __future__ import annotations

from pathlib import Path

from mcp_server.tools.list_collections import build_list_collections_tool, list_collections


def test_list_collections_returns_sorted_collection_names_and_counts(tmp_path: Path) -> None:
    documents_root = tmp_path / "data" / "documents"
    alpha = documents_root / "alpha"
    beta = documents_root / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "a.pdf").write_text("a", encoding="utf-8")
    (alpha / "nested").mkdir()
    (alpha / "nested" / "a-2.md").write_text("b", encoding="utf-8")
    (beta / "b.pdf").write_text("c", encoding="utf-8")

    payload = list_collections(documents_root)

    assert "可用集合" in payload["content"][0]["text"]
    assert payload["structuredContent"]["count"] == 2
    assert payload["structuredContent"]["collections"] == [
        {"name": "alpha", "path": str(alpha), "documentCount": 2},
        {"name": "beta", "path": str(beta), "documentCount": 1},
    ]


def test_list_collections_returns_friendly_empty_state(tmp_path: Path) -> None:
    payload = list_collections(tmp_path / "data" / "documents")

    assert "当前没有可用集合" in payload["content"][0]["text"]
    assert payload["structuredContent"]["collections"] == []


def test_tool_definition_wraps_list_collections_handler(tmp_path: Path) -> None:
    documents_root = tmp_path / "data" / "documents"
    manuals = documents_root / "manuals"
    manuals.mkdir(parents=True)
    (manuals / "guide.pdf").write_text("guide", encoding="utf-8")

    tool = build_list_collections_tool(documents_root)
    payload = tool.handler({})

    assert tool.name == "list_collections"
    assert payload["structuredContent"]["collections"][0]["name"] == "manuals"
