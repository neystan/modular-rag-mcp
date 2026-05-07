"""get_document_summary 工具测试。"""

from __future__ import annotations

import pytest

from mcp_server.protocol_handler import ProtocolHandlerError
from mcp_server.tools.get_document_summary import build_get_document_summary_tool, get_document_summary


def test_get_document_summary_returns_structured_payload() -> None:
    def resolver(doc_id: str) -> dict[str, object] | None:
        assert doc_id == "docs/azure.pdf"
        return {
            "doc_id": doc_id,
            "title": "Azure Guide",
            "summary": "How to configure Azure OpenAI.",
            "tags": ["azure", "openai"],
            "source_path": doc_id,
        }

    payload = get_document_summary("docs/azure.pdf", resolver=resolver)

    assert "Azure Guide" in payload["content"][0]["text"]
    assert payload["structuredContent"]["title"] == "Azure Guide"
    assert payload["structuredContent"]["tags"] == ["azure", "openai"]


def test_get_document_summary_raises_for_missing_document() -> None:
    with pytest.raises(ProtocolHandlerError, match="document not found: missing.pdf"):
        get_document_summary("missing.pdf", resolver=lambda doc_id: None)


def test_get_document_summary_rejects_blank_doc_id() -> None:
    with pytest.raises(ProtocolHandlerError, match="doc_id is required"):
        get_document_summary("   ", resolver=lambda doc_id: None)


def test_tool_definition_wraps_summary_handler() -> None:
    tool = build_get_document_summary_tool(
        resolver=lambda doc_id: {
            "doc_id": doc_id,
            "title": "Title",
            "summary": "Summary",
            "tags": ["a"],
            "source_path": doc_id,
        }
    )

    payload = tool.handler({"doc_id": "docs/test.pdf"})

    assert tool.name == "get_document_summary"
    assert payload["structuredContent"]["doc_id"] == "docs/test.pdf"
