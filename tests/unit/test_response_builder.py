"""ResponseBuilder 单元测试。"""

from __future__ import annotations

from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def make_result(chunk_id: str, *, page: int | None = None) -> RetrievalResult:
    metadata: dict[str, object] = {"source_path": f"docs/{chunk_id}.pdf"}
    if page is not None:
        metadata["page"] = page
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.88,
        text=f"content for {chunk_id}",
        metadata=metadata,
    )


def test_response_builder_builds_markdown_and_citations() -> None:
    builder = ResponseBuilder()

    payload = builder.build([make_result("chunk-a", page=3), make_result("chunk-b")], "azure")

    assert payload["content"][0]["type"] == "text"
    assert "[1]" in payload["content"][0]["text"]
    assert "docs/chunk-a.pdf, page 3" in payload["content"][0]["text"]
    assert payload["structuredContent"]["resultCount"] == 2
    assert payload["structuredContent"]["citations"][0]["chunk_id"] == "chunk-a"


def test_response_builder_returns_friendly_message_for_empty_results() -> None:
    builder = ResponseBuilder()

    payload = builder.build([], "missing")

    assert "未找到与“missing”相关的文档" in payload["content"][0]["text"]
    assert payload["structuredContent"]["citations"] == []
