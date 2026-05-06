"""MCP 响应构建器。"""

from __future__ import annotations

from core.response.citation_generator import CitationGenerator
from core.types import RetrievalResult


class ResponseBuilder:
    """将检索结果组装为 MCP tool 返回结构。"""

    def __init__(self, citation_generator: CitationGenerator | None = None) -> None:
        self.citation_generator = citation_generator or CitationGenerator()

    def build(self, retrieval_results: list[RetrievalResult], query: str) -> dict[str, object]:
        citations = self.citation_generator.generate(retrieval_results)
        if not retrieval_results:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"未找到与“{query}”相关的文档，请先运行 ingest.py 摄取数据或调整查询条件。",
                    }
                ],
                "structuredContent": {
                    "query": query,
                    "resultCount": 0,
                    "citations": [],
                },
            }

        markdown = self._build_markdown(query, retrieval_results, citations)
        return {
            "content": [
                {
                    "type": "text",
                    "text": markdown,
                }
            ],
            "structuredContent": {
                "query": query,
                "resultCount": len(retrieval_results),
                "citations": citations,
            },
        }

    def _build_markdown(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        citations: list[dict[str, object]],
    ) -> str:
        lines = [f"以下是与“{query}”相关的检索结果：", ""]
        for citation, result in zip(citations, retrieval_results, strict=False):
            lines.append(f"{citation['index']}. {self._summarize(result.text)} [{citation['index']}]")

        lines.extend(["", "参考来源："])
        for citation in citations:
            page = citation.get("page")
            page_text = f", page {page}" if page is not None else ""
            lines.append(f"[{citation['index']}] {citation['source']}{page_text}")
        return "\n".join(lines)

    @staticmethod
    def _summarize(text: str, limit: int = 160) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 3, 1)].rstrip() + "..."
