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
                    "results": [],
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
                "results": self._build_structured_results(retrieval_results, citations),
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
            lines.append(f"### 结果 {citation['index']} [{citation['index']}] · score={citation['score']}")
            lines.append(result.text.strip() or "<empty>")
            lines.append("")

        lines.append("参考来源：")
        for citation in citations:
            page = citation.get("page")
            page_text = f", page {page}" if page is not None else ""
            lines.append(f"[{citation['index']}] {citation['source']}{page_text}")
        return "\n".join(lines)

    @staticmethod
    def _build_structured_results(
        retrieval_results: list[RetrievalResult],
        citations: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for citation, result in zip(citations, retrieval_results, strict=False):
            results.append(
                {
                    "index": citation["index"],
                    "chunk_id": result.chunk_id,
                    "score": citation["score"],
                    "source": citation["source"],
                    "page": citation.get("page"),
                    "text": result.text,
                    "metadata": ResponseBuilder._compact_metadata(result.metadata),
                }
            )
        return results

    @staticmethod
    def _compact_metadata(metadata: dict[str, object]) -> dict[str, object]:
        compact = dict(metadata)
        compact.pop("images", None)
        compact.pop("image_captions", None)
        return compact
