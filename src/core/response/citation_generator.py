"""引用生成器。"""

from __future__ import annotations

from typing import Any

from core.types import RetrievalResult


class CitationGenerator:
    """将检索结果转换为结构化引用列表。"""

    display_score_scale = 60.0

    def generate(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for index, item in enumerate(retrieval_results, start=1):
            citations.append(
                {
                    "index": index,
                    "source": str(item.metadata.get("source_path", "<unknown>")),
                    "page": item.metadata.get("page"),
                    "chunk_id": item.chunk_id,
                    "score": round(float(item.score) * self.display_score_scale, 6),
                }
            )
        return citations
