"""结果融合。"""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult


class FusionError(ValueError):
    """Fusion 可读错误。"""


class RRFFusion:
    """基于 Reciprocal Rank Fusion 的结果融合器。"""

    default_k = 60

    def __init__(self, settings: Settings | dict[str, Any] | None = None, *, k: int | None = None) -> None:
        self.settings = settings
        self.k = k if k is not None else self._resolve_k(settings)
        if not isinstance(self.k, int) or self.k <= 0:
            raise FusionError("rrf fusion config error: k must be positive int")

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        normalized_top_k = self._normalize_top_k(top_k)
        combined: dict[str, dict[str, Any]] = {}

        self._accumulate(combined, dense_results)
        self._accumulate(combined, sparse_results)

        ranked = sorted(
            combined.values(),
            key=lambda item: (-float(item["rrf_score"]), item["first_rank"], item["chunk_id"]),
        )
        fused = [
            RetrievalResult(
                chunk_id=item["chunk_id"],
                score=round(float(item["rrf_score"]), 10),
                text=item["text"],
                metadata=item["metadata"],
            )
            for item in ranked[:normalized_top_k]
        ]

        trace_context = trace if isinstance(trace, TraceContext) else None
        if trace_context is not None:
            trace_context.record_stage(
                "fusion.rrf",
                {
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(fused),
                    "k": self.k,
                },
            )
        return fused

    def _accumulate(self, combined: dict[str, dict[str, Any]], results: list[RetrievalResult]) -> None:
        for rank, result in enumerate(results, start=1):
            score = 1.0 / (self.k + rank)
            existing = combined.get(result.chunk_id)
            if existing is None:
                combined[result.chunk_id] = {
                    "chunk_id": result.chunk_id,
                    "rrf_score": score,
                    "first_rank": rank,
                    "text": result.text,
                    "metadata": result.metadata,
                }
                continue

            existing["rrf_score"] += score
            existing["first_rank"] = min(int(existing["first_rank"]), rank)

    def _normalize_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return max(1, self.default_k)
        if not isinstance(top_k, int) or top_k <= 0:
            raise FusionError("rrf fusion input error: top_k must be positive int")
        return top_k

    def _resolve_k(self, settings: Settings | dict[str, Any] | None) -> int:
        if isinstance(settings, Settings):
            retrieval = settings.retrieval
        elif isinstance(settings, dict):
            retrieval = settings.get("retrieval", settings)
        else:
            retrieval = {}

        if isinstance(retrieval, dict):
            value = retrieval.get("rrf_k")
            if isinstance(value, int) and value > 0:
                return value
        return self.default_k
