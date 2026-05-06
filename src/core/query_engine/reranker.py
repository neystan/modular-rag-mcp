"""Core 层重排序编排。"""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class RerankerError(RuntimeError):
    """Core Reranker 可读错误。"""


class Reranker:
    """编排 libs.reranker，并在失败时回退到 fusion 原始排序。"""

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        reranker_backend: BaseReranker | None = None,
    ) -> None:
        self.settings = settings
        self.reranker_backend = reranker_backend or RerankerFactory.create(settings)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        normalized_query = self._normalize_query(query)
        normalized_candidates = self._normalize_candidates(candidates)
        limit = self._resolve_top_k(top_k, normalized_candidates)
        trace_context = trace if isinstance(trace, TraceContext) else None

        if not normalized_candidates or limit == 0:
            self._record_trace(trace_context, len(normalized_candidates), 0, False, None)
            return []

        rerank_candidates = [
            RerankCandidate(
                id=item.chunk_id,
                text=item.text,
                score=item.score,
                metadata=dict(item.metadata),
            )
            for item in normalized_candidates
        ]

        fallback_reason: str | None = None
        try:
            ranked = self.reranker_backend.rerank(normalized_query, rerank_candidates, trace=trace_context)
            results = [self._candidate_to_result(item, fallback=False, fallback_reason=None) for item in ranked[:limit]]
        except Exception as exc:  # noqa: BLE001
            fallback_reason = str(exc) or exc.__class__.__name__
            results = [
                self._result_with_fallback(item, fallback_reason)
                for item in normalized_candidates[:limit]
            ]

        self._record_trace(
            trace_context,
            len(normalized_candidates),
            len(results),
            fallback_reason is not None,
            fallback_reason,
        )
        return results

    def _resolve_top_k(self, top_k: int | None, candidates: list[RetrievalResult]) -> int:
        if top_k is None:
            configured = self._get_config_value("top_k", len(candidates))
            top_k = int(configured)
        if not isinstance(top_k, int) or top_k < 0:
            raise RerankerError("core reranker input error: top_k must be non-negative int")
        return min(top_k, len(candidates))

    def _get_config_value(self, key: str, default: Any) -> Any:
        if isinstance(self.settings, Settings):
            return self.settings.rerank.get(key, default)
        if isinstance(self.settings, dict):
            rerank = self.settings.get("rerank", {})
            if isinstance(rerank, dict):
                return rerank.get(key, default)
        return default

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise RerankerError("core reranker input error: query is required")
        return query.strip()

    @staticmethod
    def _normalize_candidates(candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        if not isinstance(candidates, list):
            raise RerankerError("core reranker input error: candidates must be list[RetrievalResult]")
        for item in candidates:
            if not isinstance(item, RetrievalResult):
                raise RerankerError("core reranker input error: candidates must be list[RetrievalResult]")
        return list(candidates)

    @staticmethod
    def _candidate_to_result(
        candidate: RerankCandidate,
        fallback: bool,
        fallback_reason: str | None,
    ) -> RetrievalResult:
        metadata = dict(candidate.metadata)
        metadata["rerank_fallback"] = fallback
        if fallback_reason:
            metadata["rerank_fallback_reason"] = fallback_reason
        else:
            metadata.pop("rerank_fallback_reason", None)
        return RetrievalResult(
            chunk_id=candidate.id,
            score=candidate.score,
            text=candidate.text,
            metadata=metadata,
        )

    def _result_with_fallback(self, item: RetrievalResult, fallback_reason: str) -> RetrievalResult:
        return self._candidate_to_result(
            RerankCandidate(
                id=item.chunk_id,
                text=item.text,
                score=item.score,
                metadata=dict(item.metadata),
            ),
            fallback=True,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _record_trace(
        trace: TraceContext | None,
        input_count: int,
        result_count: int,
        fallback: bool,
        fallback_reason: str | None,
    ) -> None:
        if trace is None:
            return

        trace.record_stage(
            "query_reranker.rerank",
            {
                "input_count": input_count,
                "result_count": result_count,
                "fallback": fallback,
                "fallback_reason": fallback_reason,
            },
        )
