"""Cross-Encoder Reranker 实现。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, cast

from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class CrossEncoderRerankerFallback(RuntimeError):
    """Cross-Encoder 不可用时的回退信号。"""


class CrossEncoderRerankerError(RuntimeError):
    """Cross-Encoder Reranker 可读错误。"""


class CrossEncoderReranker(BaseReranker):
    """使用 Cross-Encoder 风格 scorer 对候选项重排。"""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        if not isinstance(query, str) or not query.strip():
            raise CrossEncoderRerankerError("cross_encoder reranker input error: query is required")
        if not candidates:
            return []

        try:
            scores = self._score_candidates(query, candidates)
        except TimeoutError as exc:
            raise CrossEncoderRerankerFallback("cross_encoder reranker fallback: scorer timeout") from exc
        except CrossEncoderRerankerError:
            raise
        except Exception as exc:
            raise CrossEncoderRerankerFallback("cross_encoder reranker fallback: scorer failed") from exc

        if len(scores) != len(candidates):
            raise CrossEncoderRerankerError(
                "cross_encoder reranker response error: scorer result size must match candidates"
            )

        ranked = [
            replace(candidate, score=float(score))
            for candidate, score in zip(candidates, scores, strict=False)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)

        top_k = int(self.config.get("top_k", len(ranked)))
        if top_k <= 0:
            return []
        return ranked[:top_k]

    def _score_candidates(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        scorer = self.config.get("scorer")
        if scorer is not None:
            if not callable(scorer):
                raise CrossEncoderRerankerError("cross_encoder reranker config error: scorer must be callable")
            result = cast(Callable[[str, list[RerankCandidate]], list[float]], scorer)(query, candidates)
            return self._normalize_scores(result)

        return [self._default_score(query, candidate.text) for candidate in candidates]

    def _normalize_scores(self, scores: list[Any]) -> list[float]:
        if not isinstance(scores, list):
            raise CrossEncoderRerankerError("cross_encoder reranker response error: scorer must return list")
        try:
            return [float(score) for score in scores]
        except (TypeError, ValueError) as exc:
            raise CrossEncoderRerankerError(
                "cross_encoder reranker response error: scorer results must be numeric"
            ) from exc

    def _default_score(self, query: str, text: str) -> float:
        query_terms = self._tokenize(query)
        text_terms = self._tokenize(text)
        if not query_terms or not text_terms:
            return 0.0

        overlap = sum(1 for term in query_terms if term in text_terms)
        density = overlap / len(query_terms)
        coverage = overlap / len(text_terms)
        return density + coverage

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = text.lower().replace("\n", " ")
        return {token for token in normalized.split(" ") if token}
