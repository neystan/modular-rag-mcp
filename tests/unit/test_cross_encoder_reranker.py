"""Cross-Encoder Reranker 测试。"""

from __future__ import annotations

import pytest

from core.settings import Settings
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
    CrossEncoderRerankerFallback,
)
from libs.reranker.reranker_factory import RerankerFactory


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "cross_encoder", "top_k": 2},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="python testing guide", score=0.1),
        RerankCandidate(id="b", text="database storage basics", score=0.2),
        RerankCandidate(id="c", text="python unit test patterns", score=0.3),
    ]


def test_factory_creates_cross_encoder_reranker() -> None:
    reranker = RerankerFactory.create(make_settings())

    assert isinstance(reranker, CrossEncoderReranker)


def test_cross_encoder_reranks_with_injected_scorer() -> None:
    def scorer(query: str, candidates: list[RerankCandidate]) -> list[float]:
        assert query == "python testing"
        assert [candidate.id for candidate in candidates] == ["a", "b", "c"]
        return [0.7, 0.1, 0.9]

    reranker = CrossEncoderReranker({"scorer": scorer, "top_k": 2})

    ranked = reranker.rerank("python testing", make_candidates())

    assert [item.id for item in ranked] == ["c", "a"]
    assert [item.score for item in ranked] == [0.9, 0.7]


def test_cross_encoder_default_scorer_is_deterministic() -> None:
    reranker = CrossEncoderReranker({"top_k": 3})

    ranked = reranker.rerank("python test", make_candidates())

    assert [item.id for item in ranked] == ["c", "a", "b"]


def test_cross_encoder_timeout_raises_fallback_signal() -> None:
    def scorer(query: str, candidates: list[RerankCandidate]) -> list[float]:
        raise TimeoutError("timed out")

    reranker = CrossEncoderReranker({"scorer": scorer})

    with pytest.raises(CrossEncoderRerankerFallback, match="scorer timeout"):
        reranker.rerank("python testing", make_candidates())


def test_cross_encoder_failure_raises_fallback_signal() -> None:
    def scorer(query: str, candidates: list[RerankCandidate]) -> list[float]:
        raise RuntimeError("backend down")

    reranker = CrossEncoderReranker({"scorer": scorer})

    with pytest.raises(CrossEncoderRerankerFallback, match="scorer failed"):
        reranker.rerank("python testing", make_candidates())


def test_cross_encoder_invalid_score_shape_is_readable() -> None:
    def scorer(query: str, candidates: list[RerankCandidate]) -> list[float]:
        return [0.1]

    reranker = CrossEncoderReranker({"scorer": scorer})

    with pytest.raises(CrossEncoderRerankerError, match="result size must match candidates"):
        reranker.rerank("python testing", make_candidates())
