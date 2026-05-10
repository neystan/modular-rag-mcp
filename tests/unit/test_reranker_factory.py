"""Reranker 工厂测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory, RerankerFactoryError
from libs.reranker.qwen_reranker import QwenReranker


class ReverseReranker(BaseReranker):
    """测试用 Reranker Provider。"""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        return list(reversed(candidates))


class NotReranker:
    pass


@pytest.fixture(autouse=True)
def clear_reranker_registry() -> None:
    RerankerFactory.clear_providers()


def make_settings(provider: str = "none") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": provider},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="文本 A", score=0.2),
        RerankCandidate(id="b", text="文本 B", score=0.8),
        RerankCandidate(id="c", text="文本 C", score=0.5),
    ]


def test_none_reranker_preserves_candidate_order() -> None:
    reranker = NoneReranker()
    candidates = make_candidates()

    ranked = reranker.rerank("测试查询", candidates)

    assert ranked == candidates
    assert ranked is not candidates


def test_factory_creates_none_reranker_by_default() -> None:
    reranker = RerankerFactory.create(make_settings())

    assert isinstance(reranker, NoneReranker)
    assert reranker.rerank("query", make_candidates()) == make_candidates()


def test_register_provider_and_create_from_settings() -> None:
    RerankerFactory.register_provider("reverse", ReverseReranker)

    reranker = RerankerFactory.create(make_settings(provider="reverse"))

    assert isinstance(reranker, ReverseReranker)
    assert [item.id for item in reranker.rerank("query", make_candidates())] == ["c", "b", "a"]


def test_create_from_dict_uses_rerank_section() -> None:
    RerankerFactory.register_provider("reverse", ReverseReranker)

    reranker = RerankerFactory.create({"rerank": {"provider": "reverse"}})

    assert isinstance(reranker, ReverseReranker)


def test_provider_name_is_case_insensitive() -> None:
    RerankerFactory.register_provider("Reverse", ReverseReranker)

    reranker = RerankerFactory.create(make_settings(provider="REVERSE"))

    assert isinstance(reranker, ReverseReranker)


def test_unknown_provider_reports_available_providers() -> None:
    with pytest.raises(RerankerFactoryError, match="未知 Reranker provider: missing"):
        RerankerFactory.create(make_settings(provider="missing"))


def test_missing_provider_reports_config_path() -> None:
    with pytest.raises(RerankerFactoryError, match="rerank.provider"):
        RerankerFactory.create({"rerank": {"provider": ""}})


def test_register_provider_requires_basereranker_subclass() -> None:
    with pytest.raises(RerankerFactoryError, match="必须继承 BaseReranker"):
        RerankerFactory.register_provider("bad", NotReranker)  # type: ignore[arg-type]


def test_factory_creates_qwen_reranker() -> None:
    reranker = RerankerFactory.create(
        make_settings(provider="qwen")
    )

    assert isinstance(reranker, QwenReranker)
