"""Embedding 工厂测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError


class FakeEmbedding(BaseEmbedding):
    """测试用 Embedding Provider。"""

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        dimensions = int(self.config.get("dimensions", 3))
        return [[float(len(text)), float(index), float(dimensions)] for index, text in enumerate(texts)]


class NotEmbedding:
    pass


@pytest.fixture(autouse=True)
def clear_embedding_registry() -> None:
    EmbeddingFactory.clear_providers()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": provider, "model": "fake-model", "dimensions": 3},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_register_provider_and_create_from_settings() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(make_settings())

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.embed(["a", "abcd"]) == [[1.0, 0.0, 3.0], [4.0, 1.0, 3.0]]


def test_create_from_dict_uses_embedding_section() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(
        {"embedding": {"provider": "fake", "model": "dict-model", "dimensions": 8}}
    )

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.embed(["abc"]) == [[3.0, 0.0, 8.0]]


def test_provider_name_is_case_insensitive() -> None:
    EmbeddingFactory.register_provider("Fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(make_settings(provider="FAKE"))

    assert isinstance(embedding, FakeEmbedding)


def test_unknown_provider_reports_available_providers() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)

    with pytest.raises(EmbeddingFactoryError, match="未知 Embedding provider: missing"):
        EmbeddingFactory.create(make_settings(provider="missing"))


def test_missing_provider_reports_config_path() -> None:
    with pytest.raises(EmbeddingFactoryError, match="embedding.provider"):
        EmbeddingFactory.create({"embedding": {"model": "fake-model"}})


def test_register_provider_requires_baseembedding_subclass() -> None:
    with pytest.raises(EmbeddingFactoryError, match="必须继承 BaseEmbedding"):
        EmbeddingFactory.register_provider("bad", NotEmbedding)  # type: ignore[arg-type]
