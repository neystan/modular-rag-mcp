"""Splitter 工厂测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory, SplitterFactoryError


class FakeSplitter(BaseSplitter):
    """测试用 Splitter Provider。"""

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        separator = str(self.config.get("separator", "|"))
        return [part for part in text.split(separator) if part]


class NotSplitter:
    pass


@pytest.fixture(autouse=True)
def clear_splitter_registry() -> None:
    SplitterFactory.clear_providers()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": provider, "separator": "|"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_register_provider_and_create_from_settings() -> None:
    SplitterFactory.register_provider("fake", FakeSplitter)

    splitter = SplitterFactory.create(make_settings())

    assert isinstance(splitter, FakeSplitter)
    assert splitter.split_text("a|b||c") == ["a", "b", "c"]


def test_create_from_dict_uses_splitter_section() -> None:
    SplitterFactory.register_provider("fake", FakeSplitter)

    splitter = SplitterFactory.create({"splitter": {"provider": "fake", "separator": ","}})

    assert isinstance(splitter, FakeSplitter)
    assert splitter.split_text("a,b,c") == ["a", "b", "c"]


def test_provider_name_is_case_insensitive() -> None:
    SplitterFactory.register_provider("Fake", FakeSplitter)

    splitter = SplitterFactory.create(make_settings(provider="FAKE"))

    assert isinstance(splitter, FakeSplitter)


def test_unknown_provider_reports_available_providers() -> None:
    SplitterFactory.register_provider("fake", FakeSplitter)

    with pytest.raises(SplitterFactoryError, match="未知 Splitter provider: missing"):
        SplitterFactory.create(make_settings(provider="missing"))


def test_missing_provider_reports_config_path() -> None:
    with pytest.raises(SplitterFactoryError, match="splitter.provider"):
        SplitterFactory.create({"splitter": {"chunk_size": 1000}})


def test_register_provider_requires_basesplitter_subclass() -> None:
    with pytest.raises(SplitterFactoryError, match="必须继承 BaseSplitter"):
        SplitterFactory.register_provider("bad", NotSplitter)  # type: ignore[arg-type]
