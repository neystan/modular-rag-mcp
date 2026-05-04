"""LLM 工厂测试。"""

from __future__ import annotations

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory, LLMFactoryError


class FakeLLM(BaseLLM):
    """测试用 LLM Provider。"""

    def chat(self, messages: list[ChatMessage] | list[dict[str, object]]) -> str:
        return f"fake:{len(messages)}:{self.config['model']}"


class NotLLM:
    pass


@pytest.fixture(autouse=True)
def clear_llm_registry() -> None:
    LLMFactory.clear_providers()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": provider, "model": "fake-model"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_register_provider_and_create_from_settings() -> None:
    LLMFactory.register_provider("fake", FakeLLM)

    llm = LLMFactory.create(make_settings())

    assert isinstance(llm, FakeLLM)
    assert llm.chat([ChatMessage(role="user", content="hello")]) == "fake:1:fake-model"


def test_create_from_dict_uses_llm_section() -> None:
    LLMFactory.register_provider("fake", FakeLLM)

    llm = LLMFactory.create({"llm": {"provider": "fake", "model": "dict-model"}})

    assert isinstance(llm, FakeLLM)
    assert llm.chat([]) == "fake:0:dict-model"


def test_provider_name_is_case_insensitive() -> None:
    LLMFactory.register_provider("Fake", FakeLLM)

    llm = LLMFactory.create(make_settings(provider="FAKE"))

    assert isinstance(llm, FakeLLM)


def test_unknown_provider_reports_available_providers() -> None:
    LLMFactory.register_provider("fake", FakeLLM)

    with pytest.raises(LLMFactoryError, match="未知 LLM provider: missing"):
        LLMFactory.create(make_settings(provider="missing"))


def test_missing_provider_reports_config_path() -> None:
    with pytest.raises(LLMFactoryError, match="llm.provider"):
        LLMFactory.create({"llm": {"model": "fake-model"}})


def test_register_provider_requires_basellm_subclass() -> None:
    with pytest.raises(LLMFactoryError, match="必须继承 BaseLLM"):
        LLMFactory.register_provider("bad", NotLLM)  # type: ignore[arg-type]
