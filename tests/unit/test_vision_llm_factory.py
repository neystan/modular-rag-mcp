"""Vision LLM 工厂测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.llm.base_vision_llm import BaseVisionLLM, VisionChatResponse
from libs.llm.llm_factory import LLMFactory, LLMFactoryError


class FakeVisionLLM(BaseVisionLLM):
    """测试用 Vision LLM Provider。"""

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> VisionChatResponse:
        return VisionChatResponse(
            content=f"{text}:{image_path}",
            metadata={"preprocessed": self.preprocess_image(image_path)},
        )


class NotVisionLLM:
    pass


@pytest.fixture(autouse=True)
def clear_llm_registry() -> None:
    LLMFactory.clear_providers()


def make_settings(provider: str = "fake_vision") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        vision_llm={"provider": provider, "max_image_size": 2048},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_register_vision_provider_and_create_from_settings() -> None:
    LLMFactory.register_vision_provider("fake_vision", FakeVisionLLM)

    llm = LLMFactory.create_vision_llm(make_settings())

    assert isinstance(llm, FakeVisionLLM)
    response = llm.chat_with_image("describe", "image.png")
    assert response.content == "describe:image.png"
    assert response.metadata["preprocessed"] == "image.png"


def test_create_vision_llm_from_dict_uses_vision_section() -> None:
    LLMFactory.register_vision_provider("fake_vision", FakeVisionLLM)

    llm = LLMFactory.create_vision_llm(
        {"vision_llm": {"provider": "fake_vision", "max_image_size": 1024}}
    )

    assert isinstance(llm, FakeVisionLLM)


def test_vision_provider_name_is_case_insensitive() -> None:
    LLMFactory.register_vision_provider("Fake_Vision", FakeVisionLLM)

    llm = LLMFactory.create_vision_llm(make_settings(provider="FAKE_VISION"))

    assert isinstance(llm, FakeVisionLLM)


def test_unknown_vision_provider_reports_available_providers() -> None:
    LLMFactory.register_vision_provider("fake_vision", FakeVisionLLM)

    with pytest.raises(LLMFactoryError, match="未知 Vision LLM provider: missing"):
        LLMFactory.create_vision_llm(make_settings(provider="missing"))


def test_missing_vision_provider_reports_config_path() -> None:
    with pytest.raises(LLMFactoryError, match="vision_llm.provider"):
        LLMFactory.create_vision_llm({"vision_llm": {"max_image_size": 1024}})


def test_register_vision_provider_requires_basevisionllm_subclass() -> None:
    with pytest.raises(LLMFactoryError, match="必须继承 BaseVisionLLM"):
        LLMFactory.register_vision_provider("bad", NotVisionLLM)  # type: ignore[arg-type]
