"""Qwen Vision LLM 单元测试。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
from PIL import Image

from core.settings import Settings
from libs.llm.llm_factory import LLMFactory
from libs.llm.qwen_vision_llm import QwenVisionLLM, QwenVisionLLMError


class FakeHTTPResponse:
    """测试用 HTTP 响应。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        vision_llm={
            "provider": "qwen",
            "model": "qwen-vl-max",
            "api_key": "dashscope-secret",
            "max_image_size": 2048,
            "timeout": 7,
        },
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def write_image(path: Path, size: tuple[int, int], color: str = "purple") -> None:
    image = Image.new("RGB", size, color=color)
    image.save(path, format="PNG")


def test_factory_creates_qwen_vision_llm_from_settings() -> None:
    llm = LLMFactory.create_vision_llm(make_settings())

    assert isinstance(llm, QwenVisionLLM)


def test_chat_with_image_posts_expected_qwen_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    image_path = tmp_path / "sample.png"
    write_image(image_path, (144, 80))

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse({"choices": [{"message": {"content": "这是一张测试图片"}}]})

    monkeypatch.setattr("libs.llm.qwen_vision_llm.urlopen", fake_urlopen)
    llm = QwenVisionLLM(make_settings().vision_llm)

    response = llm.chat_with_image("请描述图片", str(image_path))

    assert response.content == "这是一张测试图片"
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer dashscope-secret"
    assert captured["timeout"] == 7.0
    assert captured["payload"]["model"] == "qwen-vl-max"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "请描述图片"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert response.metadata["image"]["compressed"] is False


def test_chat_with_image_accepts_base64_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "small.png"
    write_image(image_path, (96, 96), color="green")
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"choices": [{"message": {"content": [{"type": "text", "text": "base64 ok"}]}}]})

    monkeypatch.setattr("libs.llm.qwen_vision_llm.urlopen", fake_urlopen)
    llm = QwenVisionLLM(make_settings().vision_llm)

    response = llm.chat_with_image("看图", image_base64)

    assert response.content == "base64 ok"
    assert response.metadata["image"]["source"] == "base64"


def test_large_image_is_resized_to_max_image_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "large.png"
    write_image(image_path, (5000, 1000), color="orange")

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"choices": [{"message": {"content": "已压缩"}}]})

    monkeypatch.setattr("libs.llm.qwen_vision_llm.urlopen", fake_urlopen)
    llm = QwenVisionLLM({**make_settings().vision_llm, "max_image_size": 1000})

    response = llm.chat_with_image("压缩测试", str(image_path))

    assert response.metadata["image"]["compressed"] is True
    assert response.metadata["image"]["original_size"] == {"width": 5000, "height": 1000}
    assert response.metadata["image"]["processed_size"] == {"width": 1000, "height": 200}


def test_auth_failure_includes_qwen_error_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "auth.png"
    write_image(image_path, (32, 32))

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    error_payload = json.dumps(
        {"error": {"code": "InvalidApiKey", "message": "Invalid DashScope API key."}}
    ).encode("utf-8")

    def fake_read(self: HTTPError) -> bytes:
        return error_payload

    monkeypatch.setattr("libs.llm.qwen_vision_llm.urlopen", fake_urlopen)
    monkeypatch.setattr(HTTPError, "read", fake_read, raising=False)
    llm = QwenVisionLLM(make_settings().vision_llm)

    with pytest.raises(
        QwenVisionLLMError,
        match="qwen vision HTTP error: 401; code=InvalidApiKey; Invalid DashScope API key.",
    ):
        llm.chat_with_image("认证失败测试", str(image_path))
