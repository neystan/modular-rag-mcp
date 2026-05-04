"""Azure Vision LLM 单元测试。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
from PIL import Image

from core.settings import Settings
from libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from libs.llm.llm_factory import LLMFactory


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
            "provider": "azure",
            "model": "gpt-4o",
            "deployment_name": "vision-deployment",
            "azure_endpoint": "https://azure.test",
            "api_version": "2024-02-15-preview",
            "api_key": "secret",
            "max_image_size": 2048,
            "timeout": 9,
        },
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def write_image(path: Path, size: tuple[int, int], color: str = "navy") -> None:
    image = Image.new("RGB", size, color=color)
    image.save(path, format="PNG")


def test_factory_creates_azure_vision_llm_from_settings() -> None:
    llm = LLMFactory.create_vision_llm(make_settings())

    assert isinstance(llm, AzureVisionLLM)


def test_chat_with_image_posts_expected_azure_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    image_path = tmp_path / "sample.png"
    write_image(image_path, (128, 64))

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse({"choices": [{"message": {"content": "图中是一张测试图片"}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)
    llm = AzureVisionLLM(make_settings().vision_llm)

    response = llm.chat_with_image("请描述图片", str(image_path))

    assert response.content == "图中是一张测试图片"
    assert captured["url"] == (
        "https://azure.test/openai/deployments/vision-deployment/chat/completions"
        "?api-version=2024-02-15-preview"
    )
    assert captured["headers"]["Api-key"] == "secret"
    assert captured["timeout"] == 9.0
    assert captured["payload"]["model"] == "gpt-4o"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "请描述图片"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert response.metadata["image"]["compressed"] is False
    assert response.metadata["image"]["processed_size"] == {"width": 128, "height": 64}


def test_chat_with_image_accepts_base64_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "small.png"
    write_image(image_path, (96, 96), color="green")
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"choices": [{"message": {"content": [{"type": "text", "text": "base64 ok"}]}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)
    llm = AzureVisionLLM(make_settings().vision_llm)

    response = llm.chat_with_image("看图", image_base64)

    assert response.content == "base64 ok"
    assert response.metadata["image"]["source"] == "base64"


def test_large_image_is_resized_to_max_image_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "large.png"
    write_image(image_path, (4096, 1024), color="red")

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"choices": [{"message": {"content": "已压缩"}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)
    llm = AzureVisionLLM({**make_settings().vision_llm, "max_image_size": 1024})

    response = llm.chat_with_image("压缩测试", str(image_path))

    assert response.metadata["image"]["compressed"] is True
    assert response.metadata["image"]["original_size"] == {"width": 4096, "height": 1024}
    assert response.metadata["image"]["processed_size"] == {"width": 1024, "height": 256}


def test_timeout_error_is_readable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "timeout.png"
    write_image(image_path, (32, 32))

    def fake_urlopen(request: Any, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("request timed out")

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)
    llm = AzureVisionLLM(make_settings().vision_llm)

    with pytest.raises(AzureVisionLLMError, match="azure vision timeout error: TimeoutError"):
        llm.chat_with_image("超时测试", str(image_path))


def test_auth_failure_includes_azure_error_code(
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
        {"error": {"code": "Unauthorized", "message": "The API key is invalid."}}
    ).encode("utf-8")

    def fake_read(self: HTTPError) -> bytes:
        return error_payload

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)
    monkeypatch.setattr(HTTPError, "read", fake_read, raising=False)
    llm = AzureVisionLLM(make_settings().vision_llm)

    with pytest.raises(
        AzureVisionLLMError,
        match="azure vision HTTP error: 401; code=Unauthorized; The API key is invalid.",
    ):
        llm.chat_with_image("认证失败测试", str(image_path))
