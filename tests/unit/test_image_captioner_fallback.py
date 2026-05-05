"""ImageCaptioner 单元测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM, VisionChatResponse


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, response: str = "图中展示了系统架构图。", error: Exception | None = None) -> None:
        super().__init__({})
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> VisionChatResponse:
        self.calls.append({"text": text, "image_path": image_path, "trace": trace})
        if self.error is not None:
            raise self.error
        return VisionChatResponse(content=self.response, metadata={"provider": "fake"})


def make_settings(use_vision_llm: bool = False, provider: str = "placeholder") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        vision_llm={"provider": provider, "model": "fake-vision-model"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
        ingestion={"image_captioner": {"use_vision_llm": use_vision_llm}},
    )


def write_image(path: Path, size: tuple[int, int] = (64, 64), color: str = "blue") -> None:
    image = Image.new("RGB", size, color=color)
    image.save(path, format="PNG")


def make_chunk(image_path: str, with_images: bool = True) -> Chunk:
    metadata: dict[str, Any] = {"source_path": "docs/sample.pdf"}
    if with_images:
        metadata["image_refs"] = ["img-1"]
        metadata["images"] = [
            {
                "id": "img-1",
                "path": image_path,
                "page": 1,
                "text_offset": 0,
                "text_length": 14,
                "position": {},
            }
        ]
    return Chunk(
        id="chunk-1",
        text="系统流程说明 [IMAGE: img-1]",
        metadata=metadata,
        start_offset=0,
        end_offset=21,
        source_ref="doc-1",
    )


def test_captioner_generates_caption_when_enabled(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    write_image(image_path)
    vision_llm = FakeVisionLLM(response="图中展示了蓝色流程图。")
    captioner = ImageCaptioner(make_settings(use_vision_llm=True, provider="qwen"), vision_llm=vision_llm)
    trace = TraceContext()

    result = captioner.transform([make_chunk(str(image_path))], trace=trace)[0]

    assert result.metadata["captioned_by"] == "vision_llm"
    assert result.metadata["has_unprocessed_images"] is False
    assert result.metadata["image_captions"] == [{"image_id": "img-1", "caption": "图中展示了蓝色流程图。"}]
    assert result.metadata["images"][0]["caption"] == "图中展示了蓝色流程图。"
    assert "[IMAGE CAPTION: img-1] 图中展示了蓝色流程图。" in result.text
    assert len(vision_llm.calls) == 1
    assert any(stage["stage"] == "image_captioner.vision_success" for stage in trace.stages)


def test_captioner_marks_unprocessed_when_disabled(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    write_image(image_path)
    captioner = ImageCaptioner(make_settings(use_vision_llm=False))

    result = captioner.transform([make_chunk(str(image_path))])[0]

    assert result.metadata["captioned_by"] == "disabled"
    assert result.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in result.metadata
    assert "[IMAGE CAPTION:" not in result.text


def test_captioner_falls_back_when_vision_llm_errors(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    write_image(image_path)
    vision_llm = FakeVisionLLM(error=RuntimeError("vision boom"))
    captioner = ImageCaptioner(make_settings(use_vision_llm=True, provider="qwen"), vision_llm=vision_llm)
    trace = TraceContext()

    result = captioner.transform([make_chunk(str(image_path))], trace=trace)[0]

    assert result.metadata["captioned_by"] == "fallback"
    assert result.metadata["caption_fallback_reason"] == "vision boom"
    assert result.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in result.metadata
    assert any(stage["stage"] == "image_captioner.vision_fallback" for stage in trace.stages)


def test_captioner_keeps_chunk_unchanged_when_no_images() -> None:
    captioner = ImageCaptioner(make_settings(use_vision_llm=True))
    chunk = make_chunk("unused", with_images=False)

    result = captioner.transform([chunk])[0]

    assert result.metadata["captioned_by"] == "skipped"
    assert result.text == chunk.text


def test_captioner_respects_prompt_template(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    prompt_path = tmp_path / "prompt.txt"
    write_image(image_path)
    prompt_path.write_text("图片ID：{image_id}\n相关文本：{text}", encoding="utf-8")
    vision_llm = FakeVisionLLM()
    captioner = ImageCaptioner(
        make_settings(use_vision_llm=True, provider="qwen"),
        vision_llm=vision_llm,
        prompt_path=prompt_path,
    )

    captioner.transform([make_chunk(str(image_path))])

    assert "图片ID：img-1" in vision_llm.calls[0]["text"]
    assert "相关文本：系统流程说明 [IMAGE: img-1]" in vision_llm.calls[0]["text"]


def test_captioner_preserves_original_chunk_on_unexpected_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    write_image(image_path)
    captioner = ImageCaptioner(make_settings(use_vision_llm=False))

    def broken(_: Any) -> list[str]:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(captioner, "_normalize_image_refs", broken)
    result = captioner.transform([make_chunk(str(image_path))])[0]

    assert result.metadata["captioned_by"] == "original"
    assert result.metadata["caption_error"] == "unexpected"
