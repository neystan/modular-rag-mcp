"""Chunk 图片描述生成。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


DEFAULT_PROMPT = """请为输入图片生成准确、简洁的中文描述，保留图表、流程和关键文本信息。

相关文本：
{text}
"""


class ImageCaptionerError(RuntimeError):
    """ImageCaptioner 可读错误。"""


class ImageCaptioner(BaseTransform):
    """根据 chunk 关联图片生成 caption，失败时不阻塞摄取。"""

    default_prompt_path = "config/prompts/image_captioning.txt"

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.config = self._extract_config(settings)
        self.use_vision_llm = bool(self.config.get("use_vision_llm", False))
        self.prompt_text = self._load_prompt(prompt_path)
        self.vision_llm = vision_llm if self.use_vision_llm else None
        self._vision_resolution_error: str | None = None

        if self.use_vision_llm and self.vision_llm is None:
            try:
                self.vision_llm = self._resolve_vision_llm(settings)
            except Exception as exc:  # noqa: BLE001
                self._vision_resolution_error = str(exc)

    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        transformed: list[Chunk] = []
        trace_context = trace if isinstance(trace, TraceContext) else None

        for chunk in chunks:
            try:
                transformed.append(self._transform_one(chunk, trace_context))
            except Exception as exc:  # noqa: BLE001
                fallback_metadata = copy.deepcopy(chunk.metadata)
                fallback_metadata["captioned_by"] = "original"
                fallback_metadata["caption_error"] = str(exc)
                transformed.append(self._clone_chunk(chunk, metadata=fallback_metadata, text=chunk.text))
        return transformed

    def _transform_one(self, chunk: Chunk, trace: TraceContext | None) -> Chunk:
        metadata = copy.deepcopy(chunk.metadata)
        image_refs = self._normalize_image_refs(metadata.get("image_refs"))
        images = self._normalize_images(metadata.get("images"))

        if not image_refs or not images:
            metadata["captioned_by"] = "skipped"
            return self._clone_chunk(chunk, metadata=metadata, text=chunk.text)

        if not self.use_vision_llm:
            metadata["captioned_by"] = "disabled"
            metadata["has_unprocessed_images"] = True
            return self._clone_chunk(chunk, metadata=metadata, text=chunk.text)

        if self.vision_llm is None:
            metadata["captioned_by"] = "fallback"
            metadata["caption_fallback_reason"] = self._vision_resolution_error or "vision_llm_unavailable"
            metadata["has_unprocessed_images"] = True
            return self._clone_chunk(chunk, metadata=metadata, text=chunk.text)

        captions: list[dict[str, str]] = []
        enriched_images = copy.deepcopy(images)
        image_map = {str(image["id"]): image for image in enriched_images}

        for image_id in image_refs:
            image = image_map.get(image_id)
            if image is None:
                continue

            caption = self._caption_image(chunk.text, image, trace)
            if caption is None:
                metadata["captioned_by"] = "fallback"
                metadata["caption_fallback_reason"] = self._vision_resolution_error or "image_caption_failed"
                metadata["has_unprocessed_images"] = True
                return self._clone_chunk(chunk, metadata=metadata, text=chunk.text)

            image["caption"] = caption
            captions.append({"image_id": image_id, "caption": caption})

        if not captions:
            metadata["captioned_by"] = "skipped"
            return self._clone_chunk(chunk, metadata=metadata, text=chunk.text)

        metadata["images"] = enriched_images
        metadata["image_captions"] = captions
        metadata["has_unprocessed_images"] = False
        metadata["captioned_by"] = "vision_llm"
        return self._clone_chunk(chunk, metadata=metadata, text=self._append_captions(chunk.text, captions))

    def _caption_image(self, chunk_text: str, image: dict[str, Any], trace: TraceContext | None) -> str | None:
        if self.vision_llm is None:
            return None

        image_path = image.get("path")
        if not isinstance(image_path, str) or not image_path.strip():
            self._vision_resolution_error = "image_path_missing"
            return None

        prompt = self._build_prompt(chunk_text=chunk_text, image=image)
        try:
            response = self.vision_llm.chat_with_image(prompt, image_path, trace=trace)
        except Exception as exc:  # noqa: BLE001
            self._vision_resolution_error = str(exc)
            if trace is not None:
                trace.record_stage("image_captioner.vision_fallback", {"reason": str(exc)})
            return None

        caption = self._normalize_caption(response.content)
        if not caption:
            self._vision_resolution_error = "empty_image_caption"
            return None

        if trace is not None:
            trace.record_stage("image_captioner.vision_success", {"image_id": image.get("id", ""), "length": len(caption)})
        return caption

    def _build_prompt(self, chunk_text: str, image: dict[str, Any]) -> str:
        image_id = str(image.get("id", "")).strip()
        prompt = self.prompt_text
        if "{text}" in prompt:
            prompt = prompt.replace("{text}", chunk_text)
        if "{chunk_text}" in prompt:
            prompt = prompt.replace("{chunk_text}", chunk_text)
        if "{image_id}" in prompt:
            prompt = prompt.replace("{image_id}", image_id)
        if all(token not in self.prompt_text for token in ("{text}", "{chunk_text}", "{image_id}")):
            prompt = f"{prompt.rstrip()}\n\n相关文本：\n{chunk_text}"
        return prompt

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        path = Path(prompt_path or self.default_prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return DEFAULT_PROMPT.strip()

    @staticmethod
    def _normalize_image_refs(value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ImageCaptionerError("image captioner input error: image_refs must be list")
        refs: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ImageCaptionerError("image captioner input error: image_refs[] must be non-empty string")
            refs.append(item)
        return refs

    @staticmethod
    def _normalize_images(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ImageCaptionerError("image captioner input error: images must be list")
        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ImageCaptionerError("image captioner input error: images[] must be mapping/object")
            normalized.append(dict(item))
        return normalized

    @staticmethod
    def _normalize_caption(caption: str) -> str:
        return re.sub(r"\s+", " ", str(caption)).strip()

    @staticmethod
    def _append_captions(text: str, captions: list[dict[str, str]]) -> str:
        suffix_parts: list[str] = []
        for item in captions:
            marker = f"[IMAGE CAPTION: {item['image_id']}] {item['caption']}"
            if marker not in text:
                suffix_parts.append(marker)
        if not suffix_parts:
            return text
        return f"{text.rstrip()}\n\n" + "\n".join(suffix_parts)

    @staticmethod
    def _clone_chunk(chunk: Chunk, metadata: dict[str, Any], text: str) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    @staticmethod
    def _extract_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            ingestion = dict(settings.ingestion)
        elif isinstance(settings, dict):
            ingestion = dict(settings.get("ingestion", {}))
        else:
            raise ImageCaptionerError("image captioner config error: settings must be Settings or dict")

        image_captioner = ingestion.get("image_captioner", {})
        if image_captioner is None:
            return {}
        if not isinstance(image_captioner, dict):
            raise ImageCaptionerError("image captioner config error: ingestion.image_captioner must be mapping/object")
        return dict(image_captioner)

    @staticmethod
    def _resolve_vision_llm(settings: Settings | dict[str, Any]) -> BaseVisionLLM:
        return LLMFactory.create_vision_llm(settings)
