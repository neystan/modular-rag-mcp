"""Chunk 重写与去噪。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory


DEFAULT_PROMPT = """请清理下面文本中的页眉、页脚、噪声和多余空白，保持原始含义，不补充事实。

原始文本：
{text}
"""


class ChunkRefinerError(RuntimeError):
    """ChunkRefiner 可读错误。"""


class ChunkRefiner(BaseTransform):
    """先规则去噪，再按配置决定是否调用 LLM 进一步精炼。"""

    default_prompt_path = "config/prompts/chunk_refinement.txt"

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.config = self._extract_config(settings)
        self.use_llm = bool(self.config.get("use_llm", False))
        self.prompt_text = self._load_prompt(prompt_path)
        self.llm = llm if self.use_llm else None
        self._llm_resolution_error: str | None = None

        if self.use_llm and self.llm is None:
            try:
                self.llm = self._resolve_llm(settings)
            except Exception as exc:  # noqa: BLE001
                self._llm_resolution_error = str(exc)

    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        transformed: list[Chunk] = []
        trace_context = trace if isinstance(trace, TraceContext) else None

        for chunk in chunks:
            try:
                transformed.append(self._transform_one(chunk, trace_context))
            except Exception as exc:  # noqa: BLE001
                fallback_metadata = copy.deepcopy(chunk.metadata)
                fallback_metadata["refined_by"] = "original"
                fallback_metadata["refine_error"] = str(exc)
                transformed.append(
                    Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=fallback_metadata,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        source_ref=chunk.source_ref,
                    )
                )
        return transformed

    def _transform_one(self, chunk: Chunk, trace: TraceContext | None) -> Chunk:
        refined_text = self._rule_based_refine(chunk.text)
        metadata = copy.deepcopy(chunk.metadata)
        metadata["refined_by"] = "rule"

        llm_result = self._llm_refine(refined_text, trace)
        if llm_result is not None:
            refined_text = llm_result
            metadata["refined_by"] = "llm"
        elif self.use_llm:
            metadata["refine_fallback_reason"] = self._llm_resolution_error or "llm_refine_failed"

        return Chunk(
            id=chunk.id,
            text=refined_text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _rule_based_refine(self, text: str) -> str:
        if not isinstance(text, str):
            raise ChunkRefinerError("chunk refiner input error: text must be string")
        if not text.strip():
            return ""

        segments = self._split_code_and_text(text)
        cleaned_parts: list[str] = []
        for is_code, segment in segments:
            cleaned_parts.append(segment.rstrip("\n") if is_code else self._clean_text_segment(segment))

        result = "\n\n".join(part for part in cleaned_parts if part)
        return result.strip()

    def _llm_refine(self, text: str, trace: TraceContext | None) -> str | None:
        if not self.use_llm:
            return None
        if self.llm is None:
            return None

        prompt = self.prompt_text.format(text=text)
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)])
        except Exception as exc:  # noqa: BLE001
            self._llm_resolution_error = str(exc)
            if trace is not None:
                trace.record_stage("chunk_refiner.llm_fallback", {"reason": str(exc)})
            return None

        refined = response.strip()
        if not refined:
            self._llm_resolution_error = "empty_llm_response"
            return None

        if trace is not None:
            trace.record_stage("chunk_refiner.llm_success", {"length": len(refined)})
        return refined

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        path = Path(prompt_path or self.default_prompt_path)
        if path.exists():
            prompt = path.read_text(encoding="utf-8").strip()
        else:
            prompt = DEFAULT_PROMPT.strip()

        if "{text}" not in prompt:
            raise ChunkRefinerError("chunk refiner config error: prompt must contain {text} placeholder")
        return prompt

    @staticmethod
    def _extract_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            ingestion = dict(settings.ingestion)
        elif isinstance(settings, dict):
            ingestion = dict(settings.get("ingestion", {}))
        else:
            raise ChunkRefinerError("chunk refiner config error: settings must be Settings or dict")

        chunk_refiner = ingestion.get("chunk_refiner", {})
        if chunk_refiner is None:
            return {}
        if not isinstance(chunk_refiner, dict):
            raise ChunkRefinerError("chunk refiner config error: ingestion.chunk_refiner must be mapping/object")
        return dict(chunk_refiner)

    @staticmethod
    def _resolve_llm(settings: Settings | dict[str, Any]) -> BaseLLM:
        return LLMFactory.create(settings)

    @staticmethod
    def _split_code_and_text(text: str) -> list[tuple[bool, str]]:
        parts: list[tuple[bool, str]] = []
        buffer: list[str] = []
        code_buffer: list[str] | None = None

        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if code_buffer is None:
                    if buffer:
                        parts.append((False, "\n".join(buffer)))
                        buffer = []
                    code_buffer = [line]
                else:
                    code_buffer.append(line)
                    parts.append((True, "\n".join(code_buffer)))
                    code_buffer = None
                continue

            if code_buffer is not None:
                code_buffer.append(line)
            else:
                buffer.append(line)

        if code_buffer is not None:
            parts.append((True, "\n".join(code_buffer)))
        elif buffer:
            parts.append((False, "\n".join(buffer)))
        return parts

    def _clean_text_segment(self, segment: str) -> str:
        lines = segment.splitlines()
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if self._is_noise_line(stripped):
                continue
            stripped = re.sub(r"<!--.*?-->", "", stripped)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if stripped:
                cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        if re.fullmatch(r"[-_=*]{3,}", line):
            return True
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line, re.IGNORECASE):
            return True
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            return True
        if re.fullmatch(r"header[:：].*", line, re.IGNORECASE):
            return True
        if re.fullmatch(r"footer[:：].*", line, re.IGNORECASE):
            return True
        return False
