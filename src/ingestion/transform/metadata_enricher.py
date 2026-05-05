"""Chunk 元数据增强。"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory


DEFAULT_MAX_TITLE_LENGTH = 80
DEFAULT_MAX_SUMMARY_LENGTH = 220
DEFAULT_MAX_TAGS = 5
DEFAULT_PROMPT = """请基于下面的 chunk 内容生成结构化元数据，返回 JSON 对象，不要输出额外说明。

要求：
1. title: 生成简洁标题，控制在 24 个字以内
2. summary: 生成 1-2 句摘要，保留关键事实
3. tags: 提取 3-5 个标签，标签使用短语，不要重复

文本：
{text}
"""

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "will",
    "would",
    "should",
    "could",
    "about",
    "than",
    "then",
    "their",
    "there",
    "which",
    "when",
    "where",
    "如何",
    "以及",
    "或者",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "通过",
    "进行",
    "需要",
    "可以",
    "一个",
    "一些",
    "一种",
    "中的",
    "相关",
}


class MetadataEnricherError(RuntimeError):
    """MetadataEnricher 可读错误。"""


class MetadataEnricher(BaseTransform):
    """规则增强为基础，按配置可切换到 LLM 增强。"""

    default_prompt_path = "config/prompts/metadata_enrichment.txt"

    def __init__(
        self,
        settings: Settings | dict[str, Any],
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.config = self._extract_config(settings)
        self.use_llm = bool(self.config.get("use_llm", True))
        self.max_title_length = self._read_positive_int("max_title_length", DEFAULT_MAX_TITLE_LENGTH)
        self.max_summary_length = self._read_positive_int("max_summary_length", DEFAULT_MAX_SUMMARY_LENGTH)
        self.max_tags = self._read_positive_int("max_tags", DEFAULT_MAX_TAGS)
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
                fallback_metadata["metadata_enriched_by"] = "original"
                fallback_metadata["metadata_enrich_error"] = str(exc)
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
        metadata = copy.deepcopy(chunk.metadata)
        rule_result = self._rule_based_metadata(chunk)
        metadata.update(rule_result)
        metadata["metadata_enriched_by"] = "rule"

        llm_result = self._llm_metadata(chunk.text, trace)
        if llm_result is not None:
            metadata.update(llm_result)
            metadata["metadata_enriched_by"] = "llm"
        elif self.use_llm:
            metadata["metadata_enrich_fallback_reason"] = self._llm_resolution_error or "llm_enrich_failed"

        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _rule_based_metadata(self, chunk: Chunk) -> dict[str, Any]:
        if not isinstance(chunk.text, str):
            raise MetadataEnricherError("metadata enricher input error: text must be string")

        normalized_text = self._normalize_text(chunk.text)
        title = self._extract_title(normalized_text, chunk.metadata)
        summary = self._extract_summary(normalized_text)
        tags = self._extract_tags(normalized_text, chunk.metadata)

        if not title:
            title = f"Chunk {chunk.id}"
        if not summary:
            summary = title
        if not tags:
            tags = [self._safe_tag_from_title(title)]

        return {
            "title": title,
            "summary": summary,
            "tags": tags,
        }

    def _llm_metadata(self, text: str, trace: TraceContext | None) -> dict[str, Any] | None:
        if not self.use_llm or self.llm is None:
            return None

        prompt = self.prompt_text.format(text=text)
        try:
            response = self.llm.chat([ChatMessage(role="user", content=prompt)])
            parsed = self._parse_llm_response(response)
        except Exception as exc:  # noqa: BLE001
            self._llm_resolution_error = str(exc)
            if trace is not None:
                trace.record_stage("metadata_enricher.llm_fallback", {"reason": str(exc)})
            return None

        if trace is not None:
            trace.record_stage(
                "metadata_enricher.llm_success",
                {"tags": len(parsed.get("tags", [])), "summary_length": len(parsed.get("summary", ""))},
            )
        return parsed

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        content = response.strip()
        if not content:
            raise MetadataEnricherError("empty_llm_response")

        if content.startswith("```"):
            fenced_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if fenced_match:
                content = fenced_match.group(1).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise MetadataEnricherError("invalid_llm_json") from exc

        if not isinstance(data, dict):
            raise MetadataEnricherError("llm_response_must_be_object")

        title = self._truncate_text(str(data.get("title", "")).strip(), self.max_title_length)
        summary = self._truncate_text(str(data.get("summary", "")).strip(), self.max_summary_length)
        tags = self._normalize_tags(data.get("tags"))
        if not title or not summary or not tags:
            raise MetadataEnricherError("llm_response_missing_required_fields")

        return {
            "title": title,
            "summary": summary,
            "tags": tags,
        }

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        path = Path(prompt_path or self.default_prompt_path)
        if path.exists():
            prompt = path.read_text(encoding="utf-8").strip()
        else:
            prompt = DEFAULT_PROMPT.strip()

        if "{text}" not in prompt:
            raise MetadataEnricherError("metadata enricher config error: prompt must contain {text} placeholder")
        return prompt

    def _extract_title(self, text: str, metadata: dict[str, Any]) -> str:
        existing = str(metadata.get("title", "")).strip()
        if existing:
            return self._truncate_text(existing, self.max_title_length)

        lines = [line.strip(" #*-") for line in text.splitlines() if line.strip()]
        for line in lines:
            if len(line) >= 6:
                return self._truncate_text(line, self.max_title_length)

        first_sentence = self._first_sentence(text)
        return self._truncate_text(first_sentence, self.max_title_length)

    def _extract_summary(self, text: str) -> str:
        existing = ""
        sentences = self._split_sentences(text)
        if not sentences:
            return existing

        summary_parts: list[str] = []
        total_length = 0
        for sentence in sentences:
            if total_length and total_length + 1 + len(sentence) > self.max_summary_length:
                break
            summary_parts.append(sentence)
            total_length += len(sentence) + (1 if summary_parts[:-1] else 0)
            if len(summary_parts) >= 2:
                break

        summary = " ".join(summary_parts).strip()
        return self._truncate_text(summary, self.max_summary_length)

    def _extract_tags(self, text: str, metadata: dict[str, Any]) -> list[str]:
        existing = metadata.get("tags")
        existing_tags = self._normalize_tags(existing)
        if existing_tags:
            return existing_tags

        words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
        frequencies: dict[str, int] = {}
        for word in words:
            normalized = word.strip().lower()
            if normalized in _STOPWORDS:
                continue
            frequencies[normalized] = frequencies.get(normalized, 0) + 1

        ranked = sorted(
            frequencies.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
        tags: list[str] = []
        for word, _ in ranked:
            tag = word if re.search(r"[\u4e00-\u9fff]", word) else word.replace("_", " ")
            if tag not in tags:
                tags.append(tag)
            if len(tags) >= self.max_tags:
                break
        return tags

    def _normalize_tags(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_tags = re.split(r"[,，\n]", value)
        elif isinstance(value, list):
            raw_tags = [str(item) for item in value]
        else:
            raise MetadataEnricherError("tags must be string or list")

        normalized: list[str] = []
        for tag in raw_tags:
            cleaned = re.sub(r"\s+", " ", str(tag)).strip(" -#")
            if not cleaned:
                continue
            cleaned = self._truncate_text(cleaned, 32)
            if cleaned not in normalized:
                normalized.append(cleaned)
            if len(normalized) >= self.max_tags:
                break
        return normalized

    def _normalize_text(self, text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        filtered = [line for line in lines if line]
        return "\n".join(filtered).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[。！？.!?])\s+", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _first_sentence(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if sentences:
            return sentences[0]
        return text.strip()

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 1)].rstrip() + "..."

    @staticmethod
    def _safe_tag_from_title(title: str) -> str:
        return re.sub(r"\s+", " ", title).strip() or "general"

    def _read_positive_int(self, key: str, default: int) -> int:
        value = self.config.get(key, default)
        if not isinstance(value, int) or value <= 0:
            raise MetadataEnricherError(f"metadata enricher config error: {key} must be positive int")
        return value

    @staticmethod
    def _extract_config(settings: Settings | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, Settings):
            ingestion = dict(settings.ingestion)
        elif isinstance(settings, dict):
            ingestion = dict(settings.get("ingestion", {}))
        else:
            raise MetadataEnricherError("metadata enricher config error: settings must be Settings or dict")

        metadata_enricher = ingestion.get("metadata_enricher", {})
        if metadata_enricher is None:
            return {}
        if not isinstance(metadata_enricher, dict):
            raise MetadataEnricherError(
                "metadata enricher config error: ingestion.metadata_enricher must be mapping/object"
            )
        return dict(metadata_enricher)

    @staticmethod
    def _resolve_llm(settings: Settings | dict[str, Any]) -> BaseLLM:
        return LLMFactory.create(settings)
