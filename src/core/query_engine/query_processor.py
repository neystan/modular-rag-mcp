"""查询预处理。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.settings import Settings
from core.trace import TraceContext


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "如何",
    "什么",
    "关于",
    "以及",
    "一个",
    "一些",
    "这个",
    "那个",
    "请问",
    "一下",
}
CHINESE_PREFIX_STOPWORDS = ("如何", "请问", "关于", "什么是", "怎么")


@dataclass(slots=True)
class ProcessedQuery:
    """查询预处理结果。"""

    original_query: str
    normalized_query: str
    keywords: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)


class QueryProcessorError(ValueError):
    """QueryProcessor 可读错误。"""


class QueryProcessor:
    """负责提取关键词并解析基础 filters。"""

    token_pattern = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9][A-Za-z0-9_-]*")

    def __init__(self, settings: Settings | dict[str, Any] | None = None) -> None:
        self.settings = settings
        self.stopwords = self._resolve_stopwords(settings)

    def process(self, query: str, trace: Any | None = None) -> ProcessedQuery:
        normalized_query = self._normalize_query(query)
        keywords = self._extract_keywords(normalized_query)
        processed = ProcessedQuery(
            original_query=query,
            normalized_query=normalized_query,
            keywords=keywords,
            filters={},
        )

        trace_context = trace if isinstance(trace, TraceContext) else None
        if trace_context is not None:
            trace_context.record_stage(
                "query_processor.process",
                {
                    "keyword_count": len(keywords),
                    "filters_count": len(processed.filters),
                },
            )
        return processed

    def _extract_keywords(self, query: str) -> list[str]:
        seen: set[str] = set()
        keywords: list[str] = []
        for token in self.token_pattern.findall(query):
            normalized = self._normalize_token(token)
            if not normalized or normalized in self.stopwords or normalized in seen:
                continue
            keywords.append(normalized)
            seen.add(normalized)

        if keywords:
            return keywords

        fallback = query.strip().lower()
        return [fallback] if fallback else []

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = token.lower().strip()
        for prefix in CHINESE_PREFIX_STOPWORDS:
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                normalized = normalized[len(prefix) :].strip()
                break
        return normalized

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise QueryProcessorError("query processor input error: query must be string")

        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            raise QueryProcessorError("query processor input error: query is required")
        return normalized

    @staticmethod
    def _resolve_stopwords(settings: Settings | dict[str, Any] | None) -> set[str]:
        configured: list[str] = []
        if isinstance(settings, Settings):
            retrieval = settings.retrieval
        elif isinstance(settings, dict):
            retrieval = settings.get("retrieval", {})
        else:
            retrieval = {}

        if isinstance(retrieval, dict):
            raw = retrieval.get("stopwords")
            if isinstance(raw, list):
                configured = [str(item).strip().lower() for item in raw if str(item).strip()]

        return set(DEFAULT_STOPWORDS).union(configured)
