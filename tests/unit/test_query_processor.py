"""QueryProcessor 单元测试。"""

from __future__ import annotations

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.query_engine.query_processor import ProcessedQuery, QueryProcessor, QueryProcessorError


def make_settings(stopwords: list[str] | None = None) -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5, "stopwords": stopwords or []},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_process_extracts_non_empty_keywords_and_empty_filters() -> None:
    processor = QueryProcessor(make_settings())

    result = processor.process("How does hybrid retrieval fuse dense and sparse results?")

    assert isinstance(result, ProcessedQuery)
    assert result.original_query == "How does hybrid retrieval fuse dense and sparse results?"
    assert result.normalized_query == "How does hybrid retrieval fuse dense and sparse results?"
    assert result.keywords == ["does", "hybrid", "retrieval", "fuse", "dense", "sparse", "results"]
    assert result.filters == {}


def test_process_deduplicates_tokens_and_removes_stopwords() -> None:
    processor = QueryProcessor(make_settings(stopwords=["dense"]))

    result = processor.process("Dense dense retrieval retrieval 的 的 实现")

    assert result.keywords == ["retrieval", "实现"]


def test_process_supports_chinese_keywords() -> None:
    processor = QueryProcessor(make_settings())

    result = processor.process("如何查看图片索引和文档摘要")

    assert result.keywords == ["查看图片索引和文档摘要", "查看", "看图", "图片", "片索", "索引", "引和", "和文", "文档", "档摘", "摘要"]


def test_process_expands_chinese_question_into_retrievable_keywords() -> None:
    processor = QueryProcessor(make_settings())

    result = processor.process("作者叫什么名字？")

    assert "作者叫什么名字" in result.keywords
    assert "作者" in result.keywords
    assert "名字" in result.keywords


def test_process_records_trace_stage() -> None:
    processor = QueryProcessor(make_settings())
    trace = TraceContext()

    processor.process("bm25 索引 查询", trace=trace)

    assert any(stage["stage"] == "query_processor.process" for stage in trace.stages)


def test_process_rejects_blank_query() -> None:
    processor = QueryProcessor(make_settings())

    with pytest.raises(QueryProcessorError, match="query is required"):
        processor.process("   \n  ")


def test_process_rejects_non_string_query() -> None:
    processor = QueryProcessor(make_settings())

    with pytest.raises(QueryProcessorError, match="query must be string"):
        processor.process(123)  # type: ignore[arg-type]
