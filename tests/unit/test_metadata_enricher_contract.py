"""MetadataEnricher 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher, MetadataEnricherError
from libs.llm.base_llm import BaseLLM, ChatMessage


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        super().__init__({})
        self.response = response
        self.error = error
        self.calls: list[list[ChatMessage | dict[str, object]]] = []

    def chat(self, messages: list[ChatMessage] | list[dict[str, object]]) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def make_settings(use_llm: bool = False, llm_provider: str = "placeholder") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": llm_provider, "model": "fake-model"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
        ingestion={"metadata_enricher": {"use_llm": use_llm}},
    )


def make_chunk(text: str, metadata: dict[str, object] | None = None) -> Chunk:
    merged_metadata = {"source_path": "docs/sample.pdf"}
    if metadata:
        merged_metadata.update(metadata)
    return Chunk(
        id="chunk-1",
        text=text,
        metadata=merged_metadata,
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_rule_mode_populates_required_metadata() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))
    chunk = make_chunk("## Qwen Deployment Guide\nQwen supports structured outputs for metadata extraction.")

    result = enricher.transform([chunk])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert isinstance(result.metadata["title"], str) and result.metadata["title"]
    assert isinstance(result.metadata["summary"], str) and result.metadata["summary"]
    assert isinstance(result.metadata["tags"], list) and result.metadata["tags"]


def test_rule_mode_preserves_existing_metadata_and_images() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))
    chunk = make_chunk(
        "Architecture overview with image references.",
        metadata={
            "collection": "manuals",
            "images": [{"id": "img-1", "path": "data/images/1.png", "text_offset": 0, "text_length": 12}],
        },
    )

    result = enricher.transform([chunk])[0]

    assert result.metadata["collection"] == "manuals"
    assert result.metadata["images"][0]["id"] == "img-1"
    assert result.metadata["title"]


def test_llm_mode_uses_llm_response_when_enabled() -> None:
    llm = FakeLLM(
        response='{"title":"Qwen Metadata","summary":"Explain how Qwen enriches metadata.","tags":["qwen","metadata","llm"]}'
    )
    enricher = MetadataEnricher(make_settings(use_llm=True, llm_provider="openai"), llm=llm)
    trace = TraceContext()

    result = enricher.transform([make_chunk("source text")], trace=trace)[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert result.metadata["title"] == "Qwen Metadata"
    assert result.metadata["tags"] == ["qwen", "metadata", "llm"]
    assert len(llm.calls) == 1
    assert any(stage["stage"] == "metadata_enricher.llm_success" for stage in trace.stages)


def test_llm_mode_accepts_json_fenced_response() -> None:
    llm = FakeLLM(
        response="""```json
{"title":"Structured Output","summary":"JSON fenced content is accepted.","tags":["json","metadata","contract"]}
```"""
    )
    enricher = MetadataEnricher(make_settings(use_llm=True, llm_provider="openai"), llm=llm)

    result = enricher.transform([make_chunk("source text")])[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert result.metadata["title"] == "Structured Output"


def test_llm_failure_falls_back_to_rule_mode() -> None:
    llm = FakeLLM(error=RuntimeError("boom"))
    enricher = MetadataEnricher(make_settings(use_llm=True, llm_provider="openai"), llm=llm)
    trace = TraceContext()

    result = enricher.transform([make_chunk("Fallback text with deployment notes.")], trace=trace)[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_enrich_fallback_reason"] == "boom"
    assert result.metadata["title"]
    assert any(stage["stage"] == "metadata_enricher.llm_fallback" for stage in trace.stages)


def test_invalid_llm_json_falls_back_to_rule_mode() -> None:
    llm = FakeLLM(response="not-json")
    enricher = MetadataEnricher(make_settings(use_llm=True, llm_provider="openai"), llm=llm)

    result = enricher.transform([make_chunk("Fallback text for invalid JSON.")])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_enrich_fallback_reason"] == "invalid_llm_json"


def test_resolution_error_falls_back_to_rule_mode() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=True))

    result = enricher.transform([make_chunk("Qwen integration text.")])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert "metadata_enrich_fallback_reason" in result.metadata


def test_prompt_must_contain_text_placeholder(tmp_path: Path) -> None:
    prompt_path = tmp_path / "bad_prompt.txt"
    prompt_path.write_text("missing placeholder", encoding="utf-8")

    with pytest.raises(MetadataEnricherError, match=r"\{text\}"):
        MetadataEnricher(make_settings(use_llm=False), prompt_path=prompt_path)


def test_transform_preserves_original_chunk_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))

    def broken(_: Chunk) -> dict[str, object]:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(enricher, "_rule_based_metadata", broken)
    chunk = make_chunk("body")

    result = enricher.transform([chunk])[0]

    assert result.text == "body"
    assert result.metadata["metadata_enriched_by"] == "original"
    assert result.metadata["metadata_enrich_error"] == "unexpected"
