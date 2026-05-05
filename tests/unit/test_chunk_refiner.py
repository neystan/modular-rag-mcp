"""ChunkRefiner 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner, ChunkRefinerError
from libs.llm.base_llm import BaseLLM, ChatMessage


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "llm cleaned", error: Exception | None = None) -> None:
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
        ingestion={"chunk_refiner": {"use_llm": use_llm}},
    )


def make_chunk(text: str) -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        metadata={"source_path": "docs/sample.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_base_transform_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTransform()  # type: ignore[abstract]


def test_rule_based_refine_matches_fixture_expectations() -> None:
    fixture_path = Path("tests/fixtures/noisy_chunks.json")
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    refiner = ChunkRefiner(make_settings())

    for case in cases:
        cleaned = refiner._rule_based_refine(case["input"])
        for expected in case["expected_contains"]:
            assert expected in cleaned, case["name"]
        for unexpected in case["expected_absent"]:
            assert unexpected not in cleaned, case["name"]


def test_code_blocks_keep_internal_formatting() -> None:
    refiner = ChunkRefiner(make_settings())
    text = "Header: Sample\n```python\ndef hello():\n    print(\"world\")\n```\nPage 1 of 2"

    cleaned = refiner._rule_based_refine(text)

    assert "def hello():\n    print(\"world\")" in cleaned
    assert "Header:" not in cleaned
    assert "Page 1 of 2" not in cleaned


def test_transform_uses_rule_mode_by_default() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))

    result = refiner.transform([make_chunk("Header: A\nclean body\nFooter: B")])

    assert result[0].text == "clean body"
    assert result[0].metadata["refined_by"] == "rule"


def test_transform_uses_llm_when_enabled() -> None:
    llm = FakeLLM(response="llm output")
    refiner = ChunkRefiner(make_settings(use_llm=True, llm_provider="openai"), llm=llm)
    trace = TraceContext()

    result = refiner.transform([make_chunk("messy text")], trace=trace)

    assert result[0].text == "llm output"
    assert result[0].metadata["refined_by"] == "llm"
    assert len(llm.calls) == 1
    assert any(stage["stage"] == "chunk_refiner.llm_success" for stage in trace.stages)


def test_transform_falls_back_to_rule_when_llm_fails() -> None:
    llm = FakeLLM(error=RuntimeError("boom"))
    refiner = ChunkRefiner(make_settings(use_llm=True, llm_provider="openai"), llm=llm)
    trace = TraceContext()

    result = refiner.transform([make_chunk("Header: A\nbody\nFooter: B")], trace=trace)

    assert result[0].text == "body"
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].metadata["refine_fallback_reason"] == "boom"
    assert any(stage["stage"] == "chunk_refiner.llm_fallback" for stage in trace.stages)


def test_transform_records_resolution_error_when_llm_config_unavailable() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=True))

    result = refiner.transform([make_chunk("body")])

    assert result[0].metadata["refined_by"] == "rule"
    assert "refine_fallback_reason" in result[0].metadata


def test_prompt_must_contain_text_placeholder(tmp_path: Path) -> None:
    prompt_path = tmp_path / "bad_prompt.txt"
    prompt_path.write_text("missing placeholder", encoding="utf-8")

    with pytest.raises(ChunkRefinerError, match=r"\{text\}"):
        ChunkRefiner(make_settings(), prompt_path=prompt_path)


def test_transform_preserves_original_chunk_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(make_settings())

    def broken(_: str) -> str:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(refiner, "_rule_based_refine", broken)
    chunk = make_chunk("body")
    result = refiner.transform([chunk])

    assert result[0].text == "body"
    assert result[0].metadata["refined_by"] == "original"
    assert result[0].metadata["refine_error"] == "unexpected"
