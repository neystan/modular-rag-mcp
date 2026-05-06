"""RRF Fusion 单元测试。"""

from __future__ import annotations

import pytest

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from core.query_engine.fusion import FusionError, RRFFusion


def make_settings(rrf_k: int | None = None) -> Settings:
    retrieval = {"top_k": 5}
    if rrf_k is not None:
        retrieval["rrf_k"] = rrf_k
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval=retrieval,
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": f"docs/{chunk_id}.pdf"},
    )


def test_rrf_fuses_two_rankings_and_promotes_overlap() -> None:
    fusion = RRFFusion(make_settings(rrf_k=10))
    dense = [make_result("chunk-a"), make_result("chunk-b"), make_result("chunk-c")]
    sparse = [make_result("chunk-b"), make_result("chunk-a"), make_result("chunk-d")]

    results = fusion.fuse(dense, sparse, top_k=4)

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
    assert results[0].score > results[2].score
    assert results[1].score > results[3].score


def test_rrf_keeps_stable_order_for_single_route_results() -> None:
    fusion = RRFFusion(make_settings())

    results = fusion.fuse([make_result("chunk-a"), make_result("chunk-b")], [], top_k=2)

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-b"]


def test_rrf_honors_top_k_limit() -> None:
    fusion = RRFFusion(make_settings())

    results = fusion.fuse(
        [make_result("chunk-a"), make_result("chunk-b"), make_result("chunk-c"), make_result("chunk-e")],
        [make_result("chunk-d")],
        top_k=2,
    )

    assert [item.chunk_id for item in results] == ["chunk-a", "chunk-d"]


def test_rrf_records_trace_stage() -> None:
    fusion = RRFFusion(make_settings())
    trace = TraceContext()

    fusion.fuse([make_result("chunk-a")], [make_result("chunk-b")], top_k=2, trace=trace)

    assert any(stage["stage"] == "fusion.rrf" for stage in trace.stages)


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(FusionError, match="k must be positive int"):
        RRFFusion(make_settings(), k=0)


def test_rrf_rejects_invalid_top_k() -> None:
    fusion = RRFFusion(make_settings())

    with pytest.raises(FusionError, match="top_k must be positive int"):
        fusion.fuse([], [], top_k=0)
