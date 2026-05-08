"""query.py 脚本单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

from core.query_engine.query_processor import ProcessedQuery
from core.settings import Settings
from core.types import RetrievalResult
from core.trace import TraceContext


def _load_query_script_module() -> Any:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "query.py"
    spec = importlib.util.spec_from_file_location("query_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load query.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


query_script = _load_query_script_module()
QueryComponents = query_script.QueryComponents
main = query_script.main
render_execution = query_script.render_execution
run_query = query_script.run_query


class FakeQueryProcessor:
    def __init__(self, processed: ProcessedQuery) -> None:
        self.processed = processed
        self.calls: list[str] = []

    def process(self, query: str, trace: Any | None = None) -> ProcessedQuery:
        self.calls.append(query)
        return self.processed


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self.results)


class FakeSparseRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, keywords: list[str], top_k: int, trace: Any | None = None) -> list[RetrievalResult]:
        self.calls.append({"keywords": list(keywords), "top_k": top_k})
        return list(self.results)


class FakeFusion:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        *,
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "dense_ids": [item.chunk_id for item in dense_results],
                "sparse_ids": [item.chunk_id for item in sparse_results],
                "top_k": top_k,
            }
        )
        return list(self.results)


class FakeHybridSearch:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        if isinstance(trace, TraceContext):
            trace.record_stage(
                "query_processing",
                {
                    "details": {
                        "query_text": query,
                        "keywords": ["configure", "azure"],
                        "filters": dict(filters or {}),
                    }
                },
            )
            trace.record_stage(
                "dense_retrieval",
                {"details": {"chunk_ids": [item.chunk_id for item in self.results], "result_count": len(self.results)}},
            )
            trace.record_stage(
                "sparse_retrieval",
                {"details": {"chunk_ids": [item.chunk_id for item in self.results], "result_count": len(self.results)}},
            )
            trace.record_stage(
                "fusion",
                {"details": {"chunk_ids": [item.chunk_id for item in self.results], "result_count": len(self.results)}},
            )
            trace.record_stage(
                "hybrid_search.search",
                {"chunk_ids": [item.chunk_id for item in self.results], "result_count": len(self.results)},
            )
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self.results)

    @staticmethod
    def _apply_metadata_filters(
        candidates: list[RetrievalResult],
        filters: dict[str, Any] | None,
    ) -> list[RetrievalResult]:
        if not filters:
            return list(candidates)
        return [
            item for item in candidates if all(item.metadata.get(key) == value for key, value in filters.items())
        ]


class FakeReranker:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        if isinstance(trace, TraceContext):
            trace.record_stage(
                "rerank",
                {
                    "details": {
                        "input_ids": [item.chunk_id for item in candidates],
                        "result_ids": [item.chunk_id for item in self.results],
                    }
                },
            )
        self.calls.append({"query": query, "candidate_ids": [item.chunk_id for item in candidates], "top_k": top_k})
        return list(self.results)


def make_result(chunk_id: str, *, collection: str = "manuals", page: int | None = None) -> RetrievalResult:
    metadata: dict[str, object] = {"source_path": f"docs/{chunk_id}.pdf", "collection": collection}
    if page is not None:
        metadata["page"] = page
    return RetrievalResult(chunk_id=chunk_id, score=0.8, text=f"text for {chunk_id}", metadata=metadata)


def make_settings(*, top_k: int = 5) -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": top_k},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_components(
    *,
    dense_results: list[RetrievalResult] | None = None,
    sparse_results: list[RetrievalResult] | None = None,
    fusion_results: list[RetrievalResult] | None = None,
    final_results: list[RetrievalResult] | None = None,
) -> QueryComponents:
    return QueryComponents(
        query_processor=FakeQueryProcessor(
            ProcessedQuery(
                original_query="How to configure Azure?",
                normalized_query="How to configure Azure?",
                keywords=["configure", "azure"],
                filters={},
            )
        ),
        dense_retriever=FakeDenseRetriever(dense_results or []),
        sparse_retriever=FakeSparseRetriever(sparse_results or []),
        fusion=FakeFusion(fusion_results or []),
        hybrid_search=FakeHybridSearch(fusion_results or []),
        reranker=FakeReranker(final_results or fusion_results or []),
    )


def test_run_query_uses_hybrid_search_and_reranker_in_default_mode() -> None:
    fusion_results = [make_result("chunk-a")]
    final_results = [make_result("chunk-b")]
    components = make_components(fusion_results=fusion_results, final_results=final_results)

    execution = run_query(
        "How to configure Azure?",
        top_k=3,
        collection="manuals",
        components=components,
    )

    assert execution.processed_query.normalized_query == "How to configure Azure?"
    assert components.hybrid_search.calls == [
        {"query": "How to configure Azure?", "top_k": 3, "filters": {"collection": "manuals"}}
    ]
    assert components.reranker.calls == [
        {"query": "How to configure Azure?", "candidate_ids": ["chunk-a"], "top_k": 3}
    ]
    assert [item.chunk_id for item in execution.final_results] == ["chunk-b"]


def test_run_query_verbose_collects_dense_sparse_and_fusion_results() -> None:
    dense_results = [make_result("chunk-d", collection="manuals")]
    sparse_results = [make_result("chunk-s", collection="faq")]
    fusion_results = [make_result("chunk-d", collection="manuals"), make_result("chunk-s", collection="faq")]
    components = make_components(
        dense_results=dense_results,
        sparse_results=sparse_results,
        fusion_results=fusion_results,
        final_results=fusion_results,
    )

    execution = run_query(
        "How to configure Azure?",
        top_k=2,
        collection="manuals",
        verbose=True,
        no_rerank=True,
        components=components,
    )

    assert components.dense_retriever.calls == [
        {"query": "How to configure Azure?", "top_k": 2, "filters": {}}
    ]
    assert components.sparse_retriever.calls == [{"keywords": ["configure", "azure"], "top_k": 2}]
    assert components.fusion.calls == [{"dense_ids": ["chunk-d"], "sparse_ids": ["chunk-s"], "top_k": 2}]
    assert components.reranker.calls == []
    assert [item.chunk_id for item in execution.final_results] == ["chunk-d"]


def test_run_query_uses_configured_top_k_when_argument_is_omitted() -> None:
    dense_results = [make_result("chunk-d")]
    sparse_results = [make_result("chunk-s")]
    fusion_results = [make_result("chunk-d"), make_result("chunk-s")]
    components = make_components(
        dense_results=dense_results,
        sparse_results=sparse_results,
        fusion_results=fusion_results,
        final_results=fusion_results,
    )

    execution = run_query(
        "How to configure Azure?",
        verbose=True,
        no_rerank=True,
        settings=make_settings(top_k=2),
        components=components,
    )

    assert components.dense_retriever.calls == [{"query": "How to configure Azure?", "top_k": 2, "filters": {}}]
    assert components.sparse_retriever.calls == [{"keywords": ["configure", "azure"], "top_k": 2}]
    assert components.fusion.calls == [{"dense_ids": ["chunk-d"], "sparse_ids": ["chunk-s"], "top_k": 2}]
    assert [item.chunk_id for item in execution.final_results] == ["chunk-d", "chunk-s"]


def test_render_execution_returns_friendly_message_when_no_results() -> None:
    components = make_components()

    execution = run_query("How to configure Azure?", components=components, no_rerank=True)

    assert render_execution(execution) == "未找到相关文档，请先运行 ingest.py 摄取数据。"


def test_render_execution_includes_verbose_sections() -> None:
    result = make_result("chunk-a", page=3)
    components = make_components(
        dense_results=[result],
        sparse_results=[result],
        fusion_results=[result],
        final_results=[result],
    )

    execution = run_query("How to configure Azure?", verbose=True, components=components)
    rendered = render_execution(execution, verbose=True)

    assert "Dense Results:" in rendered
    assert "Sparse Results:" in rendered
    assert "Fusion Results:" in rendered
    assert "Rerank Results:" in rendered
    assert "page=3" in rendered


def test_main_returns_error_for_invalid_top_k(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--query", "test", "--top-k", "0"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "top_k must be positive int" in captured.err


def test_run_query_persists_query_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: list[dict[str, Any]] = []
    fusion_results = [make_result("chunk-a")]
    components = make_components(fusion_results=fusion_results, final_results=fusion_results)
    monkeypatch.setattr(query_script, "_persist_query_trace", persisted.append)

    execution = run_query("How to configure Azure?", top_k=3, components=components)

    assert [item.chunk_id for item in execution.final_results] == ["chunk-a"]
    assert len(persisted) == 1
    assert persisted[0]["trace_type"] == "query"
    stage_names = [stage["stage"] for stage in persisted[0]["stages"]]
    assert "query.execution" in stage_names
