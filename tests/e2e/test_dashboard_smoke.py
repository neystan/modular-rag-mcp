"""Dashboard 冒烟测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.settings import Settings
from ingestion.document_manager import CollectionStats, DocumentDetail, DocumentInfo
from observability.dashboard.services.config_service import ComponentCard
from observability.dashboard.services.trace_service import (
    QueryTraceDetail,
    QueryTraceSummary,
    StageTiming,
    TraceDetail,
    TraceSummary,
)


def _make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp", "environment": "test"},
        llm={"provider": "qwen", "model": "demo-llm"},
        vision_llm={"provider": "qwen", "model": "demo-vl"},
        embedding={"provider": "qwen", "model": "demo-embedding"},
        splitter={"provider": "recursive"},
        vector_store={"provider": "memory", "collection": "manuals", "persist_path": "./tmp/chroma"},
        retrieval={"top_k": 3},
        rerank={"provider": "qwen", "model": "qwen3-rerank"},
        evaluation={"provider": "ragas", "enabled": True},
        observability={"log_level": "INFO"},
        ingestion={},
    )


@dataclass
class _FakeConfigService:
    settings: Settings

    def get_settings(self) -> Settings:
        return self.settings

    def get_app_summary(self) -> dict[str, str]:
        return {
            "name": "modular-rag-mcp",
            "environment": "test",
            "settings_path": "config/settings.yaml",
        }

    def get_component_cards(self) -> list[ComponentCard]:
        return [
            ComponentCard(label="LLM", provider="qwen", model="demo-llm", details=["base_url: https://example.test"]),
            ComponentCard(label="Embedding", provider="qwen", model="demo-embedding", details=[]),
        ]

    def get_raw_config(self) -> dict[str, object]:
        return {
            "app": self.settings.app,
            "llm": self.settings.llm,
            "embedding": self.settings.embedding,
            "vector_store": self.settings.vector_store,
            "retrieval": self.settings.retrieval,
            "rerank": self.settings.rerank,
            "evaluation": self.settings.evaluation,
        }


@dataclass
class _FakeDataService:
    documents: list[DocumentInfo]
    detail: DocumentDetail
    collections: list[str]
    stats: CollectionStats

    def list_collections(self) -> list[str]:
        return list(self.collections)

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        if collection is None:
            return list(self.documents)
        return [item for item in self.documents if item.collection == collection]

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        assert doc_id == self.detail.source_path
        return self.detail

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        del collection
        return self.stats


@dataclass
class _FakeTraceService:
    ingestion_summaries: list[TraceSummary]
    ingestion_detail: TraceDetail
    query_summaries: list[QueryTraceSummary]
    query_detail: QueryTraceDetail

    def list_ingestion_traces(self) -> list[TraceSummary]:
        return list(self.ingestion_summaries)

    def get_ingestion_trace(self, trace_id: str) -> TraceDetail:
        assert trace_id == self.ingestion_detail.trace_id
        return self.ingestion_detail

    def list_query_traces(self, query_keyword: str | None = None) -> list[QueryTraceSummary]:
        keyword = str(query_keyword or "").strip().lower()
        if not keyword:
            return list(self.query_summaries)
        return [item for item in self.query_summaries if keyword in item.query_text.lower()]

    def get_query_trace(self, trace_id: str) -> QueryTraceDetail:
        assert trace_id == self.query_detail.trace_id
        return self.query_detail


def _render_overview(config_service) -> None:
    from observability.dashboard.pages.overview import render

    render(config_service=config_service)


def _render_data_browser(data_service) -> None:
    from observability.dashboard.pages.data_browser import render

    render(data_service=data_service)


def _render_ingestion_manager(data_service, settings) -> None:
    from observability.dashboard.pages.ingestion_manager import render

    render(
        data_service=data_service,
        settings_loader=lambda _: settings,
    )


def _render_ingestion_traces(trace_service) -> None:
    from observability.dashboard.pages.ingestion_traces import render

    render(trace_service=trace_service)


def _render_query_traces(trace_service) -> None:
    from observability.dashboard.pages.query_traces import render

    render(trace_service=trace_service)


def _render_evaluation_panel(settings) -> None:
    from observability.dashboard.pages.evaluation_panel import render

    render(settings_loader=lambda _: settings)


@pytest.fixture
def dashboard_fixtures(tmp_path: Path) -> dict[str, object]:
    settings = _make_settings()
    document = DocumentInfo(
        source_path="docs/blogger_intro.pdf",
        collection="manuals",
        file_hash="f" * 64,
        chunk_count=2,
        image_count=0,
        updated_at="2026-05-10 18:00:00",
    )
    detail = DocumentDetail(
        source_path=document.source_path,
        collection=document.collection,
        file_hash=document.file_hash,
        chunk_count=2,
        image_count=0,
        updated_at=document.updated_at,
        chunks=[
            {"id": "chunk_0001", "text": "chunk text 1", "metadata": {"chunk_index": 0, "source_path": document.source_path}},
            {"id": "chunk_0002", "text": "chunk text 2", "metadata": {"chunk_index": 1, "source_path": document.source_path}},
        ],
        images=[],
    )
    data_service = _FakeDataService(
        documents=[document],
        detail=detail,
        collections=["manuals"],
        stats=CollectionStats(collection="manuals", document_count=1, chunk_count=2, image_count=0),
    )
    config_service = _FakeConfigService(settings=settings)
    trace_service = _FakeTraceService(
        ingestion_summaries=[
            TraceSummary(
                trace_id="ingestion-trace-1",
                trace_type="ingestion",
                started_at="2026-05-10 18:00:00",
                finished_at="2026-05-10 18:00:05",
                total_elapsed_ms=5230.0,
                stage_count=5,
                source_name="blogger_intro.pdf",
            )
        ],
        ingestion_detail=TraceDetail(
            trace_id="ingestion-trace-1",
            trace_type="ingestion",
            started_at="2026-05-10 18:00:00",
            finished_at="2026-05-10 18:00:05",
            total_elapsed_ms=5230.0,
            source_name="blogger_intro.pdf",
            stages=[
                StageTiming(stage="load", elapsed_ms=300.0, duration_ms=300.0),
                StageTiming(stage="split", elapsed_ms=900.0, duration_ms=600.0),
            ],
            raw_stages=[{"stage": "load", "payload": {}}, {"stage": "split", "payload": {}}],
        ),
        query_summaries=[
            QueryTraceSummary(
                trace_id="query-trace-1",
                trace_type="query",
                started_at="2026-05-10 18:10:00",
                finished_at="2026-05-10 18:10:02",
                total_elapsed_ms=2100.0,
                query_text="What is Modular RAG?",
                result_count=2,
            )
        ],
        query_detail=QueryTraceDetail(
            trace_id="query-trace-1",
            trace_type="query",
            started_at="2026-05-10 18:10:00",
            finished_at="2026-05-10 18:10:02",
            total_elapsed_ms=2100.0,
            query_text="What is Modular RAG?",
            keywords=["modular", "rag"],
            collection="manuals",
            rerank_enabled=True,
            rerank_applied=True,
            final_ids=["chunk_0001", "chunk_0002"],
            dense_ids=["chunk_0001"],
            sparse_ids=["chunk_0002"],
            fusion_ids=["chunk_0001", "chunk_0002"],
            rerank_input_ids=["chunk_0001", "chunk_0002"],
            rerank_result_ids=["chunk_0001", "chunk_0002"],
            stages=[
                StageTiming(stage="query_processing", elapsed_ms=120.0, duration_ms=0.0),
                StageTiming(stage="fusion", elapsed_ms=800.0, duration_ms=0.0),
            ],
            raw_stages=[{"stage": "query_processing", "payload": {}}, {"stage": "fusion", "payload": {}}],
        ),
    )
    return {
        "settings": settings,
        "config_service": config_service,
        "data_service": data_service,
        "trace_service": trace_service,
        "tmp_path": tmp_path,
    }


@pytest.mark.parametrize(
    ("wrapper", "arg_keys", "expected_title"),
    [
        (_render_overview, ("config_service",), "Modular RAG Dashboard"),
        (_render_data_browser, ("data_service",), "数据浏览器"),
        (_render_ingestion_manager, ("data_service", "settings"), "Ingestion 管理"),
        (_render_ingestion_traces, ("trace_service",), "Ingestion 追踪"),
        (_render_query_traces, ("trace_service",), "Query 追踪"),
        (_render_evaluation_panel, ("settings",), "评估面板"),
    ],
)
def test_dashboard_pages_smoke(
    dashboard_fixtures: dict[str, object],
    wrapper: object,
    arg_keys: tuple[str, ...],
    expected_title: str,
) -> None:
    args = tuple(dashboard_fixtures[key] for key in arg_keys)
    app = AppTest.from_function(wrapper, args=args, default_timeout=10)
    app.run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.title) == 1
    assert app.title[0].value == expected_title
