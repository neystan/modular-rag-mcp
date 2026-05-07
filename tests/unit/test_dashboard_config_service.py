"""Dashboard 总览配置服务测试。"""

from __future__ import annotations

import json
from pathlib import Path

from core.settings import Settings
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.pages.overview import collect_overview_data
from observability.dashboard.services.config_service import ConfigService


def make_settings(persist_path: Path) -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp", "environment": "local"},
        llm={"provider": "qwen", "model": "qwen3"},
        vision_llm={"provider": "qwen-vision", "model": "qwen-vl"},
        embedding={"provider": "qwen", "model": "text-embedding"},
        splitter={"provider": "recursive"},
        vector_store={"provider": "chroma", "collection": "overview", "persist_path": str(persist_path)},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom", "enabled": False},
        observability={"log_level": "INFO"},
        ingestion={},
    )


def test_config_service_returns_component_cards(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "chroma")
    service = ConfigService(settings_path=tmp_path / "settings.yaml", settings_loader=lambda _: settings)

    cards = service.get_component_cards()
    summary = service.get_app_summary()

    assert summary["name"] == "modular-rag-mcp"
    assert any(card.label == "LLM" and card.provider == "qwen" for card in cards)
    assert any(card.label == "Vector Store" and card.provider == "chroma" for card in cards)


def test_collect_overview_data_includes_chroma_stats(tmp_path: Path) -> None:
    persist_path = tmp_path / "db"
    store = ChromaStore({"collection": "overview", "persist_path": str(persist_path)})
    store.upsert(
        [
            VectorRecord(
                id="chunk-1",
                vector=[1.0, 0.0],
                text="doc a chunk 1",
                metadata={
                    "source_path": "docs/a.pdf",
                    "collection": "overview",
                    "images": json.dumps([{"id": "img-1"}, {"id": "img-2"}]),
                },
            ),
            VectorRecord(
                id="chunk-2",
                vector=[0.8, 0.2],
                text="doc b chunk 1",
                metadata={"source_path": "docs/b.pdf", "collection": "overview"},
            ),
        ]
    )
    settings = make_settings(persist_path)
    service = ConfigService(settings_path=tmp_path / "settings.yaml", settings_loader=lambda _: settings)

    payload = collect_overview_data(service)

    assert payload["stats"]["collection"] == "overview"
    assert payload["stats"]["document_count"] == 2
    assert payload["stats"]["chunk_count"] == 2
    assert payload["stats"]["image_count"] == 2
