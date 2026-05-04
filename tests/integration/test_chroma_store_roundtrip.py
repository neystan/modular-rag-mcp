"""ChromaStore roundtrip 集成测试。"""

from __future__ import annotations

from pathlib import Path

from core.settings import Settings
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


def make_settings(persist_path: Path) -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={
            "provider": "chroma",
            "collection": "test_collection",
            "persist_path": str(persist_path),
        },
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_factory_creates_chroma_store(tmp_path: Path) -> None:
    store = VectorStoreFactory.create(make_settings(tmp_path / "chroma"))

    assert isinstance(store, ChromaStore)


def test_chroma_roundtrip_supports_top_k_and_filters(tmp_path: Path) -> None:
    store = ChromaStore({"collection": "roundtrip", "persist_path": str(tmp_path / "db")})
    records = [
        VectorRecord(
            id="doc-1",
            vector=[1.0, 0.0],
            text="第一条文档",
            metadata={"collection": "alpha", "doc_type": "guide"},
        ),
        VectorRecord(
            id="doc-2",
            vector=[0.9, 0.1],
            text="第二条文档",
            metadata={"collection": "alpha", "doc_type": "faq"},
        ),
        VectorRecord(
            id="doc-3",
            vector=[0.0, 1.0],
            text="第三条文档",
            metadata={"collection": "beta", "doc_type": "guide"},
        ),
    ]

    assert store.upsert(records) == 3

    top_one = store.query([1.0, 0.0], top_k=1)
    assert len(top_one) == 1
    assert top_one[0].id == "doc-1"
    assert top_one[0].text == "第一条文档"

    alpha_only = store.query([1.0, 0.0], top_k=3, filters={"collection": "alpha"})
    assert [item.id for item in alpha_only] == ["doc-1", "doc-2"]

    faq_only = store.query([1.0, 0.0], top_k=3, filters={"doc_type": "faq"})
    assert [item.id for item in faq_only] == ["doc-2"]
    assert faq_only[0].metadata["doc_type"] == "faq"


def test_chroma_persists_records_across_instances(tmp_path: Path) -> None:
    persist_path = tmp_path / "persistent-db"
    first_store = ChromaStore({"collection": "persisted", "persist_path": str(persist_path)})
    first_store.upsert(
        [
            VectorRecord(
                id="doc-persisted",
                vector=[0.2, 0.8],
                text="持久化文档",
                metadata={"collection": "persist"},
            )
        ]
    )

    second_store = ChromaStore({"collection": "persisted", "persist_path": str(persist_path)})
    results = second_store.query([0.2, 0.8], top_k=1, filters={"collection": "persist"})

    assert [item.id for item in results] == ["doc-persisted"]
    assert results[0].text == "持久化文档"
