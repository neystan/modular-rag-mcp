"""VectorStore 契约测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorQueryResult,
    VectorRecord,
)
from libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
    VectorStoreFactoryError,
)


class FakeVectorStore(BaseVectorStore):
    """测试用内存向量存储。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        for record in records:
            self.records[record.id] = record
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        filters = filters or {}
        matched_records = [
            record
            for record in self.records.values()
            if all(record.metadata.get(key) == value for key, value in filters.items())
        ]
        results = [
            VectorQueryResult(
                id=record.id,
                score=sum(vector) + sum(record.vector),
                text=record.text,
                metadata=record.metadata,
            )
            for record in matched_records
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[dict[str, Any]]:
        del trace
        normalized_ids = [str(item).strip() for item in ids if str(item).strip()]
        return [
            {
                "id": record.id,
                "text": record.text,
                "metadata": dict(record.metadata),
            }
            for item_id in normalized_ids
            if (record := self.records.get(item_id)) is not None
        ]


class QueryOnlyVectorStore(BaseVectorStore):
    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        del records, trace
        return 0

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        del vector, top_k, filters, trace
        return []


class NotVectorStore:
    pass


@pytest.fixture(autouse=True)
def clear_vector_store_registry() -> None:
    VectorStoreFactory.clear_providers()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": provider, "collection": "test"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_vector_record_shape() -> None:
    record = VectorRecord(
        id="chunk-1",
        vector=[0.1, 0.2],
        text="测试文本",
        metadata={"source": "sample.md"},
    )

    assert record.id == "chunk-1"
    assert record.vector == [0.1, 0.2]
    assert record.metadata["source"] == "sample.md"


def test_upsert_and_query_contract() -> None:
    store = FakeVectorStore({"collection": "test"})
    records = [
        VectorRecord(id="a", vector=[1.0, 0.0], text="文本 A", metadata={"collection": "test"}),
        VectorRecord(id="b", vector=[0.0, 2.0], text="文本 B", metadata={"collection": "test"}),
        VectorRecord(id="c", vector=[10.0], text="文本 C", metadata={"collection": "other"}),
    ]

    assert store.upsert(records) == 3
    results = store.query([1.0], top_k=1, filters={"collection": "test"})

    assert len(results) == 1
    assert isinstance(results[0], VectorQueryResult)
    assert results[0].id == "b"
    assert isinstance(results[0].score, float)
    assert results[0].metadata["collection"] == "test"


def test_get_by_ids_preserves_requested_order() -> None:
    store = FakeVectorStore({"collection": "test"})
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0], text="文本 A", metadata={}),
            VectorRecord(id="b", vector=[2.0], text="文本 B", metadata={}),
        ]
    )

    results = store.get_by_ids(["b", "missing", "a"])

    assert results == [
        {"id": "b", "text": "文本 B", "metadata": {}},
        {"id": "a", "text": "文本 A", "metadata": {}},
    ]


def test_base_vector_store_get_by_ids_raises_when_not_overridden() -> None:
    store = QueryOnlyVectorStore()

    with pytest.raises(NotImplementedError, match="get_by_ids"):
        store.get_by_ids(["a"])


def test_register_provider_and_create_from_settings() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)

    store = VectorStoreFactory.create(make_settings())

    assert isinstance(store, FakeVectorStore)
    assert store.config["collection"] == "test"


def test_create_from_dict_uses_vector_store_section() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)

    store = VectorStoreFactory.create({"vector_store": {"provider": "fake", "collection": "dict"}})

    assert isinstance(store, FakeVectorStore)
    assert store.config["collection"] == "dict"


def test_unknown_provider_reports_available_providers() -> None:
    VectorStoreFactory.register_provider("fake", FakeVectorStore)

    with pytest.raises(VectorStoreFactoryError, match="未知 VectorStore provider: missing"):
        VectorStoreFactory.create(make_settings(provider="missing"))


def test_missing_provider_reports_config_path() -> None:
    with pytest.raises(VectorStoreFactoryError, match="vector_store.provider"):
        VectorStoreFactory.create({"vector_store": {"collection": "test"}})


def test_register_provider_requires_basevectorstore_subclass() -> None:
    with pytest.raises(VectorStoreFactoryError, match="必须继承 BaseVectorStore"):
        VectorStoreFactory.register_provider("bad", NotVectorStore)  # type: ignore[arg-type]


def test_create_requires_settings_or_dict() -> None:
    with pytest.raises(VectorStoreFactoryError, match="settings 必须是 Settings 或 dict"):
        VectorStoreFactory.create("bad-settings")  # type: ignore[arg-type]
