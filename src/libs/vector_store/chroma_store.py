"""Chroma 向量存储实现。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from libs.vector_store.base_vector_store import BaseVectorStore, VectorQueryResult, VectorRecord


class ChromaStore(BaseVectorStore):
    """基于 Chroma 的本地持久化向量存储。"""

    default_persist_path = "data/db/chroma"
    default_collection = "default"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        collection_name = str(self.config.get("collection", self.default_collection)).strip()
        if not collection_name:
            raise ValueError("chroma config error: collection is required")

        persist_path = Path(str(self.config.get("persist_path", self.default_persist_path))).expanduser()
        persist_path.mkdir(parents=True, exist_ok=True)
        self._persist_path = persist_path
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> int:
        if not records:
            return 0

        self._collection.upsert(
            ids=[record.id for record in records],
            embeddings=[record.vector for record in records],
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
        )
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        if not vector:
            raise ValueError("chroma query error: vector must be non-empty")
        if top_k <= 0:
            return []

        response = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[VectorQueryResult] = []
        for item_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            results.append(
                VectorQueryResult(
                    id=str(item_id),
                    score=float(-distance),
                    text=str(text or ""),
                    metadata=dict(metadata or {}),
                )
            )
        return results

    def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[dict[str, Any]]:
        normalized_ids = [str(item).strip() for item in ids if str(item).strip()]
        if not normalized_ids:
            return []

        response = self._collection.get(
            ids=normalized_ids,
            include=["documents", "metadatas"],
        )
        response_ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        by_id: dict[str, dict[str, Any]] = {}
        for item_id, text, metadata in zip(response_ids, documents, metadatas, strict=False):
            by_id[str(item_id)] = {
                "id": str(item_id),
                "text": str(text or ""),
                "metadata": dict(metadata or {}),
            }
        return [by_id[item_id] for item_id in normalized_ids if item_id in by_id]

    def get_collection_stats(self) -> dict[str, Any]:
        response = self._collection.get(include=["metadatas"])
        metadatas = response.get("metadatas", [])
        chunk_count = int(self._collection.count())
        source_paths: set[str] = set()
        image_count = 0

        for metadata in metadatas:
            normalized = dict(metadata or {})
            source_path = str(normalized.get("source_path", "")).strip()
            if source_path:
                source_paths.add(source_path)
            image_count += self._count_images(normalized.get("images"))

        return {
            "collection": self._collection_name,
            "chunk_count": chunk_count,
            "document_count": len(source_paths),
            "image_count": image_count,
            "persist_path": str(self._persist_path),
        }

    @staticmethod
    def _count_images(raw_images: Any) -> int:
        if isinstance(raw_images, list):
            return len(raw_images)
        if isinstance(raw_images, str) and raw_images.strip():
            try:
                decoded = json.loads(raw_images)
            except json.JSONDecodeError:
                return 0
            if isinstance(decoded, list):
                return len(decoded)
        return 0
