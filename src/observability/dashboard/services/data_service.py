"""Dashboard 数据服务。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.settings import Settings, load_settings
from ingestion.document_manager import CollectionStats, DeleteResult, DocumentDetail, DocumentInfo, DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


class DataService:
    """封装 Dashboard 数据浏览与管理所需的文档操作。"""

    def __init__(
        self,
        settings_path: str | Path = "config/settings.yaml",
        *,
        document_manager: DocumentManager | None = None,
        settings_loader: Callable[[str | Path], Settings] = load_settings,
    ) -> None:
        self.settings_path = Path(settings_path)
        self._settings_loader = settings_loader
        self._document_manager = document_manager

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        return self._manager().list_documents(collection)

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        return self._manager().get_document_detail(doc_id)

    def list_collections(self) -> list[str]:
        collections = {item.collection for item in self._manager().list_documents()}
        return sorted(collections)

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        return self._manager().delete_document(source_path, collection)

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        return self._manager().get_collection_stats(collection)

    def _manager(self) -> DocumentManager:
        if self._document_manager is not None:
            return self._document_manager

        settings = self._settings_loader(self.settings_path)
        chroma = ChromaStore(dict(settings.vector_store))
        bm25 = BM25Indexer()
        image_storage = ImageStorage()
        integrity = SQLiteIntegrityChecker()
        self._document_manager = DocumentManager(chroma, bm25, image_storage, integrity)
        return self._document_manager
