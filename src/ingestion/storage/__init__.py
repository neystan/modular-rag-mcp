"""摄取存储模块。"""

from ingestion.storage.bm25_indexer import BM25Indexer, BM25QueryResult

__all__ = [
    "BM25Indexer",
    "BM25QueryResult",
]
