"""BM25 倒排索引构建与持久化。"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba

from core.types import ChunkRecord

# 关闭 jieba 的 debug 日志
jieba.setLogLevel(jieba.logging.INFO)

# 英文/数字 token 规则（中文交给 jieba）
ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class BM25IndexerError(RuntimeError):
    """BM25Indexer 可读错误。"""


@dataclass(slots=True)
class BM25QueryResult:
    """BM25 查询结果。"""

    chunk_id: str
    score: float


class BM25Indexer:
    """维护可落盘的 BM25 倒排索引。"""

    default_index_file = "bm25_index.pkl"

    def __init__(self, index_dir: str | Path = "data/db/bm25") -> None:
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / self.default_index_file
        self.documents: dict[str, dict[str, Any]] = {}
        self.index: dict[str, dict[str, Any]] = {}
        self.doc_count = 0

    def build(self, records: list[ChunkRecord], rebuild: bool = False) -> None:
        if rebuild:
            self.documents = {}

        for record in records:
            self.documents[record.id] = self._document_payload(record)

        self._rebuild_index()
        self._persist()

    def load(self) -> None:
        if not self.index_file.exists():
            raise BM25IndexerError(f"bm25 index file not found: {self.index_file}")

        with self.index_file.open("rb") as file:
            payload = pickle.load(file)

        if not isinstance(payload, dict):
            raise BM25IndexerError("bm25 index file is invalid")

        documents = payload.get("documents", {})
        index = payload.get("index", {})
        doc_count = payload.get("doc_count", 0)
        if not isinstance(documents, dict) or not isinstance(index, dict) or not isinstance(doc_count, int):
            raise BM25IndexerError("bm25 index payload is invalid")

        self.documents = documents
        self.index = index
        self.doc_count = doc_count

    def query(
        self,
        query: str | list[str],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[BM25QueryResult]:
        if not isinstance(top_k, int) or top_k <= 0:
            raise BM25IndexerError("bm25 query error: top_k must be positive int")
        if not self.index:
            return []

        tokens = self._tokenize(query) if isinstance(query, str) else [str(token).lower() for token in query if str(token)]
        scores: dict[str, float] = {}
        normalized_filters = self._normalize_filters(filters)

        for token in tokens:
            entry = self.index.get(token)
            if entry is None:
                continue

            postings = entry.get("postings", [])
            idf = self._query_idf(len(postings))
            for posting in postings:
                chunk_id = str(posting["chunk_id"])
                document = self.documents.get(chunk_id)
                if normalized_filters and not self._matches_filters(document, normalized_filters):
                    continue
                tf = float(posting["tf"])
                score = idf * tf
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [BM25QueryResult(chunk_id=chunk_id, score=round(score, 8)) for chunk_id, score in ranked[:top_k]]

    def remove_document(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise BM25IndexerError("bm25 remove error: source must be non-empty string")

        self.documents = {
            chunk_id: payload
            for chunk_id, payload in self.documents.items()
            if str(payload.get("source_path", "")) != source
        }
        self._rebuild_index()
        self._persist()

    def _rebuild_index(self) -> None:
        self.doc_count = len(self.documents)
        if self.doc_count == 0:
            self.index = {}
            return

        term_to_postings: dict[str, list[dict[str, Any]]] = {}
        for chunk_id, payload in self.documents.items():
            sparse_vector = payload["sparse_vector"]
            doc_length = int(payload["doc_length"])
            for term, tf in sparse_vector.items():
                term_to_postings.setdefault(term, []).append(
                    {"chunk_id": chunk_id, "tf": float(tf), "doc_length": doc_length}
                )

        rebuilt: dict[str, dict[str, Any]] = {}
        for term, postings in term_to_postings.items():
            df = len(postings)
            rebuilt[term] = {
                "idf": round(math.log((self.doc_count - df + 0.5) / (df + 0.5)), 8),
                "postings": sorted(postings, key=lambda item: item["chunk_id"]),
            }
        self.index = rebuilt

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "index": self.index,
            "doc_count": self.doc_count,
        }
        with self.index_file.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def _tokenize(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [str(token).lower() for token in value if str(token)]
        if not isinstance(value, str):
            raise BM25IndexerError("bm25 tokenize error: query must be string or list[str]")
        tokens: list[str] = []
        for word in jieba.cut(value):
            word = word.strip().lower()
            if not word:
                continue
            if not any(c.isalnum() or c == "_" or c == "-" for c in word):
                continue
            for match in ENGLISH_TOKEN_PATTERN.finditer(word):
                tokens.append(match.group().lower())
            if ENGLISH_TOKEN_PATTERN.fullmatch(word):
                continue
            if word and not ENGLISH_TOKEN_PATTERN.fullmatch(word):
                tokens.append(word)
        return tokens

    @staticmethod
    def _document_payload(record: ChunkRecord) -> dict[str, Any]:
        sparse_vector = record.sparse_vector or {}
        return {
            "chunk_id": record.id,
            "source_path": str(record.metadata.get("source_path", "")),
            "collection": str(record.metadata.get("collection", "")).strip(),
            "doc_length": int(record.metadata.get("sparse_doc_length", len(BM25Indexer._tokenize(record.text)))),
            "sparse_vector": {str(term): float(tf) for term, tf in sparse_vector.items()},
        }

    def _query_idf(self, df: int) -> float:
        if df <= 0 or self.doc_count <= 0:
            return 0.0
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    @staticmethod
    def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(filters, dict):
            return {}
        return {str(key): value for key, value in filters.items() if str(key).strip() and value not in (None, "")}

    @staticmethod
    def _matches_filters(document: dict[str, Any] | None, filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        if not isinstance(document, dict):
            return False
        return all(document.get(key) == value for key, value in filters.items())
