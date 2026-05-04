"""文件完整性检查。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_INGESTION_HISTORY_DB = Path("data/db/ingestion_history.db")
SUCCESS_STATUS = "success"
FAILED_STATUS = "failed"


class FileIntegrityError(RuntimeError):
    """文件完整性检查错误。"""


class FileIntegrityChecker(ABC):
    """文件完整性检查抽象接口。"""

    @abstractmethod
    def compute_sha256(self, path: str | Path) -> str:
        """计算文件 SHA256。"""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """判断文件 hash 是否已经成功处理过。"""

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str | Path, **metadata: Any) -> None:
        """标记文件处理成功。"""

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """标记文件处理失败。"""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的默认文件完整性检查器。"""

    def __init__(self, db_path: str | Path = DEFAULT_INGESTION_HISTORY_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def compute_sha256(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileIntegrityError(f"file not found: {file_path}")

        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        normalized_hash = self._normalize_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (normalized_hash,),
            ).fetchone()
        return row is not None and row["status"] == SUCCESS_STATUS

    def mark_success(self, file_hash: str, file_path: str | Path, **metadata: Any) -> None:
        normalized_hash = self._normalize_hash(file_hash)
        normalized_path = str(Path(file_path))
        now = self._utc_now()
        payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else None

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, error_msg, metadata_json, updated_at
                )
                VALUES (?, ?, ?, NULL, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = excluded.status,
                    error_msg = NULL,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (normalized_hash, normalized_path, SUCCESS_STATUS, payload, now),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        normalized_hash = self._normalize_hash(file_hash)
        if not isinstance(error_msg, str) or not error_msg.strip():
            raise FileIntegrityError("error_msg is required")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, error_msg, metadata_json, updated_at
                )
                VALUES (?, NULL, ?, ?, NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status = excluded.status,
                    error_msg = excluded.error_msg,
                    updated_at = excluded.updated_at
                """,
                (normalized_hash, FAILED_STATUS, error_msg, self._utc_now()),
            )

    def get_record(self, file_hash: str) -> dict[str, Any] | None:
        """读取单条历史记录，主要用于测试和诊断。"""

        normalized_hash = self._normalize_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_hash, file_path, status, error_msg, metadata_json, updated_at
                FROM ingestion_history
                WHERE file_hash = ?
                """,
                (normalized_hash,),
            ).fetchone()

        if row is None:
            return None
        record = dict(row)
        metadata_json = record.pop("metadata_json")
        record["metadata"] = json.loads(metadata_json) if metadata_json else {}
        return record

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                    error_msg TEXT,
                    metadata_json TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_history_status
                ON ingestion_history(status)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _normalize_hash(file_hash: str) -> str:
        if not isinstance(file_hash, str) or not file_hash.strip():
            raise FileIntegrityError("file_hash is required")
        return file_hash.strip().lower()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")
