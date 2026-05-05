"""图片文件存储与索引。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_IMAGE_INDEX_DB = Path("data/db/image_index.db")


class ImageStorageError(RuntimeError):
    """ImageStorage 可读错误。"""


class ImageStorage:
    """保存图片文件，并用 SQLite 维护 image_id 映射。"""

    def __init__(
        self,
        image_root: str | Path = DEFAULT_IMAGE_ROOT,
        db_path: str | Path = DEFAULT_IMAGE_INDEX_DB,
    ) -> None:
        self.image_root = Path(image_root)
        self.db_path = Path(db_path)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def save_image(
        self,
        image_id: str,
        image_bytes: bytes,
        *,
        collection: str,
        doc_hash: str | None = None,
        page_num: int | None = None,
        suffix: str = ".png",
    ) -> Path:
        normalized_image_id = self._require_non_empty_str(image_id, "image_id")
        normalized_collection = self._require_non_empty_str(collection, "collection")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ImageStorageError("image_bytes is required")
        if page_num is not None and (not isinstance(page_num, int) or page_num < 0):
            raise ImageStorageError("page_num must be non-negative int or None")

        normalized_suffix = suffix if str(suffix).startswith(".") else f".{suffix}"
        target_dir = self.image_root / normalized_collection
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{normalized_image_id}{normalized_suffix}"
        target_path.write_bytes(bytes(image_bytes))

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num
                """,
                (
                    normalized_image_id,
                    str(target_path),
                    normalized_collection,
                    doc_hash,
                    page_num,
                ),
            )

        return target_path

    def get_image_path(self, image_id: str) -> Path | None:
        normalized_image_id = self._require_non_empty_str(image_id, "image_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (normalized_image_id,),
            ).fetchone()

        if row is None:
            return None
        return Path(str(row["file_path"]))

    def list_images(self, collection: str, doc_hash: str | None = None) -> list[dict[str, Any]]:
        normalized_collection = self._require_non_empty_str(collection, "collection")
        query = """
            SELECT image_id, file_path, collection, doc_hash, page_num, created_at
            FROM image_index
            WHERE collection = ?
        """
        params: list[Any] = [normalized_collection]
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            params.append(doc_hash)
        query += " ORDER BY image_id"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def delete_images(self, collection: str, doc_hash: str | None = None) -> int:
        items = self.list_images(collection, doc_hash)
        deleted = 0

        for item in items:
            file_path = Path(str(item["file_path"]))
            if file_path.exists():
                file_path.unlink()
            deleted += 1

        query = "DELETE FROM image_index WHERE collection = ?"
        params: list[Any] = [self._require_non_empty_str(collection, "collection")]
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            params.append(doc_hash)

        with self._connect() as connection:
            connection.execute(query, params)
        return deleted

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _require_non_empty_str(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ImageStorageError(f"{field_name} is required")
        return value.strip()
