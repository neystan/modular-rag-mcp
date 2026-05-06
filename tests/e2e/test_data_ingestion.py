"""ingest.py e2e 测试。"""

from __future__ import annotations

import importlib.util
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any


def _load_ingest_module() -> Any:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("test_ingest_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest_module()


class FakePipeline:
    """测试用假 Pipeline，模拟落盘和跳过逻辑。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.root = Path.cwd()
        self.history_db = self.root / "data" / "db" / "ingestion_history.db"
        self.bm25_file = self.root / "data" / "db" / "bm25" / "bm25_index.pkl"
        self.image_db = self.root / "data" / "db" / "image_index.db"
        self.chroma_dir = self.root / "data" / "db" / "chroma"

    def run(self, path: str | Path, *, collection: str = "default", force: bool = False) -> Any:
        source = Path(path)
        file_hash = source.stem
        if not force and self._has_success(file_hash):
            return _Result(file_hash=file_hash, skipped=True)

        self._write_history(file_hash, source)
        self._write_bm25()
        self._write_image_index(collection, file_hash)
        self._write_chroma_placeholder()
        return _Result(file_hash=file_hash, skipped=False)

    def _has_success(self, file_hash: str) -> bool:
        if not self.history_db.exists():
            return False
        conn = sqlite3.connect(self.history_db)
        row = conn.execute(
            "SELECT status FROM ingestion_history WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        conn.close()
        return row is not None and row[0] == "success"

    def _write_history(self, file_hash: str, source: Path) -> None:
        self.history_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.history_db)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_history (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                status TEXT NOT NULL,
                error_msg TEXT,
                metadata_json TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, error_msg, metadata_json, updated_at)
            VALUES (?, ?, 'success', NULL, '{}', '2026-05-06T00:00:00+00:00')
            ON CONFLICT(file_hash) DO UPDATE SET
                file_path = excluded.file_path,
                status = excluded.status,
                error_msg = excluded.error_msg,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (file_hash, str(source)),
        )
        conn.commit()
        conn.close()

    def _write_bm25(self) -> None:
        self.bm25_file.parent.mkdir(parents=True, exist_ok=True)
        with self.bm25_file.open("wb") as file:
            pickle.dump({"documents": {}, "index": {}, "doc_count": 1}, file)

    def _write_image_index(self, collection: str, file_hash: str) -> None:
        self.image_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.image_db)
        conn.execute(
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
        conn.execute(
            """
            INSERT OR REPLACE INTO image_index (image_id, file_path, collection, doc_hash, page_num)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("img-1", str(self.root / "data" / "images" / "img-1.png"), collection, file_hash, 1),
        )
        conn.commit()
        conn.close()

    def _write_chroma_placeholder(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        (self.chroma_dir / "chroma.sqlite3").write_bytes(b"fake-chroma")


class _Result:
    def __init__(self, *, file_hash: str, skipped: bool) -> None:
        self.file_hash = file_hash
        self.collection = "test"
        self.document = None
        self.chunks = [] if skipped else [object()]
        self.dense_records = []
        self.sparse_records = []
        self.vector_records = [] if skipped else [object()]
        self.image_count = 0 if skipped else 1
        self.skipped = skipped


def _write_config(project_root: Path) -> None:
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(
        """
app:
  name: modular-rag-mcp
llm:
  provider: qwen
embedding:
  provider: qwen
splitter:
  provider: recursive
vector_store:
  provider: chroma
retrieval:
  top_k: 5
rerank:
  provider: none
evaluation:
  provider: custom
observability:
  log_level: INFO
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_ingest_cli_processes_directory_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    project_root = tmp_path
    fixtures_dir = project_root / "docs"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "first.pdf").write_bytes(b"%PDF-1.4 first")
    (fixtures_dir / "second.pdf").write_bytes(b"%PDF-1.4 second")
    _write_config(project_root)

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(ingest, "IngestionPipeline", FakePipeline)

    exit_code = ingest.main(["--path", str(fixtures_dir), "--collection", "manuals"])

    assert exit_code == 0
    assert (project_root / "data" / "db" / "bm25" / "bm25_index.pkl").exists()
    assert (project_root / "data" / "db" / "chroma" / "chroma.sqlite3").exists()
    assert (project_root / "data" / "db" / "image_index.db").exists()
    output = capsys.readouterr().out
    assert "ok" in output
    assert "processed=2 skipped=0 failed=0" in output


def test_ingest_cli_skips_repeated_run_without_force(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    project_root = tmp_path
    source_pdf = project_root / "docs" / "sample.pdf"
    source_pdf.parent.mkdir(parents=True, exist_ok=True)
    source_pdf.write_bytes(b"%PDF-1.4 sample")
    _write_config(project_root)

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(ingest, "IngestionPipeline", FakePipeline)

    first_code = ingest.main(["--path", str(source_pdf), "--collection", "manuals"])
    first_output = capsys.readouterr().out
    second_code = ingest.main(["--path", str(source_pdf), "--collection", "manuals"])
    second_output = capsys.readouterr().out

    assert first_code == 0
    assert second_code == 0
    assert "processed=1 skipped=0 failed=0" in first_output
    assert "skip" in second_output
    assert "processed=0 skipped=1 failed=0" in second_output
