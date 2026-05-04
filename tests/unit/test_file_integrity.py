"""文件完整性检查单元测试。"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from libs.loader.file_integrity import FileIntegrityError, SQLiteIntegrityChecker


def test_compute_sha256_is_stable_for_same_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("same content", encoding="utf-8")
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    first = checker.compute_sha256(sample)
    second = checker.compute_sha256(sample)

    assert first == second
    assert len(first) == 64


def test_mark_success_makes_should_skip_true(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "a" * 64

    assert checker.should_skip(file_hash) is False

    checker.mark_success(file_hash, "docs/sample.pdf", chunk_count=3)

    assert checker.should_skip(file_hash) is True
    record = checker.get_record(file_hash)
    assert record is not None
    assert record["file_path"] == "docs/sample.pdf"
    assert record["status"] == "success"
    assert record["metadata"] == {"chunk_count": 3}


def test_mark_failed_does_not_skip_and_can_be_replaced_by_success(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "b" * 64

    checker.mark_failed(file_hash, "parse failed")

    assert checker.should_skip(file_hash) is False
    assert checker.get_record(file_hash)["status"] == "failed"  # type: ignore[index]

    checker.mark_success(file_hash, "docs/retry.pdf")

    assert checker.should_skip(file_hash) is True
    record = checker.get_record(file_hash)
    assert record is not None
    assert record["status"] == "success"
    assert record["error_msg"] is None


def test_default_database_path_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    checker = SQLiteIntegrityChecker()

    assert checker.db_path == Path("data/db/ingestion_history.db")
    assert (tmp_path / checker.db_path).is_file()


def test_sqlite_uses_wal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    SQLiteIntegrityChecker(db_path)

    with sqlite3.connect(db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode == "wal"


def test_concurrent_mark_success_writes_are_supported(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    checker = SQLiteIntegrityChecker(db_path)

    def write_record(index: int) -> bool:
        file_hash = f"{index:064x}"
        local_checker = SQLiteIntegrityChecker(db_path)
        local_checker.mark_success(file_hash, f"docs/{index}.pdf", worker=index)
        return local_checker.should_skip(file_hash)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_record, range(20)))

    assert all(results)
    with sqlite3.connect(db_path) as connection:
        row_count = connection.execute("SELECT COUNT(*) FROM ingestion_history").fetchone()[0]
    assert row_count == 20
    assert checker.should_skip(f"{3:064x}") is True


def test_compute_sha256_missing_file_raises_clear_error(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    with pytest.raises(FileIntegrityError, match="file not found"):
        checker.compute_sha256(tmp_path / "missing.txt")
