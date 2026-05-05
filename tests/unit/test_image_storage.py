"""ImageStorage 单元测试。"""

from __future__ import annotations

from pathlib import Path

from ingestion.storage.image_storage import ImageStorage


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x89\x18\x8f"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_save_and_lookup_image_path(tmp_path: Path) -> None:
    storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )

    saved_path = storage.save_image(
        "img-1",
        PNG_BYTES,
        collection="manuals",
        doc_hash="doc-a",
        page_num=3,
    )

    assert saved_path.exists()
    assert saved_path.read_bytes() == PNG_BYTES
    assert storage.get_image_path("img-1") == saved_path


def test_mapping_is_persisted_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "image_index.db"
    image_root = tmp_path / "images"

    storage = ImageStorage(image_root=image_root, db_path=db_path)
    storage.save_image("img-1", PNG_BYTES, collection="manuals", doc_hash="doc-a")

    reloaded = ImageStorage(image_root=image_root, db_path=db_path)
    image_path = reloaded.get_image_path("img-1")

    assert image_path is not None
    assert image_path.exists()
    assert db_path.exists()


def test_list_images_filters_by_collection_and_doc_hash(tmp_path: Path) -> None:
    storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )
    storage.save_image("img-1", PNG_BYTES, collection="manuals", doc_hash="doc-a", page_num=1)
    storage.save_image("img-2", PNG_BYTES, collection="manuals", doc_hash="doc-b", page_num=2)
    storage.save_image("img-3", PNG_BYTES, collection="guides", doc_hash="doc-a", page_num=3)

    manuals = storage.list_images("manuals")
    only_doc_a = storage.list_images("manuals", "doc-a")

    assert [item["image_id"] for item in manuals] == ["img-1", "img-2"]
    assert [item["image_id"] for item in only_doc_a] == ["img-1"]


def test_delete_images_removes_files_and_rows(tmp_path: Path) -> None:
    storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )
    path_a = storage.save_image("img-1", PNG_BYTES, collection="manuals", doc_hash="doc-a")
    path_b = storage.save_image("img-2", PNG_BYTES, collection="manuals", doc_hash="doc-a")

    deleted = storage.delete_images("manuals", "doc-a")

    assert deleted == 2
    assert not path_a.exists()
    assert not path_b.exists()
    assert storage.list_images("manuals", "doc-a") == []
