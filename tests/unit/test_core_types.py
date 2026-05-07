"""核心数据契约单元测试。"""

from __future__ import annotations

import json

import pytest

from core import Chunk, ChunkRecord, Document, IMAGE_PLACEHOLDER_TEMPLATE, make_image_placeholder


def make_metadata() -> dict[str, object]:
    return {
        "source_path": "docs/sample.pdf",
        "images": [
            {
                "id": "abc123_1_0",
                "path": "data/images/demo/abc123_1_0.png",
                "page": 1,
                "text_offset": 25,
                "text_length": 19,
                "position": {"x": 12, "y": 34, "width": 200, "height": 100},
            }
        ],
        "collection": "demo",
    }


def test_document_roundtrip_is_json_serializable() -> None:
    document = Document(
        id="doc-1",
        text=f"正文开始 {make_image_placeholder('abc123_1_0')}",
        metadata=make_metadata(),
    )

    payload = document.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    restored = Document.from_dict(payload)

    assert restored == document
    assert payload["metadata"]["images"][0]["id"] == "abc123_1_0"


def test_chunk_roundtrip_preserves_offsets_and_source_ref() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="chunk body",
        metadata=make_metadata(),
        start_offset=10,
        end_offset=20,
        source_ref="doc-1",
    )

    payload = chunk.to_dict()
    restored = Chunk.from_dict(payload)

    assert restored == chunk
    assert payload["start_offset"] == 10
    assert payload["end_offset"] == 20
    assert payload["source_ref"] == "doc-1"


def test_chunk_record_roundtrip_preserves_dense_and_sparse_vectors() -> None:
    record = ChunkRecord(
        id="chunk-record-1",
        text="embedded chunk",
        metadata=make_metadata(),
        dense_vector=[0.1, 1, 2.5],
        sparse_vector={"token:a": 0.8, "token:b": 2},
    )

    payload = record.to_dict()
    restored = ChunkRecord.from_dict(payload)

    assert restored == record
    assert payload["dense_vector"] == [0.1, 1.0, 2.5]
    assert payload["sparse_vector"] == {"token:a": 0.8, "token:b": 2.0}


def test_chunk_record_accepts_serialized_images_metadata() -> None:
    metadata = make_metadata()
    metadata["images"] = json.dumps(metadata["images"], ensure_ascii=False)

    record = ChunkRecord(
        id="chunk-record-serialized",
        text="embedded chunk",
        metadata=metadata,
    )

    assert isinstance(record.metadata["images"], list)
    assert record.metadata["images"][0]["id"] == "abc123_1_0"


def test_metadata_requires_source_path() -> None:
    with pytest.raises(ValueError, match="metadata.source_path is required"):
        Document(id="doc-1", text="body", metadata={"collection": "demo"})


def test_metadata_images_must_follow_contract() -> None:
    with pytest.raises(ValueError, match="metadata.images\\[\\]\\.text_length must be positive int"):
        Document(
            id="doc-1",
            text="body",
            metadata={
                "source_path": "docs/sample.pdf",
                "images": [
                    {
                        "id": "abc123_1_0",
                        "path": "data/images/demo/abc123_1_0.png",
                        "text_offset": 0,
                        "text_length": 0,
                        "position": {},
                    }
                ],
            },
        )


def test_chunk_offsets_must_be_monotonic() -> None:
    with pytest.raises(ValueError, match="end_offset must be >= start_offset"):
        Chunk(
            id="chunk-1",
            text="body",
            metadata={"source_path": "docs/sample.pdf"},
            start_offset=8,
            end_offset=3,
        )


def test_make_image_placeholder_uses_spec_format() -> None:
    placeholder = make_image_placeholder("img-001")

    assert placeholder == "[IMAGE: img-001]"
    assert IMAGE_PLACEHOLDER_TEMPLATE == "[IMAGE: {image_id}]"
