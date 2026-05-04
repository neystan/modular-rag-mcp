"""DocumentChunker 单元测试。"""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """测试用 splitter，按配置的分隔符拆分。"""

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        separator = str(self.config.get("separator", "|"))
        return [part for part in text.split(separator) if part]


def make_settings(separator: str = "|") -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "fake_document_chunker", "separator": separator},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_document() -> Document:
    return Document(
        id="doc-123",
        text="intro [IMAGE: img_a]|middle text|tail [IMAGE: img_b] [IMAGE: img_a]",
        metadata={
            "source_path": "docs/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample",
            "images": [
                {
                    "id": "img_a",
                    "path": "data/images/doc/img_a.png",
                    "page": 1,
                    "text_offset": 6,
                    "text_length": 14,
                    "position": {"x": 1},
                },
                {
                    "id": "img_b",
                    "path": "data/images/doc/img_b.png",
                    "page": 2,
                    "text_offset": 40,
                    "text_length": 14,
                    "position": {"x": 2},
                },
            ],
        },
    )


def test_split_document_uses_splitter_factory_and_settings() -> None:
    SplitterFactory.clear_providers()
    SplitterFactory.register_provider("fake_document_chunker", FakeSplitter)
    chunker = DocumentChunker(make_settings(separator="::"))
    document = Document(
        id="doc-config",
        text="a::b::c",
        metadata={"source_path": "docs/config.pdf"},
    )

    chunks = chunker.split_document(document)

    assert [chunk.text for chunk in chunks] == ["a", "b", "c"]


def test_split_document_outputs_chunk_contract_with_stable_unique_ids() -> None:
    chunker = DocumentChunker(make_settings(), splitter=FakeSplitter({"separator": "|"}))
    document = make_document()

    first = chunker.split_document(document)
    second = chunker.split_document(document)

    assert all(isinstance(chunk, Chunk) for chunk in first)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    assert first[0].id.startswith("doc-123_0000_")
    assert first[1].id.startswith("doc-123_0001_")
    assert first[2].id.startswith("doc-123_0002_")
    assert [chunk.source_ref for chunk in first] == ["doc-123", "doc-123", "doc-123"]


def test_metadata_is_inherited_and_chunk_index_is_added() -> None:
    chunker = DocumentChunker(make_settings(), splitter=FakeSplitter({"separator": "|"}))

    chunks = chunker.split_document(make_document())

    assert chunks[1].metadata["source_path"] == "docs/sample.pdf"
    assert chunks[1].metadata["doc_type"] == "pdf"
    assert chunks[1].metadata["title"] == "Sample"
    assert chunks[1].metadata["chunk_index"] == 1
    assert "images" not in chunks[1].metadata
    assert "image_refs" not in chunks[1].metadata


def test_images_are_distributed_by_chunk_placeholders_only() -> None:
    chunker = DocumentChunker(make_settings(), splitter=FakeSplitter({"separator": "|"}))

    chunks = chunker.split_document(make_document())

    assert chunks[0].metadata["image_refs"] == ["img_a"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["img_a"]
    assert chunks[2].metadata["image_refs"] == ["img_b", "img_a"]
    assert [image["id"] for image in chunks[2].metadata["images"]] == ["img_b", "img_a"]


def test_offsets_follow_original_document_text() -> None:
    chunker = DocumentChunker(make_settings(), splitter=FakeSplitter({"separator": "|"}))
    document = make_document()

    chunks = chunker.split_document(document)

    for chunk in chunks:
        assert document.text[chunk.start_offset : chunk.end_offset] == chunk.text
        assert chunk.to_dict()["source_ref"] == "doc-123"


def test_duplicate_image_placeholders_are_deduplicated_per_chunk() -> None:
    chunker = DocumentChunker(make_settings(), splitter=FakeSplitter({"separator": "|"}))
    document = Document(
        id="doc-dup",
        text="[IMAGE: img_a] repeated [IMAGE: img_a]",
        metadata={
            "source_path": "docs/dup.pdf",
            "images": [
                {
                    "id": "img_a",
                    "path": "data/images/doc/img_a.png",
                    "page": 1,
                    "text_offset": 0,
                    "text_length": 14,
                    "position": {},
                }
            ],
        },
    )

    chunks = chunker.split_document(document)

    assert chunks[0].metadata["image_refs"] == ["img_a"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["img_a"]
