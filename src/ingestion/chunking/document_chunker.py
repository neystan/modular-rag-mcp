"""文档切分适配器。"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+?)\s*\]")


class DocumentChunker:
    """将 Document 对象转换为带业务元数据的 Chunk 列表。"""

    def __init__(self, settings: Settings | dict[str, Any], splitter: BaseSplitter | None = None) -> None:
        self.settings = settings
        self.splitter = splitter or SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        raw_chunks = self.splitter.split_text(document.text)
        chunks: list[Chunk] = []
        search_offset = 0

        for index, chunk_text in enumerate(raw_chunks):
            start_offset = self._find_chunk_offset(document.text, chunk_text, search_offset)
            end_offset = start_offset + len(chunk_text)
            search_offset = end_offset
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, chunk_text),
                    text=chunk_text,
                    metadata=self._inherit_metadata(document, index, chunk_text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{text_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict[str, Any]:
        metadata = copy.deepcopy(document.metadata)
        document_images = metadata.pop("images", [])
        metadata["chunk_index"] = chunk_index

        image_refs = self._extract_image_refs(chunk_text)
        if image_refs:
            image_by_id = {
                image["id"]: image
                for image in document_images
                if isinstance(image, dict) and isinstance(image.get("id"), str)
            }
            metadata["image_refs"] = image_refs
            metadata["images"] = [
                copy.deepcopy(image_by_id[image_id])
                for image_id in image_refs
                if image_id in image_by_id
            ]
        return metadata

    @staticmethod
    def _extract_image_refs(chunk_text: str) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for match in IMAGE_PLACEHOLDER_PATTERN.finditer(chunk_text):
            image_id = match.group(1).strip()
            if image_id and image_id not in seen:
                refs.append(image_id)
                seen.add(image_id)
        return refs

    @staticmethod
    def _find_chunk_offset(document_text: str, chunk_text: str, start_at: int) -> int:
        offset = document_text.find(chunk_text, start_at)
        if offset >= 0:
            return offset
        offset = document_text.find(chunk_text)
        if offset >= 0:
            return offset
        return start_at
