"""PDF Loader 最小实现。"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from core.types import Document, make_image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError


logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """基于 pypdf 的 PDF 文本与图片引用加载器。"""

    def __init__(self, image_root: str | Path = "data/images") -> None:
        self.image_root = Path(image_root)

    def load(self, path: str | Path) -> Document:
        pdf_path = self._resolve_file(path)
        if pdf_path.suffix.lower() != ".pdf":
            raise LoaderError(f"unsupported file type: {pdf_path.suffix}")

        doc_hash = self._compute_sha256(pdf_path)
        reader = PdfReader(str(pdf_path))
        text = ""
        images: list[dict[str, Any]] = []

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            page_text = self._extract_page_text(page)
            text = self._append_fragment(text, page_text)
            text, page_images = self._extract_page_images(
                page=page,
                doc_hash=doc_hash,
                page_number=page_number,
                text=text,
            )
            images.extend(page_images)

        metadata = {
            "source_path": str(pdf_path),
            "doc_type": "pdf",
            "images": images,
        }
        return Document(id=doc_hash, text=text, metadata=metadata)

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _extract_page_text(page: Any) -> str:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF text extraction failed: %s", exc)
            return ""
        return text.strip()

    def _extract_page_images(
        self,
        page: Any,
        doc_hash: str,
        page_number: int,
        text: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        extracted: list[dict[str, Any]] = []
        try:
            page_images = list(getattr(page, "images", []) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF image enumeration failed on page %s: %s", page_number, exc)
            return text, extracted

        for image_index, image in enumerate(page_images):
            try:
                image_id = f"{doc_hash}_{page_number}_{image_index}"
                image_path = self._write_image(image=image, doc_hash=doc_hash, image_id=image_id)
                placeholder = make_image_placeholder(image_id)
                text_offset = self._next_fragment_offset(text)
                text = self._append_fragment(text, placeholder)
                extracted.append(
                    {
                        "id": image_id,
                        "path": str(image_path),
                        "page": page_number,
                        "text_offset": text_offset,
                        "text_length": len(placeholder),
                        "position": {},
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF image extraction failed on page %s: %s", page_number, exc)
        return text, extracted

    def _write_image(self, image: Any, doc_hash: str, image_id: str) -> Path:
        image_data = getattr(image, "data", None)
        if not isinstance(image_data, bytes) or not image_data:
            raise LoaderError("image data is empty")

        image_name = str(getattr(image, "name", "") or "")
        extension = Path(image_name).suffix.lower() or ".png"
        target_dir = self.image_root / doc_hash
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{image_id}{extension}"
        target_path.write_bytes(image_data)
        return target_path

    @staticmethod
    def _append_fragment(current: str, fragment: str) -> str:
        if not fragment:
            return current
        if not current:
            return fragment
        return f"{current}\n\n{fragment}"

    @staticmethod
    def _next_fragment_offset(current: str) -> int:
        if not current:
            return 0
        return len(current) + 2
