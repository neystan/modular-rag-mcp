"""PDF Loader 契约测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from core.types import Document
from libs.loader.base_loader import BaseLoader, LoaderError
from libs.loader.pdf_loader import PdfLoader


MINIMAL_TEXT_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 56 >>
stream
BT /F1 18 Tf 50 90 Td (Hello Modular RAG PDF) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000241 00000 n 
0000000311 00000 n 
trailer
<< /Root 1 0 R /Size 6 >>
startxref
416
%%EOF
"""

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sample_documents"


class FakeImage:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.data = data


class FakePage:
    def __init__(self, text: str, images: list[FakeImage] | None = None) -> None:
        self._text = text
        self.images = images or []

    def extract_text(self) -> str:
        return self._text


class BrokenImagePage:
    images = [FakeImage("broken.png", b"")]

    def extract_text(self) -> str:
        return "page text survives"


class FakeReader:
    def __init__(self, path: str) -> None:
        self.pages = [
            FakePage("page text", [FakeImage("diagram.png", b"fake-image-bytes")]),
        ]


class BrokenImageReader:
    def __init__(self, path: str) -> None:
        self.pages = [BrokenImagePage()]


def test_base_loader_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLoader()  # type: ignore[abstract]


def test_pdf_loader_loads_minimal_text_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    pdf_path.write_bytes(MINIMAL_TEXT_PDF)
    loader = PdfLoader(image_root=tmp_path / "images")

    document = loader.load(pdf_path)

    assert isinstance(document, Document)
    assert document.id == hashlib.sha256(MINIMAL_TEXT_PDF).hexdigest()
    assert "Hello Modular RAG PDF" in document.text
    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["images"] == []


def test_pdf_loader_extracts_images_and_inserts_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "with_images.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("libs.loader.pdf_loader.PdfReader", FakeReader)
    loader = PdfLoader(image_root=tmp_path / "images")

    document = loader.load(pdf_path)

    image = document.metadata["images"][0]
    placeholder = f"[IMAGE: {image['id']}]"
    image_path = Path(image["path"])
    assert document.text == f"page text\n\n{placeholder}"
    assert image["id"] == f"{document.id}_1_0"
    assert image["page"] == 1
    assert image["text_offset"] == len("page text\n\n")
    assert image["text_length"] == len(placeholder)
    assert image["position"] == {}
    assert image_path.is_file()
    assert image_path.read_bytes() == b"fake-image-bytes"


def test_pdf_loader_degrades_when_image_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pdf_path = tmp_path / "broken_images.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("libs.loader.pdf_loader.PdfReader", BrokenImageReader)
    loader = PdfLoader(image_root=tmp_path / "images")

    document = loader.load(pdf_path)

    assert document.text == "page text survives"
    assert document.metadata["images"] == []
    assert "PDF image extraction failed" in caplog.text


def test_pdf_loader_rejects_missing_and_non_pdf_files(tmp_path: Path) -> None:
    loader = PdfLoader(image_root=tmp_path / "images")

    with pytest.raises(LoaderError, match="file not found"):
        loader.load(tmp_path / "missing.pdf")

    text_path = tmp_path / "note.txt"
    text_path.write_text("not pdf", encoding="utf-8")
    with pytest.raises(LoaderError, match="unsupported file type"):
        loader.load(text_path)


def test_pdf_loader_loads_repository_fixture_documents(tmp_path: Path) -> None:
    loader = PdfLoader(image_root=tmp_path / "images")

    simple = loader.load(FIXTURE_DIR / "simple.pdf")
    complex_doc = loader.load(FIXTURE_DIR / "complex_technical_doc.pdf")
    with_images = loader.load(FIXTURE_DIR / "with_images.pdf")

    assert "Simple Modular RAG Fixture" in simple.text
    assert simple.metadata["images"] == []
    assert "## 1. System Overview" in complex_doc.text
    assert len(complex_doc.metadata["images"]) == 3
    assert "PDF With Images Fixture" in with_images.text
    assert len(with_images.metadata["images"]) == 1
