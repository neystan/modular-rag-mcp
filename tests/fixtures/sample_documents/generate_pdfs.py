"""生成摄取测试所需的 PDF 样例文档。"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject, StreamObject


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 42
TOP = 760
LINE_HEIGHT = 14


@dataclass(slots=True)
class EmbeddedImage:
    name: str
    data: bytes
    width: int
    height: int
    x: int
    y: int
    render_width: int
    render_height: int


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_pdf(output_dir / "simple.pdf", _build_simple_pages())
    _write_pdf(output_dir / "with_images.pdf", _build_with_images_pages())
    _write_pdf(output_dir / "complex_technical_doc.pdf", _build_complex_pages())


def _write_pdf(target: Path, pages: list[tuple[list[str], list[EmbeddedImage]]]) -> None:
    writer = PdfWriter()
    font_ref = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )

    for lines, images in pages:
        page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        xobject_entries = DictionaryObject()
        image_commands: list[str] = []

        for index, image in enumerate(images):
            image_name = f"/Im{index}"
            xobject_entries[NameObject(image_name)] = writer._add_object(_make_image_stream(image))
            image_commands.append(
                f"q {image.render_width} 0 0 {image.render_height} {image.x} {image.y} cm {image_name} Do Q"
            )

        stream = DecodedStreamObject()
        stream.set_data(_build_content_stream(lines, image_commands))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
                NameObject("/ProcSet"): ArrayObject(
                    [
                        NameObject("/PDF"),
                        NameObject("/Text"),
                        NameObject("/ImageB"),
                        NameObject("/ImageC"),
                        NameObject("/ImageI"),
                    ]
                ),
                NameObject("/XObject"): xobject_entries,
            }
        )

    with target.open("wb") as file:
        writer.write(file)


def _build_content_stream(lines: list[str], image_commands: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", f"{LINE_HEIGHT} TL", f"{LEFT} {TOP} Td"]
    first_line = True
    for line in lines:
        escaped = _escape_pdf_text(line)
        if first_line:
            commands.append(f"({escaped}) Tj")
            first_line = False
        else:
            commands.append("T*")
            commands.append(f"({escaped}) Tj")
    commands.append("ET")
    commands.extend(image_commands)
    return "\n".join(commands).encode("utf-8")


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _make_image_stream(image: EmbeddedImage) -> StreamObject:
    stream = StreamObject()
    stream._data = zlib.compress(_png_to_rgb_bytes(image.data))
    stream.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(image.width),
            NameObject("/Height"): NumberObject(image.height),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
            NameObject("/Filter"): NameObject("/FlateDecode"),
        }
    )
    return stream


def _png_to_rgb_bytes(data: bytes) -> bytes:
    image = Image.open(BytesIO(data)).convert("RGB")
    return image.tobytes()


def _build_simple_pages() -> list[tuple[list[str], list[EmbeddedImage]]]:
    lines = [
        "Simple Modular RAG Fixture",
        "",
        "This PDF contains only plain text.",
        "It is used to verify the happy-path loader and pipeline flow.",
        "The content is intentionally short and deterministic.",
    ]
    return [(lines, [])]


def _build_with_images_pages() -> list[tuple[list[str], list[EmbeddedImage]]]:
    lines = [
        "PDF With Images Fixture",
        "",
        "The page includes one embedded architecture image.",
        "PdfLoader should extract the text and append one image placeholder.",
    ]
    image = _create_diagram_image(
        title="RAG Flow",
        bullets=["Load", "Split", "Embed", "Store"],
        accent=(42, 91, 215),
    )
    return [(lines, [EmbeddedImage("rag_flow.png", image, 220, 120, 320, 520, 220, 120)])]


def _build_complex_pages() -> list[tuple[list[str], list[EmbeddedImage]]]:
    images = [
        EmbeddedImage(
            "architecture.png",
            _create_diagram_image(
                title="Architecture",
                bullets=["Loader", "Chunker", "Transforms", "Stores"],
                accent=(21, 94, 117),
            ),
            220,
            120,
            330,
            520,
            220,
            120,
        ),
        EmbeddedImage(
            "retrieval.png",
            _create_diagram_image(
                title="Retrieval Path",
                bullets=["Dense", "Sparse", "Hybrid", "Rerank"],
                accent=(176, 76, 35),
            ),
            220,
            120,
            330,
            500,
            220,
            120,
        ),
        EmbeddedImage(
            "ops.png",
            _create_diagram_image(
                title="Ops Signals",
                bullets=["Trace", "Metrics", "Retry", "Audit"],
                accent=(74, 125, 56),
            ),
            220,
            120,
            330,
            500,
            220,
            120,
        ),
    ]

    page_1 = (
        [
            "Complex Technical Document",
            "",
            "## 1. System Overview",
            "The platform coordinates document loading, semantic chunking, and hybrid indexing.",
            "It keeps ingestion deterministic while exposing collection metadata for later retrieval.",
            "",
            "## 2. Data Contracts",
            "Every chunk carries source_path, collection, chunk_index, and optional image_refs.",
            "A stable SHA256 document identifier links vector, sparse, and image artifacts together.",
            "",
            "Table 1: Stage Contract",
            "| Stage | Input | Output |",
            "| load | PDF path | Document |",
            "| split | Document | Chunk[] |",
            "| store | Records | Index files |",
        ],
        [images[0]],
    )
    page_2 = (
        [
            "## 3. Retrieval Strategy",
            "Dense embeddings capture semantic similarity while BM25 protects exact-match recall.",
            "A lightweight rerank step can refine the final candidate set when latency budget allows it.",
            "",
            "## 4. Failure Handling",
            "Each pipeline stage wraps lower-level errors with a readable stage prefix for operators.",
            "Failures still mark file integrity status so repeated runs can be diagnosed quickly.",
            "",
            "Table 2: Retrieval Modes",
            "| Mode | Strength | Tradeoff |",
            "| dense | semantic recall | embedding cost |",
            "| sparse | keyword precision | lexical only |",
            "",
            "Table 3: Error Policy",
            "| Condition | Action |",
            "| loader exception | mark failed |",
            "| duplicate file | skip unless forced |",
        ],
        [images[1]],
    )
    page_3 = (
        [
            "## 5. Multi-Modal Enrichment",
            "Image placeholders preserve the location of diagrams so later captioning can fill in context.",
            "Metadata enrichment may append page-level hints, titles, and structured tags.",
            "",
            "## 6. Batch Encoding",
            "Dense and sparse encoders run over the same chunk list and return aligned record identifiers.",
            "Batch size is configurable from settings to balance throughput and external API rate limits.",
            "",
            "Table 4: Encoding Outputs",
            "| Encoder | Output |",
            "| dense | float vector |",
            "| sparse | token weight map |",
        ],
        [images[2]],
    )
    page_4 = (
        [
            "## 7. Storage Layout",
            "Image bytes persist under collection-scoped folders while SQLite stores image_id mappings.",
            "BM25 files live beside vector artifacts so offline ingest remains reproducible.",
            "",
            "## 8. Operations Checklist",
            "Before running ingest, confirm provider credentials, collection names, and source paths.",
            "After completion, validate chunk count, image count, and index presence in storage backends.",
            "",
            "Table 5: Runbook",
            "| Check | Expected |",
            "| logs | stage success messages |",
            "| images | extracted files exist |",
            "| indexes | vector and BM25 artifacts exist |",
        ],
        [],
    )
    return [page_1, page_2, page_3, page_4]


def _create_diagram_image(title: str, bullets: list[str], accent: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (220, 120), (248, 246, 240))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((6, 6, 214, 114), radius=16, outline=accent, width=3, fill=(255, 255, 255))
    draw.rectangle((16, 16, 204, 38), fill=accent)
    draw.text((24, 20), title, fill=(255, 255, 255))
    y = 48
    for bullet in bullets:
        draw.ellipse((24, y + 4, 30, y + 10), fill=accent)
        draw.text((38, y), bullet, fill=(40, 40, 40))
        y += 16

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
