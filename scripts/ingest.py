"""离线数据摄取入口。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError, IngestionPipelineResult


@dataclass(slots=True)
class IngestSummary:
    """脚本执行汇总。"""

    processed: int = 0
    skipped: int = 0
    failed: int = 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="离线摄取 PDF 文档到本地索引")
    parser.add_argument("--collection", required=True, help="目标 collection 名称")
    parser.add_argument("--path", required=True, help="待摄取的 PDF 文件或目录")
    parser.add_argument("--force", action="store_true", help="忽略去重检查并强制重新摄取")
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_ingestion(args.path, collection=args.collection, force=args.force)
    except (ValueError, FileNotFoundError, IngestionPipelineError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ingest failed: unexpected error: {exc}", file=sys.stderr)
        return 1

    print(
        "ingest done: "
        f"processed={summary.processed} skipped={summary.skipped} failed={summary.failed}"
    )
    return 0 if summary.failed == 0 else 1


def run_ingestion(
    target_path: str | Path,
    *,
    collection: str,
    force: bool = False,
    pipeline: IngestionPipeline | None = None,
    settings_path: str | Path = "config/settings.yaml",
) -> IngestSummary:
    """执行单文件或目录摄取。"""

    normalized_collection = _normalize_collection(collection)
    source = Path(target_path).expanduser()
    paths = resolve_input_paths(source)
    active_pipeline = pipeline or IngestionPipeline(load_settings(settings_path))

    summary = IngestSummary()
    for path in paths:
        result = active_pipeline.run(path, collection=normalized_collection, force=force)
        _print_result(path, result)
        if result.skipped:
            summary.skipped += 1
        else:
            summary.processed += 1
    return summary


def resolve_input_paths(path: Path) -> list[Path]:
    """将输入解析为待处理 PDF 文件列表。"""

    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"unsupported file type: {path.suffix or '<none>'}")
        return [path]

    pdf_paths = sorted(item for item in path.rglob("*.pdf") if item.is_file())
    if not pdf_paths:
        raise ValueError(f"no pdf files found under: {path}")
    return pdf_paths


def _normalize_collection(collection: str) -> str:
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError("collection is required")
    return collection.strip()


def _print_result(path: Path, result: IngestionPipelineResult) -> None:
    if result.skipped:
        print(f"skip {path} file_hash={result.file_hash}")
        return

    print(
        f"ok {path} "
        f"chunks={len(result.chunks)} images={result.image_count} "
        f"vectors={len(result.vector_records)} file_hash={result.file_hash}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
