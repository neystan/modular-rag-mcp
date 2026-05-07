"""多模态内容组装。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from core.types import RetrievalResult


class MultimodalAssembler:
    """将检索结果中的图片引用组装为 MCP image content。"""

    def assemble(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in retrieval_results:
            images = result.metadata.get("images", [])
            if not isinstance(images, list):
                continue

            for image in images:
                if not isinstance(image, dict):
                    continue

                image_id = str(image.get("id", "")).strip()
                image_path = str(image.get("path", "")).strip()
                dedupe_key = image_id or image_path
                if not image_path or not dedupe_key or dedupe_key in seen:
                    continue

                path = Path(image_path)
                if not path.exists() or not path.is_file():
                    continue

                mime_type, _ = mimetypes.guess_type(path.name)
                if not mime_type:
                    mime_type = "application/octet-stream"

                contents.append(
                    {
                        "type": "image",
                        "mimeType": mime_type,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                )
                seen.add(dedupe_key)

        return contents
