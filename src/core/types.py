"""全链路共享的核心数据契约。负责规定项目里最核心的数据长什么样、
    该怎么校验、怎么序列化、怎么在不同模块之间稳定传递。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


IMAGE_PLACEHOLDER_TEMPLATE = "[IMAGE: {image_id}]"


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        raise ValueError("metadata is required")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be mapping/object")

    normalized = dict(metadata)
    source_path = normalized.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("metadata.source_path is required")

    if "images" in normalized:
        images = normalized["images"]
        if images is None:
            normalized["images"] = []
        else:
            normalized["images"] = [_normalize_image_metadata(image) for image in images]
    return normalized


def _normalize_image_metadata(image: Any) -> dict[str, Any]:
    if not isinstance(image, Mapping):
        raise TypeError("metadata.images[] must be mapping/object")

    normalized = dict(image)
    image_id = normalized.get("id")
    image_path = normalized.get("path")
    text_offset = normalized.get("text_offset")
    text_length = normalized.get("text_length")

    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("metadata.images[].id is required")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("metadata.images[].path is required")
    if not isinstance(text_offset, int) or text_offset < 0:
        raise ValueError("metadata.images[].text_offset must be non-negative int")
    if not isinstance(text_length, int) or text_length <= 0:
        raise ValueError("metadata.images[].text_length must be positive int")

    page = normalized.get("page")
    if page is not None and (not isinstance(page, int) or page < 0):
        raise ValueError("metadata.images[].page must be non-negative int or None")

    position = normalized.get("position")
    if position is None:
        normalized["position"] = {}
    elif not isinstance(position, Mapping):
        raise TypeError("metadata.images[].position must be mapping/object")
    else:
        normalized["position"] = dict(position)

    return normalized


def _normalize_float_list(values: list[float] | tuple[float, ...] | None, field_name: str) -> list[float] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be list/tuple of numbers")

    normalized: list[float] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{field_name}[{index}] must be number")
        normalized.append(float(value))
    return normalized


@dataclass(slots=True)
class Document:
    """原始文档契约。"""

    id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.id = _require_non_empty_str(self.id, "id")
        self.text = _require_str(self.text, "text")
        self.metadata = _normalize_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
        )


@dataclass(slots=True)
class Chunk:
    """切分后的文本块契约。"""

    id: str
    text: str
    metadata: dict[str, Any]
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def __post_init__(self) -> None:
        self.id = _require_non_empty_str(self.id, "id")
        self.text = _require_str(self.text, "text")
        self.metadata = _normalize_metadata(self.metadata)
        self.start_offset = _require_non_negative_int(self.start_offset, "start_offset")
        self.end_offset = _require_non_negative_int(self.end_offset, "end_offset")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be >= start_offset")
        if self.source_ref is not None:
            self.source_ref = _require_non_empty_str(self.source_ref, "source_ref")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_metadata(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            source_ref=data.get("source_ref"),
        )


@dataclass(slots=True)
class ChunkRecord:
    """用于存储与检索层的标准块记录。"""

    id: str
    text: str
    metadata: dict[str, Any]
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.id = _require_non_empty_str(self.id, "id")
        self.text = _require_str(self.text, "text")
        self.metadata = _normalize_metadata(self.metadata)
        self.dense_vector = _normalize_float_list(self.dense_vector, "dense_vector")
        self.sparse_vector = _normalize_sparse_vector(self.sparse_vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_metadata(self.metadata),
            "dense_vector": list(self.dense_vector) if self.dense_vector is not None else None,
            "sparse_vector": dict(self.sparse_vector) if self.sparse_vector is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkRecord":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )


@dataclass(slots=True)
class RetrievalResult:
    """统一检索结果契约。"""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.chunk_id = _require_non_empty_str(self.chunk_id, "chunk_id")
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be number")
        self.score = float(self.score)
        self.text = _require_str(self.text, "text")
        self.metadata = _normalize_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": _clone_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalResult":
        return cls(
            chunk_id=data["chunk_id"],
            score=data["score"],
            text=data["text"],
            metadata=data["metadata"],
        )


def make_image_placeholder(image_id: str) -> str:
    """生成文档内图片占位符。"""

    return IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=_require_non_empty_str(image_id, "image_id"))


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be string")
    return value


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be non-negative int")
    return value


def _normalize_sparse_vector(value: dict[str, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("sparse_vector must be mapping/object")

    normalized: dict[str, float] = {}
    for key, score in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sparse_vector keys must be non-empty strings")
        if not isinstance(score, (int, float)):
            raise TypeError(f"sparse_vector[{key!r}] must be number")
        normalized[key] = float(score)
    return normalized


def _clone_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cloned = dict(metadata)
    if "images" in cloned:
        cloned["images"] = [dict(image) for image in cloned["images"]]
    return cloned
