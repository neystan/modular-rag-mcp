"""配置加载与最小校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class SettingsError(ValueError):
    """配置文件错误。"""


@dataclass(frozen=True)
class Settings:
    """项目启动所需的最小配置结构。"""

    app: dict[str, Any]
    llm: dict[str, Any]
    embedding: dict[str, Any]
    splitter: dict[str, Any]
    vector_store: dict[str, Any]
    retrieval: dict[str, Any]
    rerank: dict[str, Any]
    evaluation: dict[str, Any]
    observability: dict[str, Any]
    ingestion: dict[str, Any] = field(default_factory=dict)
    vision_llm: dict[str, Any] = field(default_factory=dict)


REQUIRED_SECTIONS = (
    "app",
    "llm",
    "embedding",
    "splitter",
    "vector_store",
    "retrieval",
    "rerank",
    "evaluation",
    "observability",
)

REQUIRED_FIELDS = (
    "app.name",
    "llm.provider",
    "embedding.provider",
    "splitter.provider",
    "vector_store.provider",
    "retrieval.top_k",
    "rerank.provider",
    "evaluation.provider",
    "observability.log_level",
)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """读取 YAML 配置并返回已校验的 Settings。"""

    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"配置文件不存在: {settings_path}")

    try:
        raw_data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"配置文件 YAML 解析失败: {settings_path}") from exc

    if raw_data is None:
        raw_data = {}
    if not isinstance(raw_data, dict):
        raise SettingsError("配置文件根节点必须是 mapping/object")

    settings = Settings(
        app=_section(raw_data, "app"),
        llm=_section(raw_data, "llm"),
        vision_llm=_section(raw_data, "vision_llm"),
        embedding=_section(raw_data, "embedding"),
        splitter=_section(raw_data, "splitter"),
        vector_store=_section(raw_data, "vector_store"),
        retrieval=_section(raw_data, "retrieval"),
        rerank=_section(raw_data, "rerank"),
        evaluation=_section(raw_data, "evaluation"),
        observability=_section(raw_data, "observability"),
        ingestion=_section(raw_data, "ingestion"),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """校验必填配置字段，错误信息包含字段路径。"""

    data = {
        "app": settings.app,
        "llm": settings.llm,
        "vision_llm": settings.vision_llm,
        "embedding": settings.embedding,
        "splitter": settings.splitter,
        "vector_store": settings.vector_store,
        "retrieval": settings.retrieval,
        "rerank": settings.rerank,
        "evaluation": settings.evaluation,
        "observability": settings.observability,
        "ingestion": settings.ingestion,
    }

    for section in REQUIRED_SECTIONS:
        if section not in data or not isinstance(data[section], dict):
            raise SettingsError(f"缺少配置项: {section}")

    for field_path in REQUIRED_FIELDS:
        if _get_path(data, field_path) in (None, ""):
            raise SettingsError(f"缺少配置项: {field_path}")


def _section(raw_data: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw_data.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SettingsError(f"配置项必须是 mapping/object: {name}")
    return value


def _get_path(data: dict[str, Any], field_path: str) -> Any:
    current: Any = data
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
