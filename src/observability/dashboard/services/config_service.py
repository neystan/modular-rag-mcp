"""Dashboard 配置读取服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.settings import Settings, load_settings


@dataclass(frozen=True)
class ComponentCard:
    """总览页展示的组件卡片。"""

    label: str
    provider: str
    model: str
    details: list[str]


class ConfigService:
    """封装 Dashboard 所需的配置读取和格式化逻辑。"""

    def __init__(
        self,
        settings_path: str | Path = "config/settings.yaml",
        *,
        settings_loader: Callable[[str | Path], Settings] = load_settings,
    ) -> None:
        self.settings_path = Path(settings_path)
        self._settings_loader = settings_loader

    def get_settings(self) -> Settings:
        return self._settings_loader(self.settings_path)

    def get_app_summary(self) -> dict[str, str]:
        settings = self.get_settings()
        return {
            "name": str(settings.app.get("name", "modular-rag-mcp")),
            "environment": str(settings.app.get("environment", "local")),
            "settings_path": str(self.settings_path),
        }

    def get_component_cards(self) -> list[ComponentCard]:
        settings = self.get_settings()
        sections = [
            ("LLM", settings.llm),
            ("Vision LLM", settings.vision_llm),
            ("Embedding", settings.embedding),
            ("Splitter", settings.splitter),
            ("Vector Store", settings.vector_store),
            ("Reranker", settings.rerank),
            ("Evaluation", settings.evaluation),
        ]

        cards: list[ComponentCard] = []
        for label, config in sections:
            if not isinstance(config, dict) or not config:
                continue
            provider = str(config.get("provider", "not-configured")).strip() or "not-configured"
            model = str(config.get("model", "")).strip()
            details = self._build_details(config)
            cards.append(ComponentCard(label=label, provider=provider, model=model, details=details))
        return cards

    def get_raw_config(self) -> dict[str, Any]:
        settings = self.get_settings()
        return {
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

    @staticmethod
    def _build_details(config: dict[str, Any]) -> list[str]:
        details: list[str] = []
        for key in ("model", "base_url", "collection", "persist_path", "top_k", "enabled"):
            value = config.get(key)
            if value in (None, ""):
                continue
            details.append(f"{key}: {value}")
        return details
