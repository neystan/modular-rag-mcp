"""MetadataEnricher 真实 LLM 集成测试。"""

from __future__ import annotations

import os

import pytest

from core.settings import Settings, load_settings
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher


def _integration_settings() -> Settings:
    settings = load_settings("config/settings.yaml")
    provider = str(settings.llm.get("provider", "")).strip().lower()
    model = str(settings.llm.get("model", "")).strip()
    api_key = str(settings.llm.get("api_key", "")).strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
    metadata_enricher = settings.ingestion.get("metadata_enricher", {})
    use_llm = True if "use_llm" not in metadata_enricher else bool(metadata_enricher.get("use_llm"))

    if not provider or provider == "placeholder" or not model or model == "placeholder":
        pytest.skip("真实 LLM 集成测试跳过：config/settings.yaml 中的 llm 尚未配置完成")
    if not api_key:
        pytest.skip("真实 LLM 集成测试跳过：缺少 llm.api_key 或 DASHSCOPE_API_KEY")
    if not use_llm:
        pytest.skip("真实 LLM 集成测试跳过：ingestion.metadata_enricher.use_llm 未开启")
    return settings


def test_metadata_enricher_real_llm_integration() -> None:
    settings = _integration_settings()
    enricher = MetadataEnricher(settings)
    chunk = Chunk(
        id="chunk-int",
        text=(
            "Qwen metadata enrichment improves downstream retrieval quality by generating a precise title, "
            "a compact summary, and high-signal topical tags for each chunk."
        ),
        metadata={"source_path": "docs/integration.pdf"},
        start_offset=0,
        end_offset=156,
        source_ref="doc-int",
    )

    result = enricher.transform([chunk])[0]

    assert result.metadata["metadata_enriched_by"] in {"llm", "rule"}
    assert isinstance(result.metadata["title"], str) and result.metadata["title"]
    assert isinstance(result.metadata["summary"], str) and result.metadata["summary"]
    assert isinstance(result.metadata["tags"], list) and result.metadata["tags"]
