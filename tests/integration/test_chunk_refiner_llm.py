"""ChunkRefiner 真实 LLM 集成测试。"""

from __future__ import annotations

import os

import pytest

from core.settings import Settings, load_settings
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner


def _integration_settings() -> Settings:
    settings = load_settings("config/settings.yaml")
    provider = str(settings.llm.get("provider", "")).strip().lower()
    model = str(settings.llm.get("model", "")).strip()
    api_key = str(settings.llm.get("api_key", "")).strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()

    if not provider or provider == "placeholder" or not model or model == "placeholder":
        pytest.skip("真实 LLM 集成测试跳过：config/settings.yaml 中的 llm 尚未配置完成")
    if not api_key:
        pytest.skip("真实 LLM 集成测试跳过：缺少 llm.api_key 或 DASHSCOPE_API_KEY")
    if not settings.ingestion.get("chunk_refiner", {}).get("use_llm"):
        pytest.skip("真实 LLM 集成测试跳过：ingestion.chunk_refiner.use_llm 未开启")
    return settings


def test_chunk_refiner_real_llm_integration() -> None:
    settings = _integration_settings()
    refiner = ChunkRefiner(settings)
    chunk = Chunk(
        id="chunk-int",
        text="Header: Quarterly Report\n\nPage 1 of 5\n\nRevenue grew 18 percent year over year.\n\nFooter: Internal",
        metadata={"source_path": "docs/integration.pdf"},
        start_offset=0,
        end_offset=96,
        source_ref="doc-int",
    )

    result = refiner.transform([chunk])[0]

    assert result.metadata["refined_by"] in {"llm", "rule"}
    assert "Revenue grew 18 percent year over year." in result.text
    assert "Page 1 of 5" not in result.text
