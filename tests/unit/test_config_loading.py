"""配置加载与校验测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, SettingsError, load_settings, validate_settings


VALID_CONFIG = """
app:
  name: modular-rag-mcp
llm:
  provider: placeholder
embedding:
  provider: placeholder
splitter:
  provider: placeholder
vector_store:
  provider: placeholder
retrieval:
  top_k: 5
rerank:
  provider: none
evaluation:
  provider: custom
observability:
  log_level: INFO
"""


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    settings = load_settings(config_path)

    assert isinstance(settings, Settings)
    assert settings.app["name"] == "modular-rag-mcp"
    assert settings.embedding["provider"] == "placeholder"


def test_load_settings_reports_missing_field_path(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        VALID_CONFIG.replace("embedding:\n  provider: placeholder\n", "embedding: {}\n"),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(config_path)


def test_validate_settings_rejects_missing_required_section() -> None:
    settings = Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={},
    )

    with pytest.raises(SettingsError, match="observability.log_level"):
        validate_settings(settings)


def test_load_settings_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="配置文件不存在"):
        load_settings(tmp_path / "missing.yaml")
