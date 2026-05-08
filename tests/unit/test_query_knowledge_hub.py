"""query_knowledge_hub 工具测试。"""

from __future__ import annotations

from pathlib import Path

from core.types import RetrievalResult
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub


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
  top_k: 2
rerank:
  provider: none
evaluation:
  provider: custom
observability:
  log_level: INFO
"""


def test_query_knowledge_hub_uses_configured_top_k_when_argument_is_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_executor(query: str, top_k: int, collection: str | None) -> list[RetrievalResult]:
        calls.append({"query": query, "top_k": top_k, "collection": collection})
        return []

    payload = query_knowledge_hub(
        "retrieval strategy",
        executor=fake_executor,
        settings_path=config_path,
    )

    assert calls == [{"query": "retrieval strategy", "top_k": 2, "collection": None}]
    assert "未找到与“retrieval strategy”相关的文档" in payload["content"][0]["text"]
