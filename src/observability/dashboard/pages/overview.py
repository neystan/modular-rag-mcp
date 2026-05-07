"""系统总览页。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


def collect_overview_data(config_service: ConfigService | None = None) -> dict[str, Any]:
    service = config_service or ConfigService()
    settings = service.get_settings()
    vector_store_config = dict(settings.vector_store)
    stats = _load_collection_stats(vector_store_config)
    return {
        "app": service.get_app_summary(),
        "components": service.get_component_cards(),
        "stats": stats,
        "raw_config": service.get_raw_config(),
    }


def render(config_service: ConfigService | None = None) -> None:
    data = collect_overview_data(config_service)

    st.title("Modular RAG Dashboard")
    st.caption("系统总览用于展示当前组件配置与知识库数据资产概况。")

    app_summary = data["app"]
    top_cols = st.columns(3)
    top_cols[0].metric("应用名称", app_summary["name"])
    top_cols[1].metric("运行环境", app_summary["environment"])
    top_cols[2].metric("配置文件", app_summary["settings_path"])

    st.subheader("组件配置")
    component_cols = st.columns(3)
    for index, card in enumerate(data["components"]):
        with component_cols[index % 3]:
            st.markdown(f"### {card.label}")
            st.caption(f"provider: {card.provider}")
            if card.model:
                st.write(f"model: `{card.model}`")
            for detail in card.details:
                st.write(detail)

    stats = data["stats"]
    st.subheader("数据资产统计")
    stats_cols = st.columns(4)
    stats_cols[0].metric("集合", stats["collection"])
    stats_cols[1].metric("文档数", stats["document_count"])
    stats_cols[2].metric("Chunk 数", stats["chunk_count"])
    stats_cols[3].metric("图片数", stats["image_count"])
    st.caption(f"向量库路径: {stats['persist_path']}")

    with st.expander("查看原始配置", expanded=False):
        st.json(data["raw_config"])


def _load_collection_stats(vector_store_config: dict[str, Any]) -> dict[str, Any]:
    provider = str(vector_store_config.get("provider", "")).strip().lower()
    if provider != "chroma":
        return {
            "collection": str(vector_store_config.get("collection", "unknown")),
            "document_count": 0,
            "chunk_count": 0,
            "image_count": 0,
            "persist_path": str(vector_store_config.get("persist_path", "")),
        }

    try:
        return ChromaStore(vector_store_config).get_collection_stats()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"读取 Chroma 统计失败: {exc}")
        return {
            "collection": str(vector_store_config.get("collection", "unknown")),
            "document_count": 0,
            "chunk_count": 0,
            "image_count": 0,
            "persist_path": str(vector_store_config.get("persist_path", "")),
        }
