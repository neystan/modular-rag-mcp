"""Streamlit Dashboard 入口。"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from observability.dashboard.pages.data_browser import render as render_data_browser
from observability.dashboard.pages.ingestion_manager import render as render_ingestion_manager
from observability.dashboard.pages.ingestion_traces import render as render_ingestion_traces
from observability.dashboard.pages.overview import render as render_overview


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    navigation = st.navigation(
        {
            "系统": [
                st.Page(render_overview, title="系统总览", url_path="overview", icon=":material/dashboard:"),
                st.Page(render_data_browser, title="数据浏览器", url_path="data-browser", icon=":material/folder_open:"),
            ],
            "操作": [
                st.Page(
                    render_ingestion_manager,
                    title="Ingestion 管理",
                    url_path="ingestion-manager",
                    icon=":material/upload_file:",
                ),
            ],
            "追踪": [
                st.Page(
                    render_ingestion_traces,
                    title="Ingestion 追踪",
                    url_path="ingestion-traces",
                    icon=":material/timeline:",
                ),
                st.Page(
                    _placeholder_page("Query 追踪", "G6 将在这里展示查询 Trace 历史与召回细节。"),
                    title="Query 追踪",
                    url_path="query-traces",
                    icon=":material/search:",
                ),
            ],
            "评估": [
                st.Page(
                    _placeholder_page("评估面板", "H4 将在这里接入评估任务运行与指标展示。"),
                    title="评估面板",
                    url_path="evaluation-panel",
                    icon=":material/analytics:",
                ),
            ],
        }
    )
    navigation.run()


def _placeholder_page(title: str, description: str) -> Callable[[], None]:
    def render() -> None:
        st.title(title)
        st.info(description)

    return render


if __name__ == "__main__":
    main()
