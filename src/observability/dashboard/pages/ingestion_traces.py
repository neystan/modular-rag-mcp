"""Ingestion 追踪页。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import TraceDetail, TraceService


STAGE_LABELS = {
    "load": "文档加载",
    "split": "文本切分",
    "transform": "内容增强",
    "embed": "向量编码",
    "upsert": "索引写入",
}


def collect_ingestion_trace_data(
    trace_service: TraceService | None = None,
    *,
    selected_trace_id: str | None = None,
) -> dict[str, Any]:
    service = trace_service or TraceService()
    traces = service.list_ingestion_traces()
    active_trace_id = selected_trace_id or (traces[0].trace_id if traces else None)
    detail = service.get_ingestion_trace(active_trace_id) if active_trace_id else None
    return {
        "traces": traces,
        "selected_trace_id": active_trace_id,
        "detail": detail,
    }


def render(trace_service: TraceService | None = None) -> None:
    service = trace_service or TraceService()
    payload = collect_ingestion_trace_data(service)

    st.title("Ingestion 追踪")
    st.caption("查看摄取历史、阶段耗时和原始 Trace 明细。")

    traces = payload["traces"]
    if not traces:
        st.info("当前没有 ingestion trace。先执行摄取，再回到这里查看历史。")
        return

    selected_trace_id = st.selectbox(
        "选择 Trace",
        options=[item.trace_id for item in traces],
        format_func=lambda trace_id: _format_trace_option(trace_id, traces),
    )
    detail = service.get_ingestion_trace(selected_trace_id)

    st.subheader("历史列表")
    st.dataframe(
        [
            {
                "trace_id": item.trace_id,
                "started_at": item.started_at,
                "finished_at": item.finished_at,
                "total_elapsed_ms": item.total_elapsed_ms,
                "stage_count": item.stage_count,
            }
            for item in traces
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Trace 概览")
    top_cols = st.columns(4)
    top_cols[0].metric("Trace ID", detail.trace_id[:12])
    top_cols[1].metric("总耗时", f"{detail.total_elapsed_ms:.1f} ms")
    top_cols[2].metric("开始时间", detail.started_at)
    top_cols[3].metric("结束时间", detail.finished_at)

    st.subheader("阶段耗时")
    _render_stage_waterfall(detail)

    st.subheader("原始阶段明细")
    for stage in detail.raw_stages:
        stage_name = str(stage.get("stage", "unknown"))
        with st.expander(stage_name, expanded=stage_name in STAGE_LABELS):
            st.json(stage)


def _render_stage_waterfall(detail: TraceDetail) -> None:
    if not detail.stages:
        st.info("当前 Trace 没有可展示的标准阶段耗时。")
        return

    max_duration = max((item.duration_ms for item in detail.stages), default=0.0)
    for item in detail.stages:
        label = STAGE_LABELS.get(item.stage, item.stage)
        ratio = item.duration_ms / max_duration if max_duration > 0 else 0.0
        row_cols = st.columns([2, 6, 2])
        row_cols[0].write(label)
        row_cols[1].progress(ratio, text=f"{item.duration_ms:.1f} ms")
        row_cols[2].caption(f"累计 {item.elapsed_ms:.1f} ms")


def _format_trace_option(trace_id: str, traces: list[Any]) -> str:
    for item in traces:
        if item.trace_id == trace_id:
            return f"{item.started_at} · {item.total_elapsed_ms:.1f} ms · {trace_id[:12]}"
    return trace_id
