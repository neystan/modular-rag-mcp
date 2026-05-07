"""Query 追踪页。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import QueryTraceDetail, TraceService


STAGE_LABELS = {
    "query_processing": "查询预处理",
    "dense_retrieval": "Dense 检索",
    "sparse_retrieval": "Sparse 检索",
    "fusion": "融合排序",
    "rerank": "重排序",
}


def collect_query_trace_data(
    trace_service: TraceService | None = None,
    *,
    query_keyword: str | None = None,
    selected_trace_id: str | None = None,
) -> dict[str, Any]:
    service = trace_service or TraceService()
    traces = service.list_query_traces(query_keyword)
    active_trace_id = selected_trace_id or (traces[0].trace_id if traces else None)
    detail = service.get_query_trace(active_trace_id) if active_trace_id else None
    return {
        "traces": traces,
        "selected_trace_id": active_trace_id,
        "detail": detail,
    }


def render(trace_service: TraceService | None = None) -> None:
    service = trace_service or TraceService()

    st.title("Query 追踪")
    st.caption("查看查询历史、Dense/Sparse 召回差异和 Rerank 前后变化。")

    query_keyword = st.text_input("按 Query 关键词过滤", value="")
    payload = collect_query_trace_data(service, query_keyword=query_keyword)
    traces = payload["traces"]
    if not traces:
        st.info("当前没有 query trace。先执行 query，再回到这里查看历史。")
        return

    selected_trace_id = st.selectbox(
        "选择 Trace",
        options=[item.trace_id for item in traces],
        format_func=lambda trace_id: _format_trace_option(trace_id, traces),
    )
    detail = service.get_query_trace(selected_trace_id)

    st.subheader("历史列表")
    st.dataframe(
        [
            {
                "trace_id": item.trace_id,
                "query_text": item.query_text,
                "started_at": item.started_at,
                "total_elapsed_ms": item.total_elapsed_ms,
                "result_count": item.result_count,
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
    top_cols[2].metric("最终结果数", len(detail.final_ids))
    top_cols[3].metric("集合过滤", detail.collection or "<none>")
    st.markdown(f"**Query**: `{detail.query_text}`")
    st.caption(f"关键词: {', '.join(detail.keywords) or '<none>'}")

    st.subheader("阶段耗时")
    _render_stage_elapsed(detail)

    st.subheader("Dense vs Sparse 对比")
    compare_cols = st.columns(2)
    compare_cols[0].markdown("**Dense Results**")
    compare_cols[0].dataframe(_rank_rows(detail.dense_ids), use_container_width=True, hide_index=True)
    compare_cols[1].markdown("**Sparse Results**")
    compare_cols[1].dataframe(_rank_rows(detail.sparse_ids), use_container_width=True, hide_index=True)

    st.subheader("Rerank 变化")
    before_ids = detail.rerank_input_ids or detail.fusion_ids
    after_ids = detail.rerank_result_ids or detail.final_ids
    st.dataframe(_rerank_rows(before_ids, after_ids), use_container_width=True, hide_index=True)
    if detail.rerank_enabled:
        st.caption(f"rerank_applied={detail.rerank_applied}")
    else:
        st.caption("本次查询未启用 rerank。")

    with st.expander("查看原始 Trace", expanded=False):
        st.json(detail.raw_stages)


def _render_stage_elapsed(detail: QueryTraceDetail) -> None:
    if not detail.stages:
        st.info("当前 Trace 没有可展示的标准阶段。")
        return
    for item in detail.stages:
        row_cols = st.columns([2, 6, 2])
        row_cols[0].write(STAGE_LABELS.get(item.stage, item.stage))
        row_cols[1].progress(1.0, text=f"{item.elapsed_ms:.1f} ms")
        row_cols[2].caption(f"累计 {item.elapsed_ms:.1f} ms")


def _rank_rows(chunk_ids: list[str]) -> list[dict[str, Any]]:
    return [{"rank": index, "chunk_id": chunk_id} for index, chunk_id in enumerate(chunk_ids, start=1)] or [{"rank": "-", "chunk_id": "<empty>"}]


def _rerank_rows(before_ids: list[str], after_ids: list[str]) -> list[dict[str, Any]]:
    union_ids = list(dict.fromkeys([*before_ids, *after_ids]))
    rows: list[dict[str, Any]] = []
    for chunk_id in union_ids:
        before_rank = before_ids.index(chunk_id) + 1 if chunk_id in before_ids else "-"
        after_rank = after_ids.index(chunk_id) + 1 if chunk_id in after_ids else "-"
        rows.append({"chunk_id": chunk_id, "before_rank": before_rank, "after_rank": after_rank})
    return rows or [{"chunk_id": "<empty>", "before_rank": "-", "after_rank": "-"}]


def _format_trace_option(trace_id: str, traces: list[Any]) -> str:
    for item in traces:
        if item.trace_id == trace_id:
            return f"{item.started_at} · {item.query_text[:36]} · {item.total_elapsed_ms:.1f} ms"
    return trace_id
