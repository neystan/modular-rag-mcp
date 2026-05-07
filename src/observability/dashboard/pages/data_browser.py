"""数据浏览器页面。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.data_service import DataService


def collect_browser_data(
    data_service: DataService | None = None,
    *,
    collection: str | None = None,
    selected_doc_id: str | None = None,
) -> dict[str, Any]:
    service = data_service or DataService()
    documents = service.list_documents(collection)
    selected_id = selected_doc_id or (documents[0].source_path if documents else None)
    detail = service.get_document_detail(selected_id) if selected_id else None
    return {
        "collections": service.list_collections(),
        "documents": documents,
        "selected_doc_id": selected_id,
        "detail": detail,
    }


def render(data_service: DataService | None = None) -> None:
    service = data_service or DataService()
    collections = service.list_collections()

    st.title("数据浏览器")
    st.caption("浏览已摄入文档、Chunk 详情和关联图片。")

    collection_options = ["全部集合", *collections]
    selected_collection = st.selectbox("集合筛选", collection_options, index=0)
    collection_filter = None if selected_collection == "全部集合" else selected_collection

    documents = service.list_documents(collection_filter)
    if not documents:
        st.info("当前没有已摄入文档。先执行 ingest，再回到这里浏览数据。")
        return

    selected_doc_id = st.selectbox(
        "选择文档",
        options=[item.source_path for item in documents],
        format_func=lambda source: _format_document_label(source, documents),
    )
    detail = service.get_document_detail(selected_doc_id)

    st.subheader("文档列表")
    table_rows = [
        {
            "source_path": item.source_path,
            "collection": item.collection,
            "chunk_count": item.chunk_count,
            "image_count": item.image_count,
            "updated_at": item.updated_at,
        }
        for item in documents
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.subheader("文档详情")
    summary_cols = st.columns(4)
    summary_cols[0].metric("集合", detail.collection)
    summary_cols[1].metric("Chunk 数", detail.chunk_count)
    summary_cols[2].metric("图片数", detail.image_count)
    summary_cols[3].metric("更新时间", detail.updated_at)

    st.markdown(f"**source_path**: `{detail.source_path}`")
    st.markdown(f"**file_hash**: `{detail.file_hash}`")

    st.subheader("Chunk 详情")
    for index, chunk in enumerate(detail.chunks, start=1):
        title = f"Chunk {index}: {chunk['id']}"
        with st.expander(title, expanded=index == 1):
            st.code(chunk["text"], language="markdown")
            st.json(chunk["metadata"])

    st.subheader("关联图片")
    if not detail.images:
        st.info("该文档没有关联图片。")
    else:
        image_cols = st.columns(2)
        for index, image in enumerate(detail.images):
            with image_cols[index % 2]:
                st.caption(f"{image['image_id']} · page {image.get('page_num')}")
                st.image(str(image["file_path"]), use_container_width=True)


def _format_document_label(source_path: str, documents: list[Any]) -> str:
    for item in documents:
        if item.source_path == source_path:
            return f"{item.source_path} · {item.collection} · chunks={item.chunk_count}"
    return source_path
