"""Ingestion 管理页。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Callable, Protocol

import streamlit as st

from core.settings import Settings, load_settings
from ingestion.pipeline import IngestionPipeline, IngestionPipelineResult
from observability.dashboard.services.data_service import DataService


ProgressCallback = Callable[[str, int, int], None]
SESSION_FEEDBACK_KEY = "ingestion_manager_feedback"
STAGE_LABELS = {
    "load": "文档加载",
    "split": "文本切分",
    "transform": "内容增强",
    "embed": "向量编码",
    "upsert": "索引写入",
}


class PipelineRunner(Protocol):
    def run(
        self,
        path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        trace: Any | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionPipelineResult:
        """执行单文件摄取。"""


def collect_ingestion_data(
    data_service: DataService | None = None,
    *,
    collection: str | None = None,
) -> dict[str, Any]:
    service = data_service or DataService()
    documents = service.list_documents(collection)
    stats = service.get_collection_stats(collection)
    return {
        "collections": service.list_collections(),
        "documents": documents,
        "stats": stats,
    }


def ingest_uploaded_file(
    file_name: str,
    file_bytes: bytes,
    collection: str,
    *,
    pipeline: PipelineRunner,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> IngestionPipelineResult:
    normalized_collection = _require_non_empty_str(collection, "collection")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name is required")
    if not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes:
        raise ValueError("file_bytes is required")

    suffix = Path(file_name.strip()).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(prefix="dashboard-upload-", suffix=suffix, delete=False) as temp_file:
        temp_file.write(bytes(file_bytes))
        temp_path = Path(temp_file.name)

    try:
        return pipeline.run(
            temp_path,
            collection=normalized_collection,
            force=force,
            on_progress=progress_callback,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def render(
    data_service: DataService | None = None,
    *,
    pipeline: PipelineRunner | None = None,
    settings_path: str | Path = "config/settings.yaml",
    settings_loader: Callable[[str | Path], Settings] = load_settings,
) -> None:
    service = data_service or DataService(settings_path=settings_path, settings_loader=settings_loader)
    page_data = collect_ingestion_data(service)
    default_collection = _resolve_default_collection(page_data["collections"], settings_path, settings_loader)

    st.title("Ingestion 管理")
    st.caption("上传 PDF 触发摄取，跟踪阶段进度，并管理已入库文档。")
    _render_feedback()

    st.subheader("上传并摄取")
    with st.form("ingestion-upload-form", clear_on_submit=False):
        upload_cols = st.columns([2, 1])
        collection_options = ["使用默认集合", *page_data["collections"]]
        selected_option = upload_cols[0].selectbox("选择已有集合", collection_options, index=0)
        custom_collection = upload_cols[1].text_input("目标集合名称", value=default_collection)
        uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"])
        force = st.checkbox("强制重建（忽略去重）", value=False)
        submitted = st.form_submit_button("开始摄取", type="primary", use_container_width=True)

    if submitted:
        target_collection = custom_collection if selected_option == "使用默认集合" else selected_option
        active_pipeline = pipeline or IngestionPipeline(settings_loader(settings_path))
        _handle_ingestion_submit(uploaded_file, target_collection, force, active_pipeline)

    st.subheader("已摄入文档")
    filter_options = ["全部集合", *page_data["collections"]]
    selected_filter = st.selectbox("集合筛选", filter_options, index=0)
    collection_filter = None if selected_filter == "全部集合" else selected_filter
    filtered_data = collect_ingestion_data(service, collection=collection_filter)
    stats = filtered_data["stats"]

    stats_cols = st.columns(3)
    stats_cols[0].metric("文档数", stats.document_count)
    stats_cols[1].metric("Chunk 数", stats.chunk_count)
    stats_cols[2].metric("图片数", stats.image_count)

    documents = filtered_data["documents"]
    if not documents:
        st.info("当前没有可管理的文档。先上传 PDF 完成摄取。")
        return

    for item in documents:
        row_cols = st.columns([5, 2, 2, 2, 1])
        row_cols[0].markdown(f"**{item.source_path}**")
        row_cols[1].caption(item.collection)
        row_cols[2].caption(f"chunks: {item.chunk_count}")
        row_cols[3].caption(f"images: {item.image_count}")
        if row_cols[4].button("删除", key=f"delete-{item.collection}-{item.file_hash}", use_container_width=True):
            _delete_document(service, item.source_path, item.collection)
            return


def _handle_ingestion_submit(
    uploaded_file: Any,
    collection: str,
    force: bool,
    pipeline: PipelineRunner,
) -> None:
    normalized_collection = _require_non_empty_str(collection, "collection")
    if uploaded_file is None:
        st.warning("请先选择一个 PDF 文件。")
        return

    progress_bar = st.progress(0.0, text="准备开始摄取")
    progress_text = st.empty()

    def _on_progress(stage_name: str, current: int, total: int) -> None:
        ratio = min(current / total, 1.0) if total else 0.0
        stage_label = STAGE_LABELS.get(stage_name, stage_name)
        progress_bar.progress(ratio, text=f"{stage_label} ({current}/{total})")
        progress_text.caption(f"当前阶段：{stage_label}")

    try:
        result = ingest_uploaded_file(
            uploaded_file.name,
            uploaded_file.getvalue(),
            normalized_collection,
            pipeline=pipeline,
            force=force,
            progress_callback=_on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        progress_text.empty()
        st.error(f"摄取失败：{exc}")
        return

    if result.skipped:
        progress_bar.progress(1.0, text="文件已存在，已跳过")
        _set_feedback("warning", f"文件已存在，跳过摄取：{uploaded_file.name}")
    else:
        progress_bar.progress(1.0, text="摄取完成")
        _set_feedback(
            "success",
            f"摄取完成：{uploaded_file.name}，集合 `{result.collection}`，Chunk {len(result.chunks)}，图片 {result.image_count}",
        )
    st.rerun()


def _delete_document(data_service: DataService, source_path: str, collection: str) -> None:
    try:
        result = data_service.delete_document(source_path, collection)
    except Exception as exc:  # noqa: BLE001
        st.error(f"删除失败：{exc}")
        return

    _set_feedback(
        "success",
        f"已删除 `{result.source_path}`：chunks {result.deleted_chunks}，images {result.deleted_images}",
    )
    st.rerun()


def _resolve_default_collection(
    collections: list[str],
    settings_path: str | Path,
    settings_loader: Callable[[str | Path], Settings],
) -> str:
    if collections:
        return collections[0]
    try:
        settings = settings_loader(settings_path)
    except Exception:  # noqa: BLE001
        return "default"
    return str(settings.vector_store.get("collection", "default")).strip() or "default"


def _render_feedback() -> None:
    feedback = st.session_state.pop(SESSION_FEEDBACK_KEY, None)
    if not isinstance(feedback, dict):
        return

    level = str(feedback.get("level", "info"))
    message = str(feedback.get("message", "")).strip()
    if not message:
        return

    renderer = getattr(st, level, st.info)
    renderer(message)


def _set_feedback(level: str, message: str) -> None:
    st.session_state[SESSION_FEEDBACK_KEY] = {
        "level": level,
        "message": message,
    }


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()
