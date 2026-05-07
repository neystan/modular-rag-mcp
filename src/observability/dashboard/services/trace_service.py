"""Dashboard Trace 读取服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observability.logger import DEFAULT_TRACE_LOG_PATH


TRACKED_INGESTION_STAGES = ("load", "split", "transform", "embed", "upsert")
TRACKED_QUERY_STAGES = ("query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank")


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: str
    elapsed_ms: float
    duration_ms: float
    payload: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = ""


@dataclass(frozen=True, slots=True)
class TraceSummary:
    trace_id: str
    trace_type: str
    started_at: str
    finished_at: str
    total_elapsed_ms: float
    stage_count: int
    source_name: str = ""


@dataclass(frozen=True, slots=True)
class TraceDetail:
    trace_id: str
    trace_type: str
    started_at: str
    finished_at: str
    total_elapsed_ms: float
    source_name: str = ""
    stages: list[StageTiming] = field(default_factory=list)
    raw_stages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryTraceSummary:
    trace_id: str
    trace_type: str
    started_at: str
    finished_at: str
    total_elapsed_ms: float
    query_text: str
    result_count: int


@dataclass(frozen=True, slots=True)
class QueryTraceDetail:
    trace_id: str
    trace_type: str
    started_at: str
    finished_at: str
    total_elapsed_ms: float
    query_text: str
    keywords: list[str] = field(default_factory=list)
    collection: str = ""
    rerank_enabled: bool = True
    rerank_applied: bool = False
    final_ids: list[str] = field(default_factory=list)
    dense_ids: list[str] = field(default_factory=list)
    sparse_ids: list[str] = field(default_factory=list)
    fusion_ids: list[str] = field(default_factory=list)
    rerank_input_ids: list[str] = field(default_factory=list)
    rerank_result_ids: list[str] = field(default_factory=list)
    stages: list[StageTiming] = field(default_factory=list)
    raw_stages: list[dict[str, Any]] = field(default_factory=list)


class TraceService:
    """解析 JSONL trace 文件，供 Dashboard 查询。"""

    def __init__(self, log_path: str | Path = DEFAULT_TRACE_LOG_PATH) -> None:
        self.log_path = Path(log_path)

    def list_ingestion_traces(self) -> list[TraceSummary]:
        traces = [trace for trace in self._load_traces() if str(trace.get("trace_type", "")) == "ingestion"]
        return [
            TraceSummary(
                trace_id=str(trace.get("trace_id", "")),
                trace_type="ingestion",
                started_at=str(trace.get("started_at", "")),
                finished_at=str(trace.get("finished_at", "")),
                total_elapsed_ms=_safe_float(trace.get("total_elapsed_ms")),
                stage_count=len(_as_stage_list(trace.get("stages"))),
                source_name=_extract_source_name(_as_stage_list(trace.get("stages"))),
            )
            for trace in traces
            if str(trace.get("trace_id", "")).strip()
        ]

    def get_ingestion_trace(self, trace_id: str) -> TraceDetail:
        normalized_trace_id = _require_non_empty_str(trace_id, "trace_id")
        for trace in self._load_traces():
            if str(trace.get("trace_type", "")) != "ingestion":
                continue
            if str(trace.get("trace_id", "")).strip() != normalized_trace_id:
                continue
            raw_stages = _as_stage_list(trace.get("stages"))
            return TraceDetail(
                trace_id=normalized_trace_id,
                trace_type="ingestion",
                started_at=str(trace.get("started_at", "")),
                finished_at=str(trace.get("finished_at", "")),
                total_elapsed_ms=_safe_float(trace.get("total_elapsed_ms")),
                source_name=_extract_source_name(raw_stages),
                stages=_extract_ingestion_stages(raw_stages),
                raw_stages=raw_stages,
            )
        raise ValueError(f"trace not found: {normalized_trace_id}")

    def list_query_traces(self, query_keyword: str | None = None) -> list[QueryTraceSummary]:
        normalized_keyword = str(query_keyword or "").strip().lower()
        traces = [trace for trace in self._load_traces() if str(trace.get("trace_type", "")) == "query"]
        summaries: list[QueryTraceSummary] = []
        for trace in traces:
            raw_stages = _as_stage_list(trace.get("stages"))
            query_text = _extract_query_text(raw_stages)
            if normalized_keyword and normalized_keyword not in query_text.lower():
                continue
            summaries.append(
                QueryTraceSummary(
                    trace_id=str(trace.get("trace_id", "")),
                    trace_type="query",
                    started_at=str(trace.get("started_at", "")),
                    finished_at=str(trace.get("finished_at", "")),
                    total_elapsed_ms=_safe_float(trace.get("total_elapsed_ms")),
                    query_text=query_text,
                    result_count=len(_extract_final_ids(raw_stages)),
                )
            )
        return [item for item in summaries if item.trace_id.strip()]

    def get_query_trace(self, trace_id: str) -> QueryTraceDetail:
        normalized_trace_id = _require_non_empty_str(trace_id, "trace_id")
        for trace in self._load_traces():
            if str(trace.get("trace_type", "")) != "query":
                continue
            if str(trace.get("trace_id", "")).strip() != normalized_trace_id:
                continue
            raw_stages = _as_stage_list(trace.get("stages"))
            execution = _extract_query_execution(raw_stages)
            return QueryTraceDetail(
                trace_id=normalized_trace_id,
                trace_type="query",
                started_at=str(trace.get("started_at", "")),
                finished_at=str(trace.get("finished_at", "")),
                total_elapsed_ms=_safe_float(trace.get("total_elapsed_ms")),
                query_text=_extract_query_text(raw_stages),
                keywords=_extract_query_keywords(raw_stages),
                collection=str(execution.get("collection", "") or ""),
                rerank_enabled=bool(execution.get("rerank_enabled", True)),
                rerank_applied=bool(execution.get("rerank_applied", False)),
                final_ids=_extract_final_ids(raw_stages),
                dense_ids=_extract_stage_chunk_ids(raw_stages, "dense_retrieval"),
                sparse_ids=_extract_stage_chunk_ids(raw_stages, "sparse_retrieval"),
                fusion_ids=_extract_stage_chunk_ids(raw_stages, "fusion"),
                rerank_input_ids=_extract_stage_detail_list(raw_stages, "rerank", "input_ids"),
                rerank_result_ids=_extract_stage_detail_list(raw_stages, "rerank", "result_ids"),
                stages=_extract_query_stages(raw_stages),
                raw_stages=raw_stages,
            )
        raise ValueError(f"trace not found: {normalized_trace_id}")

    def _load_traces(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        traces: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.log_path.read_text(encoding="utf-8").splitlines(), start=1):
            content = line.strip()
            if not content:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trace json at line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"invalid trace payload at line {line_number}: object required")
            traces.append(payload.get("trace") if isinstance(payload.get("trace"), dict) else payload)

        traces.sort(key=lambda item: str(item.get("started_at", "")), reverse=True)
        return traces


def _extract_ingestion_stages(raw_stages: list[dict[str, Any]]) -> list[StageTiming]:
    stage_points: list[tuple[str, float, dict[str, Any], str]] = []
    for stage_name in TRACKED_INGESTION_STAGES:
        matched = next(
            (
                item
                for item in raw_stages
                if str(item.get("stage", "")).strip() == stage_name
            ),
            None,
        )
        if matched is None:
            continue
        stage_points.append(
            (
                stage_name,
                _safe_float(matched.get("elapsed_ms")),
                dict(matched.get("payload", {})) if isinstance(matched.get("payload"), dict) else {},
                str(matched.get("recorded_at", "")),
            )
        )

    stage_timings: list[StageTiming] = []
    previous_elapsed = 0.0
    for stage_name, elapsed_ms, payload, recorded_at in stage_points:
        duration_ms = max(elapsed_ms - previous_elapsed, 0.0)
        previous_elapsed = elapsed_ms
        stage_timings.append(
            StageTiming(
                stage=stage_name,
                elapsed_ms=elapsed_ms,
                duration_ms=round(duration_ms, 3),
                payload=payload,
                recorded_at=recorded_at,
            )
        )
    return stage_timings


def _extract_query_stages(raw_stages: list[dict[str, Any]]) -> list[StageTiming]:
    stage_points: list[tuple[str, float, dict[str, Any], str]] = []
    for stage_name in TRACKED_QUERY_STAGES:
        matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == stage_name), None)
        if matched is None:
            continue
        stage_points.append(
            (
                stage_name,
                _safe_float(matched.get("elapsed_ms")),
                dict(matched.get("payload", {})) if isinstance(matched.get("payload"), dict) else {},
                str(matched.get("recorded_at", "")),
            )
        )
    return [StageTiming(stage=name, elapsed_ms=elapsed, duration_ms=0.0, payload=payload, recorded_at=recorded_at) for name, elapsed, payload, recorded_at in stage_points]


def _as_stage_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _safe_float(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _extract_source_name(raw_stages: list[dict[str, Any]]) -> str:
    for stage_name in ("dashboard.upload", "dashboard.result", "pipeline.skip", "load"):
        matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == stage_name), None)
        if matched is None:
            continue
        payload = matched.get("payload")
        if not isinstance(payload, dict):
            continue
        original_filename = str(payload.get("original_filename", "")).strip()
        if original_filename:
            return original_filename
        path_value = str(payload.get("path", "")).strip()
        if path_value:
            return Path(path_value).name
        details = payload.get("details")
        if isinstance(details, dict):
            document_id = str(details.get("document_id", "")).strip()
            if document_id:
                return document_id
    return ""


def _extract_query_text(raw_stages: list[dict[str, Any]]) -> str:
    matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == "query_processing"), None)
    if matched is None:
        matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == "query.execution"), None)
    if matched is None:
        return ""
    payload = matched.get("payload")
    if not isinstance(payload, dict):
        return ""
    details = payload.get("details")
    if isinstance(details, dict):
        query_text = str(details.get("query_text", "")).strip()
        if query_text:
            return query_text
    return str(payload.get("query_text", "")).strip()


def _extract_query_keywords(raw_stages: list[dict[str, Any]]) -> list[str]:
    matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == "query_processing"), None)
    if matched is None:
        return []
    payload = matched.get("payload")
    if not isinstance(payload, dict):
        return []
    details = payload.get("details")
    keywords = details.get("keywords", []) if isinstance(details, dict) else []
    if not isinstance(keywords, list):
        return []
    return [str(keyword) for keyword in keywords if str(keyword).strip()]


def _extract_query_execution(raw_stages: list[dict[str, Any]]) -> dict[str, Any]:
    matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == "query.execution"), None)
    if matched is None:
        return {}
    payload = matched.get("payload")
    return dict(payload) if isinstance(payload, dict) else {}


def _extract_final_ids(raw_stages: list[dict[str, Any]]) -> list[str]:
    execution = _extract_query_execution(raw_stages)
    final_ids = execution.get("final_ids", [])
    if isinstance(final_ids, list):
        return [str(item) for item in final_ids if str(item).strip()]
    rerank_ids = _extract_stage_detail_list(raw_stages, "rerank", "result_ids")
    if rerank_ids:
        return rerank_ids
    return _extract_stage_chunk_ids(raw_stages, "fusion")


def _extract_stage_chunk_ids(raw_stages: list[dict[str, Any]], stage_name: str) -> list[str]:
    return _extract_stage_detail_list(raw_stages, stage_name, "chunk_ids")


def _extract_stage_detail_list(raw_stages: list[dict[str, Any]], stage_name: str, key: str) -> list[str]:
    matched = next((item for item in raw_stages if str(item.get("stage", "")).strip() == stage_name), None)
    if matched is None:
        return []
    payload = matched.get("payload")
    if not isinstance(payload, dict):
        return []
    details = payload.get("details")
    source = details if isinstance(details, dict) else payload
    values = source.get(key, []) if isinstance(source, dict) else []
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item).strip()]


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()
