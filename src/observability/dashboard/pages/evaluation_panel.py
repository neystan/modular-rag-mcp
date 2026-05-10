"""评估面板页。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from core.query_engine.hybrid_search import HybridSearch
from core.settings import Settings, load_settings
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.llm.llm_factory import LLMFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner


DEFAULT_TEST_SET_PATH = "tests/fixtures/golden_test_set.json"
HISTORY_KEY = "evaluation_panel_history"
CONFIGURED_BACKEND = "配置默认"
BACKEND_OPTIONS = (CONFIGURED_BACKEND, "custom", "ragas")


def collect_evaluation_panel_data(
    *,
    settings_path: str | Path = "config/settings.yaml",
    test_set_path: str | Path = DEFAULT_TEST_SET_PATH,
    settings_loader: Callable[[str | Path], Settings] = load_settings,
) -> dict[str, Any]:
    settings = settings_loader(settings_path)
    return {
        "evaluation": settings.evaluation,
        "retrieval": settings.retrieval,
        "test_set": load_test_set_summary(test_set_path),
        "backend_options": list(BACKEND_OPTIONS),
    }


def load_test_set_summary(test_set_path: str | Path) -> dict[str, Any]:
    path = Path(test_set_path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "case_count": 0,
        "queries": [],
        "error": None,
    }
    if not path.exists():
        summary["error"] = f"文件不存在: {path}"
        return summary

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cases = payload.get("test_cases") if isinstance(payload, dict) else None
        if not isinstance(raw_cases, list):
            summary["error"] = "test_cases 必须是 list"
            return summary
        queries = [
            str(item.get("question") or item.get("query") or "").strip()
            for item in raw_cases
            if isinstance(item, dict) and str(item.get("question") or item.get("query") or "").strip()
        ]
        summary["case_count"] = len(raw_cases)
        summary["queries"] = queries
        return summary
    except Exception as exc:  # noqa: BLE001
        summary["error"] = str(exc)
        return summary


def run_dashboard_evaluation(
    *,
    settings_path: str | Path = "config/settings.yaml",
    test_set_path: str | Path = DEFAULT_TEST_SET_PATH,
    backend: str = CONFIGURED_BACKEND,
    settings_loader: Callable[[str | Path], Settings] = load_settings,
    runner_factory: Callable[[Settings], EvalRunner] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> EvalReport:
    settings = settings_loader(settings_path)
    active_settings = _settings_with_backend(settings, backend)
    runner = runner_factory(active_settings) if runner_factory else _build_default_runner(active_settings)
    return runner.run_with_progress(test_set_path, progress_callback=progress_callback)


def render(
    *,
    settings_path: str | Path = "config/settings.yaml",
    settings_loader: Callable[[str | Path], Settings] = load_settings,
) -> None:
    st.title("评估面板")
    st.caption("运行 golden test set，查看检索与生成质量指标。")

    default_data = collect_evaluation_panel_data(settings_path=settings_path, settings_loader=settings_loader)
    _render_config_summary(default_data)

    st.subheader("运行评估")
    with st.form("evaluation-run-form", clear_on_submit=False):
        form_cols = st.columns([2, 1])
        test_set_path = form_cols[0].text_input("Golden Test Set", value=DEFAULT_TEST_SET_PATH)
        backend = form_cols[1].selectbox("评估后端", options=list(BACKEND_OPTIONS), index=0)
        submitted = st.form_submit_button("运行评估", type="primary", width="stretch")

    preview = load_test_set_summary(test_set_path)
    _render_test_set_preview(preview)

    if submitted:
        progress = st.progress(0.0, text="准备开始评估")
        status = st.empty()

        def _on_progress(current: int, total: int, message: str) -> None:
            ratio = (current / total) if total else 0.0
            progress.progress(min(max(ratio, 0.0), 1.0), text=message)
            status.caption(message)

        try:
            report = run_dashboard_evaluation(
                settings_path=settings_path,
                test_set_path=test_set_path,
                backend=str(backend),
                settings_loader=settings_loader,
                progress_callback=_on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            progress.empty()
            status.empty()
            st.error(f"评估失败：{exc}")
        else:
            progress.progress(1.0, text="评估完成")
            status.caption("评估完成")
            _append_history(report, str(backend), test_set_path)
            _render_report(report)

    history = st.session_state.get(HISTORY_KEY, [])
    if history:
        st.subheader("历史对比")
        st.dataframe(history, width="stretch", hide_index=True)


def _build_default_runner(settings: Settings) -> EvalRunner:
    return EvalRunner(
        settings=settings,
        hybrid_search=HybridSearch(settings),
        evaluator=EvaluatorFactory.create(settings),
        llm=LLMFactory.create(settings),
    )


def _settings_with_backend(settings: Settings, backend: str) -> Settings:
    normalized = str(backend).strip()
    if not normalized or normalized == CONFIGURED_BACKEND:
        return settings

    evaluation = {key: value for key, value in settings.evaluation.items() if key != "backends"}
    evaluation["provider"] = normalized
    return Settings(
        app=settings.app,
        llm=settings.llm,
        vision_llm=settings.vision_llm,
        embedding=settings.embedding,
        splitter=settings.splitter,
        vector_store=settings.vector_store,
        retrieval=settings.retrieval,
        rerank=settings.rerank,
        evaluation=evaluation,
        observability=settings.observability,
        ingestion=settings.ingestion,
    )


def _render_config_summary(data: dict[str, Any]) -> None:
    top_cols = st.columns(4)
    evaluation = data["evaluation"]
    retrieval = data["retrieval"]
    test_set = data["test_set"]
    configured_backend = evaluation.get("backends") or evaluation.get("provider", "custom")
    top_cols[0].metric("配置后端", str(configured_backend))
    top_cols[1].metric("retrieval.top_k", str(retrieval.get("top_k", "<missing>")))
    top_cols[2].metric("测试用例", int(test_set["case_count"]))
    top_cols[3].metric("Golden Set", "可用" if test_set["exists"] and not test_set["error"] else "需检查")


def _render_test_set_preview(summary: dict[str, Any]) -> None:
    if summary["error"]:
        st.warning(f"Golden test set 无法读取：{summary['error']}")
        return
    with st.expander("Golden Test Set 预览", expanded=False):
        st.write(f"路径：`{summary['path']}`")
        rows = [{"index": index, "query": query} for index, query in enumerate(summary["queries"], start=1)]
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_report(report: EvalReport) -> None:
    st.subheader("评估结果")
    primary_metrics = [
        ("测试用例", float(report.total_cases)),
        ("context_precision", report.metrics.get("context_precision", report.hit_rate)),
        ("context_recall", report.metrics.get("context_recall", 0.0)),
        ("faithfulness", report.metrics.get("faithfulness", report.mrr)),
        ("answer_relevancy", report.metrics.get("answer_relevancy", 0.0)),
    ]
    metric_cols = st.columns(len(primary_metrics))
    for column, (label, value) in zip(metric_cols, primary_metrics, strict=False):
        column.metric(label, f"{value:.3f}" if label != "测试用例" else str(int(value)))

    st.markdown("**平均指标**")
    st.dataframe(
        [{"metric": key, "value": value} for key, value in sorted(report.metrics.items())],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Query 明细**")
    st.dataframe(_case_rows(report), width="stretch", hide_index=True)


def _case_rows(report: EvalReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.results:
        row = {
            "query": item.query,
            "reference": item.reference,
            "answer": item.answer,
            "retrieved_ids": ", ".join(item.retrieved_ids),
            "contexts": "\n---\n".join(item.contexts),
            "expected_chunk_ids": ", ".join(item.expected_chunk_ids),
            "expected_sources": ", ".join(item.expected_sources),
        }
        row.update({key: round(value, 4) for key, value in item.metrics.items()})
        rows.append(row)
    return rows


def _append_history(report: EvalReport, backend: str, test_set_path: str | Path) -> None:
    history = list(st.session_state.get(HISTORY_KEY, []))
    history.append(
        {
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend,
            "test_set": str(test_set_path),
            "total_cases": report.total_cases,
            "context_precision": round(report.metrics.get("context_precision", report.hit_rate), 4),
            "context_recall": round(report.metrics.get("context_recall", 0.0), 4),
            "faithfulness": round(report.metrics.get("faithfulness", report.mrr), 4),
            "answer_relevancy": round(report.metrics.get("answer_relevancy", 0.0), 4),
        }
    )
    st.session_state[HISTORY_KEY] = history[-20:]
