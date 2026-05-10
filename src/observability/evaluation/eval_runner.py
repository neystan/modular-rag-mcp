"""评估执行器。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.query_engine.hybrid_search import HybridSearch
from core.settings import Settings
from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.llm.base_llm import BaseLLM


class EvalRunnerError(RuntimeError):
    """EvalRunner 可读错误。"""


@dataclass(frozen=True, slots=True)
class EvalCase:
    """黄金测试集中的单条用例。"""

    query: str
    reference: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """单条评估结果。"""

    query: str
    reference: str
    answer: str
    contexts: list[str]
    retrieved_ids: list[str]
    expected_chunk_ids: list[str]
    expected_sources: list[str]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class EvalReport:
    """评估汇总报告。"""

    total_cases: int
    hit_rate: float
    mrr: float
    metrics: dict[str, float]
    results: list[EvalCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvalRunner:
    """读取黄金测试集，执行检索并产出评估报告。"""

    def __init__(
        self,
        settings: Settings,
        hybrid_search: HybridSearch,
        evaluator: BaseEvaluator,
        llm: BaseLLM | None = None,
    ) -> None:
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.llm = llm

    def run(self, test_set_path: str | Path) -> EvalReport:
        cases = self._load_test_cases(test_set_path)
        top_k = self._resolve_top_k()

        results: list[EvalCaseResult] = []
        metric_totals: dict[str, float] = {}
        for case in cases:
            retrieval_results = [] if case.contexts else self.hybrid_search.search(case.query, top_k=top_k)
            retrieved_ids = [item.chunk_id for item in retrieval_results]
            golden_ids = case.expected_chunk_ids or self._matched_source_ids(retrieval_results, case.expected_sources)
            contexts = case.contexts or [item.text for item in retrieval_results if str(item.text).strip()]
            answer = case.answer or self._build_answer(case.query, contexts)
            trace = {"answer": answer, "contexts": contexts, "reference": case.reference}
            metrics = self.evaluator.evaluate(case.query, retrieved_ids, golden_ids, trace=trace)
            normalized_metrics = self._normalize_metrics(metrics)
            results.append(
                EvalCaseResult(
                    query=case.query,
                    reference=case.reference,
                    answer=answer,
                    contexts=contexts,
                    retrieved_ids=retrieved_ids,
                    expected_chunk_ids=list(case.expected_chunk_ids),
                    expected_sources=list(case.expected_sources),
                    metrics=normalized_metrics,
                )
            )
            for key, value in normalized_metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value

        averaged = {key: value / len(results) for key, value in metric_totals.items()}
        return EvalReport(
            total_cases=len(results),
            hit_rate=averaged.get("hit_rate", 0.0),
            mrr=averaged.get("mrr", 0.0),
            metrics=averaged,
            results=results,
        )

    @staticmethod
    def _load_test_cases(test_set_path: str | Path) -> list[EvalCase]:
        path = Path(test_set_path)
        if not path.exists():
            raise EvalRunnerError(f"golden test set not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvalRunnerError(f"golden test set JSON parse failed: {path}") from exc

        raw_cases = payload.get("test_cases") if isinstance(payload, dict) else None
        if not isinstance(raw_cases, list) or not raw_cases:
            raise EvalRunnerError("golden test set error: test_cases must be a non-empty list")

        cases: list[EvalCase] = []
        for index, item in enumerate(raw_cases):
            if not isinstance(item, dict):
                raise EvalRunnerError(f"golden test set error: test_cases[{index}] must be object")
            query = item.get("query")
            if query is None:
                query = item.get("question")
            if not isinstance(query, str) or not query.strip():
                raise EvalRunnerError(f"golden test set error: test_cases[{index}].question is required")
            reference = item.get("reference")
            if reference is None:
                reference = item.get("ground_truth") or item.get("ground_truths")
            if not isinstance(reference, str) or not reference.strip():
                raise EvalRunnerError(f"golden test set error: test_cases[{index}].reference is required")
            answer = item.get("answer", "")
            if answer is None:
                answer = ""
            if not isinstance(answer, str):
                raise EvalRunnerError(f"golden test set error: test_cases[{index}].answer must be string")
            contexts = _normalize_string_list(
                item.get("contexts", []),
                f"test_cases[{index}].contexts",
            )
            expected_chunk_ids = _normalize_string_list(
                item.get("expected_chunk_ids", []),
                f"test_cases[{index}].expected_chunk_ids",
            )
            expected_sources = _normalize_string_list(
                item.get("expected_sources", []),
                f"test_cases[{index}].expected_sources",
            )
            cases.append(
                EvalCase(
                    query=query.strip(),
                    reference=reference.strip(),
                    answer=answer.strip(),
                    contexts=contexts,
                    expected_chunk_ids=expected_chunk_ids,
                    expected_sources=expected_sources,
                )
            )
        return cases

    def _resolve_top_k(self) -> int:
        top_k = self.settings.retrieval.get("top_k")
        if not isinstance(top_k, int) or top_k <= 0:
            raise EvalRunnerError("eval runner config error: retrieval.top_k must be positive int")
        return top_k

    def _build_answer(self, query: str, contexts: list[str]) -> str:
        if not contexts:
            return ""
        if self.llm is None:
            return "\n\n".join(contexts)

        context_text = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
        return self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是 RAG 评估生成器。只能基于给定上下文回答，避免编造。",
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n\n上下文：\n{context_text}\n\n请给出简洁答案。",
                },
            ]
        )

    @staticmethod
    def _matched_source_ids(results: list[RetrievalResult], expected_sources: list[str]) -> list[str]:
        if not expected_sources:
            return []
        matched_ids: list[str] = []
        for result in results:
            source_path = str(result.metadata.get("source_path") or result.metadata.get("source") or "")
            source_name = Path(source_path).name
            if any(source_path == expected or source_name == expected for expected in expected_sources):
                matched_ids.append(result.chunk_id)
        return matched_ids

    @staticmethod
    def _normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
        if not isinstance(metrics, dict):
            raise EvalRunnerError("eval runner response error: metrics must be mapping")
        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise EvalRunnerError(f"eval runner response error: metric {key} must be numeric") from exc
        return normalized


def _normalize_string_list(value: Any, field_path: str) -> list[str]:
    if not isinstance(value, list):
        raise EvalRunnerError(f"golden test set error: {field_path} must be list")
    return [str(item).strip() for item in value if str(item).strip()]
