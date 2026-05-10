"""组合 Evaluator 实现。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluatorError(RuntimeError):
    """Composite Evaluator 可读错误。"""


class CompositeEvaluator(BaseEvaluator):
    """并行执行多个 Evaluator 并汇总 metrics。"""

    def __init__(self, evaluators: list[BaseEvaluator]) -> None:
        super().__init__({})
        if not isinstance(evaluators, list) or not evaluators:
            raise CompositeEvaluatorError("composite evaluator config error: evaluators must be non-empty list")
        for evaluator in evaluators:
            if not isinstance(evaluator, BaseEvaluator):
                raise CompositeEvaluatorError("composite evaluator config error: evaluators must inherit BaseEvaluator")
        self.evaluators = list(evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        if len(self.evaluators) == 1:
            return self._normalize_metrics(self.evaluators[0].evaluate(query, retrieved_ids, golden_ids, trace=trace))

        merged: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=len(self.evaluators)) as executor:
            futures = [
                executor.submit(evaluator.evaluate, query, retrieved_ids, golden_ids, trace=trace)
                for evaluator in self.evaluators
            ]
            for future in as_completed(futures):
                merged.update(self._normalize_metrics(future.result()))
        return merged

    @staticmethod
    def _normalize_metrics(metrics: dict[str, float]) -> dict[str, float]:
        if not isinstance(metrics, dict):
            raise CompositeEvaluatorError("composite evaluator response error: metrics must be mapping")
        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise CompositeEvaluatorError(
                    f"composite evaluator response error: metric {key} must be numeric"
                ) from exc
        return normalized
