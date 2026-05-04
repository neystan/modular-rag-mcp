"""自定义轻量 Evaluator。"""

from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """计算 hit_rate 与 mrr 的最小评估器。"""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        golden_set = set(golden_ids)
        if not retrieved_ids or not golden_set:
            return {"hit_rate": 0.0, "mrr": 0.0}

        first_hit_rank = next(
            (index for index, doc_id in enumerate(retrieved_ids, start=1) if doc_id in golden_set),
            None,
        )
        if first_hit_rank is None:
            return {"hit_rate": 0.0, "mrr": 0.0}

        return {"hit_rate": 1.0, "mrr": 1.0 / float(first_hit_rank)}
