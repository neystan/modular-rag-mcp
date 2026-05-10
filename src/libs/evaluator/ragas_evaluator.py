"""Ragas Evaluator 实现。"""

from __future__ import annotations

import importlib
import os
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class RagasEvaluatorError(RuntimeError):
    """Ragas Evaluator 可读错误。"""


class RagasEvaluator(BaseEvaluator):
    """封装 Ragas 的生成质量评估器。"""

    default_metrics = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
    ) -> dict[str, float]:
        normalized_query = self._normalize_query(query)
        contexts = self._resolve_contexts(retrieved_ids, trace)
        answer = self._resolve_answer(trace)
        reference = self._resolve_reference(golden_ids, trace)

        runner = self.config.get("runner")
        if callable(runner):
            result = runner(normalized_query, answer, contexts, reference, self._metric_names())
        else:
            result = self._run_ragas(normalized_query, answer, contexts, reference)

        return self._normalize_metrics(result)

    def _run_ragas(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        reference: str,
    ) -> Any:
        ragas = self._import_required("ragas")
        metrics_module = self._import_required("ragas.metrics")
        datasets = self._import_required("datasets")

        metrics = [getattr(metrics_module, name) for name in self._metric_names()]
        dataset = datasets.Dataset.from_list(
            [
                {
                    "question": query,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": reference,
                }
            ]
        )
        evaluate_kwargs: dict[str, Any] = {"metrics": metrics}
        ragas_llm = self._build_ragas_llm()
        if ragas_llm is not None:
            evaluate_kwargs["llm"] = ragas_llm
        ragas_embeddings = self._build_ragas_embeddings()
        if ragas_embeddings is not None:
            evaluate_kwargs["embeddings"] = ragas_embeddings
        return ragas.evaluate(dataset, **evaluate_kwargs)

    @staticmethod
    def _import_required(module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                "RagasEvaluator requires optional dependencies: install ragas and datasets to use "
                "evaluation.provider=ragas. Install with: uv add ragas datasets"
            ) from exc

    def _build_ragas_llm(self) -> Any | None:
        llm_config = self.config.get("llm_config")
        if not isinstance(llm_config, dict):
            return None

        provider = str(llm_config.get("provider", "")).strip().lower()
        model = str(llm_config.get("model", "")).strip()
        if not provider or not model:
            return None

        if provider not in {"openai", "qwen", "deepseek"}:
            raise RagasEvaluatorError(
                f"ragas evaluator config error: unsupported llm provider for ragas: {provider}"
            )

        ragas_llms = self._import_required("ragas.llms")
        openai_module = self._import_required("openai")
        client = openai_module.OpenAI(
            api_key=self._resolve_api_key(llm_config),
            base_url=str(llm_config.get("base_url", "")).strip() or None,
        )
        return ragas_llms.llm_factory(
            model=model,
            provider="openai",
            client=client,
        )

    def _build_ragas_embeddings(self) -> Any | None:
        embedding_config = self.config.get("embedding_config")
        if not isinstance(embedding_config, dict):
            return None

        provider = str(embedding_config.get("provider", "")).strip().lower()
        model = str(embedding_config.get("model", "")).strip()
        if not provider or not model:
            return None

        if provider not in {"openai", "qwen", "deepseek"}:
            raise RagasEvaluatorError(
                f"ragas evaluator config error: unsupported embedding provider for ragas: {provider}"
            )

        ragas_embeddings = self._import_required("ragas.embeddings")
        openai_module = self._import_required("openai")
        client = openai_module.OpenAI(
            api_key=self._resolve_api_key(embedding_config),
            base_url=str(embedding_config.get("base_url", "")).strip() or None,
        )
        return ragas_embeddings.OpenAIEmbeddings(client=client, model=model)

    @staticmethod
    def _resolve_api_key(config: dict[str, Any]) -> str | None:
        api_key = str(config.get("api_key", "")).strip()
        if api_key:
            return api_key
        provider = str(config.get("provider", "")).strip().lower()
        if provider == "qwen":
            return os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("QWEN_API_KEY", "").strip() or None
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY", "").strip() or None
        return os.getenv("OPENAI_API_KEY", "").strip() or None

    def _metric_names(self) -> list[str]:
        configured = self.config.get("metrics", self.default_metrics)
        if not isinstance(configured, (list, tuple)) or not configured:
            raise RagasEvaluatorError("ragas evaluator config error: metrics must be a non-empty list")
        metric_names = [str(item).strip() for item in configured if str(item).strip()]
        if not metric_names:
            raise RagasEvaluatorError("ragas evaluator config error: metrics must be a non-empty list")
        return metric_names

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise RagasEvaluatorError("ragas evaluator input error: query is required")
        return query.strip()

    def _resolve_contexts(self, retrieved_ids: list[str], trace: Any | None) -> list[str]:
        configured = self.config.get("contexts")
        if configured is None:
            configured = self.config.get("context_texts")
        if isinstance(trace, dict):
            trace_contexts = trace.get("contexts") or trace.get("context_texts")
            if isinstance(trace_contexts, list):
                normalized = [str(item).strip() for item in trace_contexts if str(item).strip()]
                if normalized:
                    return normalized
        contexts = configured if isinstance(configured, list) else retrieved_ids
        if not isinstance(contexts, list) or not contexts:
            raise RagasEvaluatorError("ragas evaluator input error: contexts are required")
        normalized = [str(item).strip() for item in contexts if str(item).strip()]
        if not normalized:
            raise RagasEvaluatorError("ragas evaluator input error: contexts are required")
        return normalized

    def _resolve_answer(self, trace: Any | None) -> str:
        configured = self.config.get("answer")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        if isinstance(trace, dict):
            answer = trace.get("answer") or trace.get("generated_answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        raise RagasEvaluatorError("ragas evaluator input error: answer is required")

    def _resolve_reference(self, golden_ids: list[str], trace: Any | None) -> str:
        configured = self.config.get("reference")
        if configured is None:
            configured = self.config.get("ground_truth")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        if isinstance(trace, dict):
            reference = trace.get("reference") or trace.get("ground_truth")
            if isinstance(reference, str) and reference.strip():
                return reference.strip()
        if isinstance(golden_ids, list) and golden_ids:
            joined = "\n".join(str(item).strip() for item in golden_ids if str(item).strip())
            if joined:
                return joined
        raise RagasEvaluatorError("ragas evaluator input error: reference is required")

    @staticmethod
    def _normalize_metrics(result: Any) -> dict[str, float]:
        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            if hasattr(frame, "to_dict"):
                records = frame.to_dict(orient="records")
                if records:
                    result = records[0]
        elif hasattr(result, "scores"):
            result = result.scores

        if not isinstance(result, dict):
            try:
                result = dict(result)
            except (TypeError, ValueError) as exc:
                raise RagasEvaluatorError("ragas evaluator response error: metrics must be mapping") from exc

        metrics: dict[str, float] = {}
        for key, value in result.items():
            if key in {"question", "answer", "contexts", "ground_truth", "reference"}:
                continue
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise RagasEvaluatorError(
                    f"ragas evaluator response error: metric {key} must be numeric"
                ) from exc
        if not metrics:
            raise RagasEvaluatorError("ragas evaluator response error: metrics are empty")
        return metrics
