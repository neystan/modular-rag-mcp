"""Ragas Evaluator 实现。"""

from __future__ import annotations

import copy
import importlib
import asyncio
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.embedding.base_embedding import BaseEmbedding
from libs.llm.base_llm import BaseLLM


def _create_ragas_llm_adapter(ragas_base_cls: type[Any], llm: BaseLLM, max_tokens: int | None = None) -> Any:
    class RagasLLMAdapter(ragas_base_cls):
        def __init__(self, wrapped_llm: BaseLLM, wrapped_max_tokens: int | None = None) -> None:
            super().__init__(multiple_completion_supported=False)
            self._llm = wrapped_llm
            self._max_tokens = wrapped_max_tokens

        def generate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float = 0.01,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            del stop, callbacks
            generation_text = self._invoke_llm(prompt, temperature=temperature)
            outputs = importlib.import_module("langchain_core.outputs")
            generations = [[outputs.Generation(text=generation_text) for _ in range(max(1, n))]]
            return outputs.LLMResult(generations=generations)

        async def agenerate_text(
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: list[str] | None = None,
            callbacks: Any = None,
        ) -> Any:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n,
                0.01 if temperature is None else temperature,
                stop,
                callbacks,
            )

        def is_finished(self, response: Any) -> bool:
            del response
            return True

        def _invoke_llm(self, prompt: Any, *, temperature: float) -> str:
            prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            llm_config = dict(self._llm.config)
            llm_config["temperature"] = temperature
            if self._max_tokens is not None and "max_tokens" not in llm_config:
                llm_config["max_tokens"] = self._max_tokens
            llm_instance = self._llm.__class__(llm_config)
            return llm_instance.chat([{"role": "user", "content": prompt_text}])

    return RagasLLMAdapter(llm, wrapped_max_tokens=max_tokens)


def _create_ragas_embedding_adapter(ragas_base_cls: type[Any], embedding: BaseEmbedding) -> Any:
    class RagasEmbeddingAdapter(ragas_base_cls):
        def __init__(self, wrapped_embedding: BaseEmbedding) -> None:
            super().__init__()
            self._embedding = wrapped_embedding

        def embed_query(self, text: str) -> list[float]:
            return self._embedding.embed([text])[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embedding.embed(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    return RagasEmbeddingAdapter(embedding)


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

        return self._normalize_metrics(result, self._metric_names())

    def _run_ragas(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        reference: str,
    ) -> Any:
        ragas = self._import_required("ragas")
        metrics_module = self._import_required("ragas.metrics")
        dataset_schema = self._import_required("ragas.dataset_schema")
        run_config_module = self._import_required("ragas.run_config")

        metrics = self._build_metrics(metrics_module)
        dataset = dataset_schema.EvaluationDataset.from_list(
            [
                {
                    "user_input": query,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": reference,
                }
            ]
        )
        evaluate_kwargs: dict[str, Any] = {
            "metrics": metrics,
            "show_progress": False,
            "raise_exceptions": True,
            "run_config": self._build_run_config(run_config_module),
        }
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

    def _build_metrics(self, metrics_module: Any) -> list[Any]:
        metrics: list[Any] = []
        for name in self._metric_names():
            metric = copy.deepcopy(getattr(metrics_module, name))
            if name == "answer_relevancy" and hasattr(metric, "strictness"):
                strictness = self.config.get("answer_relevancy_strictness", 1)
                if isinstance(strictness, int) and strictness > 0:
                    metric.strictness = strictness
            metrics.append(metric)
        return metrics

    def _build_run_config(self, run_config_module: Any) -> Any:
        timeout = self._positive_int(self.config.get("timeout"), default=60)
        max_retries = self._non_negative_int(self.config.get("max_retries"), default=1)
        max_wait = self._positive_int(self.config.get("max_wait"), default=5)
        max_workers = self._positive_int(self.config.get("max_workers"), default=4)
        return run_config_module.RunConfig(
            timeout=timeout,
            max_retries=max_retries,
            max_wait=max_wait,
            max_workers=max_workers,
        )

    def _build_ragas_llm(self) -> Any | None:
        llm_config = self.config.get("llm_config")
        if not isinstance(llm_config, dict):
            return None

        provider = str(llm_config.get("provider", "")).strip().lower()
        model = str(llm_config.get("model", "")).strip()
        if not provider or not model:
            return None

        if provider not in {"openai", "qwen", "deepseek", "ollama"}:
            raise RagasEvaluatorError(
                f"ragas evaluator config error: unsupported llm provider for ragas: {provider}"
            )

        ragas_base_llm = self._import_required("ragas.llms.base").BaseRagasLLM
        llm_factory = self._import_required("libs.llm.llm_factory").LLMFactory
        llm = llm_factory.create({"llm": llm_config})
        max_tokens = llm_config.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            max_tokens = 4096
        return _create_ragas_llm_adapter(ragas_base_llm, llm, max_tokens=max_tokens)

    def _build_ragas_embeddings(self) -> Any | None:
        embedding_config = self.config.get("embedding_config")
        if not isinstance(embedding_config, dict):
            return None

        provider = str(embedding_config.get("provider", "")).strip().lower()
        model = str(embedding_config.get("model", "")).strip()
        if not provider or not model:
            return None

        if provider not in {"openai", "qwen", "deepseek", "ollama"}:
            raise RagasEvaluatorError(
                f"ragas evaluator config error: unsupported embedding provider for ragas: {provider}"
            )

        ragas_base_embeddings = self._import_required("ragas.embeddings.base").BaseRagasEmbeddings
        embedding_factory = self._import_required("libs.embedding.embedding_factory").EmbeddingFactory
        embedding = embedding_factory.create({"embedding": embedding_config})
        return _create_ragas_embedding_adapter(ragas_base_embeddings, embedding)

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
    def _normalize_metrics(result: Any, metric_names: list[str]) -> dict[str, float]:
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
        expected_keys = {str(name).strip() for name in metric_names if str(name).strip()}
        for key in expected_keys:
            if key not in result:
                continue
            value = result[key]
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise RagasEvaluatorError(
                    f"ragas evaluator response error: metric {key} must be numeric"
                ) from exc
        if not metrics:
            raise RagasEvaluatorError("ragas evaluator response error: metrics are empty")
        return metrics

    @staticmethod
    def _positive_int(value: Any, *, default: int) -> int:
        return value if isinstance(value, int) and value > 0 else default

    @staticmethod
    def _non_negative_int(value: Any, *, default: int) -> int:
        return value if isinstance(value, int) and value >= 0 else default
