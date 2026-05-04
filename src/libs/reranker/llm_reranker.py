"""LLM Reranker 实现。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class LLMRerankerError(RuntimeError):
    """LLM Reranker 可读错误。"""


class LLMReranker(BaseReranker):
    """使用 LLM 对候选片段进行结构化重排。"""

    default_prompt_path = "config/prompts/rerank.txt"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._prompt_text = self._load_prompt()
        self._llm_client = self._resolve_llm_client()

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        if not isinstance(query, str) or not query.strip():
            raise LLMRerankerError("llm reranker input error: query is required")
        if not candidates:
            return []

        ranked_ids = self._invoke_llm(query, candidates)
        candidate_map = {candidate.id: candidate for candidate in candidates}

        unknown_ids = [item_id for item_id in ranked_ids if item_id not in candidate_map]
        if unknown_ids:
            raise LLMRerankerError(
                f"llm reranker response error: unknown candidate ids: {', '.join(unknown_ids)}"
            )

        deduped_ids: list[str] = []
        for item_id in ranked_ids:
            if item_id not in deduped_ids:
                deduped_ids.append(item_id)

        remaining_ids = [candidate.id for candidate in candidates if candidate.id not in deduped_ids]
        ordered_ids = deduped_ids + remaining_ids

        top_k = int(self.config.get("top_k", len(candidates)))
        if top_k <= 0:
            return []
        return [candidate_map[item_id] for item_id in ordered_ids[:top_k]]

    def _resolve_llm_client(self) -> BaseLLM:
        llm_client = self.config.get("llm_client")
        if llm_client is not None:
            if not isinstance(llm_client, BaseLLM):
                raise LLMRerankerError("llm reranker config error: llm_client must inherit BaseLLM")
            return llm_client

        llm_config = self.config.get("llm")
        if not isinstance(llm_config, dict):
            raise LLMRerankerError("llm reranker config error: llm config is required")
        return LLMFactory.create({"llm": llm_config})

    def _load_prompt(self) -> str:
        prompt_text = self.config.get("prompt_text")
        if isinstance(prompt_text, str) and prompt_text.strip():
            return prompt_text.strip()

        prompt_path = Path(str(self.config.get("prompt_path", self.default_prompt_path)))
        if not prompt_path.exists():
            raise LLMRerankerError(f"llm reranker config error: prompt file not found: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise LLMRerankerError("llm reranker config error: prompt text is empty")
        return prompt

    def _invoke_llm(self, query: str, candidates: list[RerankCandidate]) -> list[str]:
        user_prompt = self._build_user_prompt(query, candidates)
        response = self._llm_client.chat(
            [
                ChatMessage(role="system", content=self._prompt_text),
                ChatMessage(role="user", content=user_prompt),
            ]
        )

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise LLMRerankerError("llm reranker response error: response must be valid JSON") from exc

        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not ranked_ids:
            raise LLMRerankerError("llm reranker response error: ranked_ids must be a non-empty list")
        if not all(isinstance(item, str) and item for item in ranked_ids):
            raise LLMRerankerError("llm reranker response error: ranked_ids must be a list of strings")
        return ranked_ids

    def _build_user_prompt(self, query: str, candidates: list[RerankCandidate]) -> str:
        payload = {
            "query": query,
            "candidates": [
                {
                    "id": candidate.id,
                    "text": candidate.text,
                    "score": candidate.score,
                    "metadata": candidate.metadata,
                }
                for candidate in candidates
            ],
            "output_schema": {"ranked_ids": ["candidate-id-1", "candidate-id-2"]},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
