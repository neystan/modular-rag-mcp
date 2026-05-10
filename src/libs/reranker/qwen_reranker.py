"""Qwen DashScope Reranker 实现。"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.reranker.base_reranker import BaseReranker, RerankCandidate


class QwenRerankerError(RuntimeError):
    """Qwen Reranker 可读错误。"""


class QwenReranker(BaseReranker):
    """基于 DashScope 文本重排接口的 Qwen Reranker。"""

    provider_name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-api/v1"
    default_model = "qwen3-rerank"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        resolved_config = dict(config or {})
        if not resolved_config.get("api_key"):
            env_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("QWEN_API_KEY", "").strip()
            if env_api_key:
                resolved_config["api_key"] = env_api_key
        super().__init__(resolved_config)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        normalized_query = self._normalize_query(query)
        normalized_candidates = self._normalize_candidates(candidates)
        if not normalized_candidates:
            return []

        payload = self._build_payload(normalized_query, normalized_candidates)
        request = Request(
            self._rerank_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen(request, timeout=float(self.config.get("timeout", 30))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = self._extract_http_error_detail(exc)
            raise QwenRerankerError(f"qwen reranker HTTP error: {exc.code}{detail}") from exc
        except URLError as exc:
            raise QwenRerankerError(f"qwen reranker network error: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise QwenRerankerError(f"qwen reranker response error: {type(exc).__name__}") from exc

        return self._parse_response(data, normalized_candidates)

    def _build_payload(self, query: str, candidates: list[RerankCandidate]) -> dict[str, Any]:
        model = str(self.config.get("model", self.default_model)).strip()
        if not model:
            raise QwenRerankerError("qwen reranker config error: model is required")

        top_n = int(self.config.get("top_k", len(candidates)))
        payload: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": [candidate.text for candidate in candidates],
            "top_n": max(1, min(top_n, len(candidates))),
        }
        instruct = self.config.get("instruct")
        if isinstance(instruct, str) and instruct.strip():
            payload["instruct"] = instruct.strip()
        return payload

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise QwenRerankerError("qwen reranker input error: query is required")
        return query.strip()

    @staticmethod
    def _normalize_candidates(candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        if not isinstance(candidates, list):
            raise QwenRerankerError("qwen reranker input error: candidates must be list[RerankCandidate]")
        for candidate in candidates:
            if not isinstance(candidate, RerankCandidate):
                raise QwenRerankerError("qwen reranker input error: candidates must be list[RerankCandidate]")
            if not isinstance(candidate.text, str) or not candidate.text.strip():
                raise QwenRerankerError("qwen reranker input error: candidate text is required")
        return list(candidates)

    def _rerank_endpoint(self) -> str:
        base_url = str(self.config.get("base_url", self.default_base_url)).rstrip("/")
        return f"{base_url}/reranks"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _parse_response(self, data: dict[str, Any], candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            raise QwenRerankerError("qwen reranker response error: missing results")

        ranked: list[RerankCandidate] = []
        seen_indexes: set[int] = set()
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise QwenRerankerError(f"qwen reranker response error: results[{index}] must be object")
            candidate_index = item.get("index")
            if not isinstance(candidate_index, int):
                raise QwenRerankerError(f"qwen reranker response error: results[{index}].index must be int")
            if candidate_index < 0 or candidate_index >= len(candidates):
                raise QwenRerankerError(f"qwen reranker response error: results[{index}].index out of range")
            relevance_score = item.get("relevance_score")
            if not isinstance(relevance_score, (int, float)):
                raise QwenRerankerError(
                    f"qwen reranker response error: results[{index}].relevance_score must be numeric"
                )
            if candidate_index in seen_indexes:
                raise QwenRerankerError(f"qwen reranker response error: duplicate result index {candidate_index}")
            seen_indexes.add(candidate_index)
            ranked.append(replace(candidates[candidate_index], score=float(relevance_score)))
        return ranked

    @staticmethod
    def _extract_http_error_detail(exc: HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return ""

        error = payload.get("error")
        if not isinstance(error, dict):
            return ""
        code = str(error.get("code", "")).strip()
        message = str(error.get("message", "")).strip()
        details = [part for part in (f"code={code}" if code else "", message) if part]
        return f"; {'; '.join(details)}" if details else ""
