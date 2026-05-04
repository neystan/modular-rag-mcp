"""Ollama Embedding 本地后端实现。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import EmbeddingProviderError


class OllamaEmbedding(BaseEmbedding):
    """通过 Ollama 本地 HTTP API 调用 Embedding 模型。"""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        payload = self._build_payload(texts)
        request = Request(
            self._embedding_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=float(self.config.get("timeout", 30))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise EmbeddingProviderError(f"{self.provider_name} HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise EmbeddingProviderError(f"{self.provider_name} network error: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingProviderError(f"{self.provider_name} response error: {type(exc).__name__}") from exc

        return self._parse_response(data)

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        model = self.config.get("model")
        if not model:
            raise EmbeddingProviderError(f"{self.provider_name} config error: model is required")

        return {
            "model": model,
            "input": self._normalize_texts(texts),
            "truncate": self.config.get("truncate", False),
        }

    def _normalize_texts(self, texts: list[str]) -> list[str]:
        if not isinstance(texts, list) or not texts:
            raise EmbeddingProviderError(f"{self.provider_name} input error: texts must be a non-empty list")

        max_input_length = self.config.get("max_input_length")
        too_long_strategy = str(self.config.get("too_long_strategy", "error")).strip().lower()
        if too_long_strategy not in {"error", "truncate"}:
            raise EmbeddingProviderError(
                f"{self.provider_name} config error: too_long_strategy must be error or truncate"
            )

        normalized: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingProviderError(f"{self.provider_name} input error: texts[{index}] must be string")
            if not text.strip():
                raise EmbeddingProviderError(f"{self.provider_name} input error: texts[{index}] cannot be empty")

            if isinstance(max_input_length, int) and max_input_length > 0 and len(text) > max_input_length:
                if too_long_strategy == "truncate":
                    normalized.append(text[:max_input_length])
                    continue
                raise EmbeddingProviderError(
                    f"{self.provider_name} input error: texts[{index}] exceeds max_input_length={max_input_length}"
                )

            normalized.append(text)
        return normalized

    def _embedding_endpoint(self) -> str:
        base_url = str(self.config.get("base_url", self.default_base_url)).rstrip("/")
        return f"{base_url}/api/embed"

    def _parse_response(self, data: dict[str, Any]) -> list[list[float]]:
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingProviderError(f"{self.provider_name} response error: missing embeddings")

        vectors: list[list[float]] = []
        for index, embedding in enumerate(embeddings):
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingProviderError(
                    f"{self.provider_name} response error: embeddings[{index}] must be non-empty list"
                )
            try:
                vectors.append([float(value) for value in embedding])
            except (TypeError, ValueError) as exc:
                raise EmbeddingProviderError(
                    f"{self.provider_name} response error: embeddings[{index}] must be numeric list"
                ) from exc
        return vectors
