"""OpenAI-Compatible Embedding 实现。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingProviderError(RuntimeError):
    """Embedding Provider 调用错误。"""


class OpenAIEmbedding(BaseEmbedding):
    """兼容 OpenAI Embeddings 协议的基础实现。"""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        payload = self._build_payload(texts)
        request = Request(
            self._embedding_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
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

        normalized_texts = self._normalize_texts(texts)
        payload: dict[str, Any] = {"model": model, "input": normalized_texts}
        if "dimensions" in self.config:
            payload["dimensions"] = self.config["dimensions"]
        if "encoding_format" in self.config:
            payload["encoding_format"] = self.config["encoding_format"]
        if "user" in self.config:
            payload["user"] = self.config["user"]
        return payload

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
        return f"{base_url}/embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _parse_response(self, data: dict[str, Any]) -> list[list[float]]:
        raw_embeddings = data.get("data")
        if not isinstance(raw_embeddings, list) or not raw_embeddings:
            raise EmbeddingProviderError(f"{self.provider_name} response error: missing data")

        vectors: list[list[float]] = []
        for index, item in enumerate(raw_embeddings):
            if not isinstance(item, dict):
                raise EmbeddingProviderError(
                    f"{self.provider_name} response error: data[{index}] must be object"
                )
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingProviderError(
                    f"{self.provider_name} response error: data[{index}].embedding is required"
                )
            try:
                vectors.append([float(value) for value in embedding])
            except (TypeError, ValueError) as exc:
                raise EmbeddingProviderError(
                    f"{self.provider_name} response error: data[{index}].embedding must be numeric list"
                ) from exc
        return vectors
