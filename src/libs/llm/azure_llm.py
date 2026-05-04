"""Azure OpenAI LLM 实现。"""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class AzureLLM(OpenAICompatibleLLM):
    """Azure OpenAI Chat Completions 实现。"""

    provider_name = "azure"

    def _chat_endpoint(self) -> str:
        endpoint = str(self.config.get("endpoint", self.config.get("base_url", ""))).rstrip("/")
        deployment = self.config.get("deployment") or self.config.get("model")
        api_version = self.config.get("api_version", "2024-02-15-preview")
        if not endpoint:
            raise ValueError("azure config error: endpoint is required")
        if not deployment:
            raise ValueError("azure config error: deployment/model is required")
        return f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key:
            headers["api-key"] = str(api_key)
        return headers
