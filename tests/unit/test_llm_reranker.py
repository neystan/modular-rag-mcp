"""LLM Reranker 测试。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.llm_reranker import LLMReranker, LLMRerankerError
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM(BaseLLM):
    """测试用 LLM。"""

    def __init__(self, response: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.response = response
        self.messages: list[ChatMessage | dict[str, Any]] = []

    def chat(self, messages: list[ChatMessage] | list[dict[str, Any]]) -> str:
        self.messages = list(messages)
        return self.response


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "placeholder"},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={
            "provider": "llm",
            "top_k": 2,
            "prompt_text": "请返回 JSON",
            "llm_client": FakeLLM('{"ranked_ids":["b","a"]}'),
        },
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="文本 A", score=0.2, metadata={"source": "a.md"}),
        RerankCandidate(id="b", text="文本 B", score=0.8, metadata={"source": "b.md"}),
        RerankCandidate(id="c", text="文本 C", score=0.5, metadata={"source": "c.md"}),
    ]


def test_factory_creates_llm_reranker() -> None:
    reranker = RerankerFactory.create(make_settings())

    assert isinstance(reranker, LLMReranker)


def test_llm_reranker_reads_prompt_and_returns_ranked_candidates() -> None:
    llm = FakeLLM('{"ranked_ids":["b","a"]}')
    reranker = LLMReranker({"prompt_text": "系统提示", "llm_client": llm, "top_k": 2})

    ranked = reranker.rerank("测试查询", make_candidates())

    assert [item.id for item in ranked] == ["b", "a"]
    assert len(llm.messages) == 2
    assert isinstance(llm.messages[0], ChatMessage)
    assert llm.messages[0].content == "系统提示"
    user_payload = json.loads(llm.messages[1].content)
    assert user_payload["query"] == "测试查询"
    assert [item["id"] for item in user_payload["candidates"]] == ["a", "b", "c"]


def test_llm_reranker_appends_unranked_candidates_after_ranked_ids() -> None:
    reranker = LLMReranker(
        {"prompt_text": "系统提示", "llm_client": FakeLLM('{"ranked_ids":["c"]}'), "top_k": 3}
    )

    ranked = reranker.rerank("测试查询", make_candidates())

    assert [item.id for item in ranked] == ["c", "a", "b"]


def test_invalid_json_response_is_readable() -> None:
    reranker = LLMReranker({"prompt_text": "系统提示", "llm_client": FakeLLM("not-json")})

    with pytest.raises(LLMRerankerError, match="response must be valid JSON"):
        reranker.rerank("测试查询", make_candidates())


def test_missing_ranked_ids_is_readable() -> None:
    reranker = LLMReranker({"prompt_text": "系统提示", "llm_client": FakeLLM('{"items":["a"]}')})

    with pytest.raises(LLMRerankerError, match="ranked_ids must be a non-empty list"):
        reranker.rerank("测试查询", make_candidates())


def test_unknown_candidate_id_is_readable() -> None:
    reranker = LLMReranker({"prompt_text": "系统提示", "llm_client": FakeLLM('{"ranked_ids":["x"]}')})

    with pytest.raises(LLMRerankerError, match="unknown candidate ids: x"):
        reranker.rerank("测试查询", make_candidates())
