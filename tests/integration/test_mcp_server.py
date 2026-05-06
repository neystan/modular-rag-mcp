"""MCP Server 集成测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.types import RetrievalResult
from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.tools.query_knowledge_hub import build_query_knowledge_hub_tool


def _write_message(stdin: object, payload: dict[str, object]) -> None:
    assert stdin is not None
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    stdin.write(body)
    stdin.flush()


def _read_message(stdout: object) -> dict[str, object]:
    assert stdout is not None
    headers: dict[str, str] = {}
    while True:
        line = stdout.readline()
        if line == b"":
            raise RuntimeError("unexpected EOF while reading MCP response")
        stripped = line.decode("utf-8").strip()
        if not stripped:
            break
        key, _, value = stripped.partition(":")
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers["content-length"])
    body = stdout.read(content_length)
    return json.loads(body.decode("utf-8"))


def test_server_handles_initialize_over_stdio_without_polluting_stdout() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "mcp_server.server"],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _write_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0.0"},
                },
            },
        )

        response = _read_message(process.stdout)

        process.stdin.close()
        exit_code = process.wait(timeout=10)
        stderr_text = process.stderr.read().decode("utf-8")
        trailing_stdout = process.stdout.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert exit_code == 0
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "modular-rag-mcp"
    assert result["capabilities"] == {"tools": {}}
    assert trailing_stdout == b""
    assert "mcp server starting" in stderr_text
    assert "handled initialize request" in stderr_text


def test_server_returns_method_not_found_for_unknown_request() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "mcp_server.server"],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _write_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "missing/method",
            },
        )

        response = _read_message(process.stdout)
        process.stdin.close()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 99
    assert response["error"]["code"] == -32601


def test_query_knowledge_hub_tool_returns_markdown_and_structured_citations() -> None:
    def fake_executor(query: str, top_k: int, collection: str | None) -> list[RetrievalResult]:
        assert query == "azure config"
        assert top_k == 2
        assert collection == "manuals"
        return [
            RetrievalResult(
                chunk_id="chunk-a",
                score=0.91,
                text="Azure OpenAI can be configured with endpoint and api key.",
                metadata={"source_path": "docs/azure.pdf", "page": 4},
            )
        ]

    handler = ProtocolHandler(tools=[build_query_knowledge_hub_tool(executor=fake_executor)])

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "azure config",
                    "top_k": 2,
                    "collection": "manuals",
                },
            },
        }
    )

    result = response["result"]
    assert result["content"][0]["type"] == "text"
    assert "[1]" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"][0]["source"] == "docs/azure.pdf"
    assert result["structuredContent"]["citations"][0]["page"] == 4


def test_query_knowledge_hub_tool_returns_friendly_empty_state() -> None:
    handler = ProtocolHandler(tools=[build_query_knowledge_hub_tool(executor=lambda q, k, c: [])])

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "unknown topic",
                },
            },
        }
    )

    result = response["result"]
    assert "未找到与“unknown topic”相关的文档" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"] == []
