"""MCP Client 端到端调用测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


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


def test_mcp_client_can_initialize_list_tools_and_call_query_knowledge_hub() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    server_script = """
from core.types import RetrievalResult
from mcp_server.protocol_handler import ProtocolHandler
from mcp_server.server import McpServer
from mcp_server.tools.query_knowledge_hub import build_query_knowledge_hub_tool

def fake_executor(query: str, top_k: int, collection: str | None):
    assert query == "What is Modular RAG?"
    assert top_k == 3
    assert collection == "demo"
    return [
        RetrievalResult(
            chunk_id="chunk-modular-rag-001",
            score=0.93,
            text="Modular RAG is a Retrieval-Augmented Generation system built from pluggable components.",
            metadata={"source_path": "tests/fixtures/sample_documents/blogger_intro.pdf", "page": 1},
        )
    ]

handler = ProtocolHandler(tools=[build_query_knowledge_hub_tool(executor=fake_executor)])
raise SystemExit(McpServer(protocol_handler=handler).serve_forever())
"""
    process = subprocess.Popen(
        ["uv", "run", "python", "-c", server_script],
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
                    "clientInfo": {"name": "pytest-e2e", "version": "1.0.0"},
                },
            },
        )
        initialize_response = _read_message(process.stdout)

        _write_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
            },
        )
        tools_response = _read_message(process.stdout)

        _write_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": "What is Modular RAG?",
                        "top_k": 1,
                        "collection": "demo",
                    },
                },
            },
        )
        tool_response = _read_message(process.stdout)

        assert process.stdin is not None
        process.stdin.close()
        exit_code = process.wait(timeout=10)
        stderr_text = process.stderr.read().decode("utf-8")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert exit_code == 0
    assert initialize_response["result"]["serverInfo"]["name"] == "modular-rag-mcp"
    tool_names = [tool["name"] for tool in tools_response["result"]["tools"]]
    assert tool_names == ["query_knowledge_hub"]

    result = tool_response["result"]
    assert result["content"][0]["type"] == "text"
    assert "Modular RAG is a Retrieval-Augmented Generation system" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"][0]["source"] == "tests/fixtures/sample_documents/blogger_intro.pdf"
    assert result["structuredContent"]["citations"][0]["page"] == 1
    assert result["structuredContent"]["results"][0]["chunk_id"] == "chunk-modular-rag-001"
    assert "handled initialize request" in stderr_text
    assert "handled tools/list request" in stderr_text
    assert "handled tools/call request" in stderr_text
