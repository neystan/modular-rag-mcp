"""ProtocolHandler 单元测试。"""

from __future__ import annotations

from typing import Any

from mcp_server.protocol_handler import ProtocolHandler, ToolDefinition


def make_handler() -> ProtocolHandler:
    def echo_tool(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"echo:{arguments['query']}",
                }
            ]
        }

    def broken_tool(arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("backend exploded")

    return ProtocolHandler(
        tools=[
            ToolDefinition(
                name="query_knowledge_hub",
                description="查询知识库",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                handler=echo_tool,
            ),
            ToolDefinition(
                name="broken_tool",
                description="始终失败",
                input_schema={"type": "object", "properties": {}},
                handler=broken_tool,
            ),
        ]
    )


def test_initialize_returns_server_info_and_capabilities() -> None:
    handler = make_handler()

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp"
    assert response["result"]["capabilities"] == {"tools": {}}


def test_tools_list_returns_registered_tool_schemas() -> None:
    handler = make_handler()

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }
    )

    tools = response["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["query_knowledge_hub", "broken_tool"]
    assert tools[0]["inputSchema"]["required"] == ["query"]


def test_tools_call_routes_to_registered_tool() -> None:
    handler = make_handler()

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {"query": "azure"},
            },
        }
    )

    assert response["result"]["content"][0]["text"] == "echo:azure"


def test_unknown_method_returns_method_not_found() -> None:
    handler = make_handler()

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "missing/method",
        }
    )

    assert response["error"]["code"] == -32601


def test_invalid_tools_call_params_return_invalid_params() -> None:
    handler = make_handler()

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "arguments": {"query": "azure"},
            },
        }
    )

    assert response["error"]["code"] == -32602
    assert "params.name" in response["error"]["message"]


def test_tool_internal_error_returns_internal_error_without_stack() -> None:
    handler = make_handler()

    response = handler.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "broken_tool",
                "arguments": {},
            },
        }
    )

    assert response["error"]["code"] == -32603
    assert response["error"]["message"] == "Internal error"
