"""MCP 协议处理与能力协商。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "modular-rag-mcp"
SERVER_VERSION = "0.1.0"


@dataclass(slots=True, frozen=True)
class JsonRpcError:
    """JSON-RPC 错误对象。"""

    code: int
    message: str


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """协议层可注册工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def to_mcp_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }


class ProtocolHandler:
    """处理 initialize/tools/list/tools/call 的最小协议层。"""

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools = {tool.name: tool for tool in tools or []}

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params")

        if not isinstance(message, dict):
            return self._error_response(None, JsonRpcError(-32600, "Invalid Request"))
        if not isinstance(method, str) or not method.strip():
            return self._error_response(request_id, JsonRpcError(-32600, "Invalid Request"))

        if method == "initialized":
            return None
        if method == "initialize":
            try:
                result = self.handle_initialize(params)
            except (ProtocolHandlerError, ValueError, TypeError) as exc:
                return self._error_response(request_id, JsonRpcError(-32602, str(exc)))
            return self._success_response(request_id, result)
        if method == "tools/list":
            result = self.handle_tools_list()
            return self._success_response(request_id, result)
        if method == "tools/call":
            try:
                result = self._handle_tools_call_params(params)
            except (ProtocolHandlerError, ValueError, TypeError) as exc:
                return self._error_response(request_id, JsonRpcError(-32602, str(exc)))
            except Exception:  # noqa: BLE001
                return self._error_response(request_id, JsonRpcError(-32603, "Internal error"))
            return self._success_response(request_id, result)

        return self._error_response(request_id, JsonRpcError(-32601, f"Method not found: {method}"))

    def handle_initialize(self, params: Any) -> dict[str, Any]:
        if params is not None and not isinstance(params, dict):
            raise ProtocolHandlerError("initialize params must be object")

        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "capabilities": {
                "tools": {},
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        return {
            "tools": [tool.to_mcp_schema() for tool in self._tools.values()],
        }

    def handle_tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ProtocolHandlerError("tool name is required")
        if not isinstance(arguments, dict):
            raise ProtocolHandlerError("tool arguments must be object")

        tool = self._tools.get(normalized_name)
        if tool is None:
            raise ProtocolHandlerError(f"unknown tool: {normalized_name}")
        return tool.handler(dict(arguments))

    def _handle_tools_call_params(self, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ProtocolHandlerError("tools/call params must be object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ProtocolHandlerError("tools/call params.name is required")
        return self.handle_tools_call(name, arguments)

    @staticmethod
    def _success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "result": result,
        }

    @staticmethod
    def _error_response(request_id: Any, error: JsonRpcError) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {
                "code": error.code,
                "message": error.message,
            },
        }


class ProtocolHandlerError(ValueError):
    """协议层可读参数错误。"""
