"""MCP Server 入口。"""

from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

from core.settings import SettingsError, load_settings
from mcp_server.protocol_handler import ProtocolHandler
from observability.logger import get_logger


class McpServer:
    """基于 stdio transport 的最小 MCP Server。"""

    def __init__(
        self,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        stderr: BinaryIO | None = None,
        protocol_handler: ProtocolHandler | None = None,
    ) -> None:
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self.stderr = stderr or sys.stderr.buffer
        self.logger = get_logger(__name__)
        self.protocol_handler = protocol_handler or ProtocolHandler()

    def serve_forever(self) -> int:
        """循环读取并处理 MCP 消息，直到 EOF。"""

        while True:
            message = self._read_message()
            if message is None:
                self.logger.info("stdio closed, server exiting")
                return 0

            response = self._handle_message(message)
            if response is not None:
                self._write_message(response)

    def _handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if method == "initialize":
            self.logger.info("handled initialize request")
        elif method == "initialized":
            self.logger.info("received initialized notification")
        elif method == "tools/list":
            self.logger.info("handled tools/list request")
        elif method == "tools/call":
            self.logger.info("handled tools/call request")

        return self.protocol_handler.handle_message(message)

    def _read_message(self) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            line = self.stdin.readline()
            if line == b"":
                return None

            stripped = line.decode("utf-8").strip()
            if not stripped:
                break

            key, separator, value = stripped.partition(":")
            if not separator:
                raise ValueError(f"invalid header line: {stripped}")
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            raise ValueError("missing or invalid Content-Length header")

        payload = self.stdin.read(content_length)
        if len(payload) != content_length:
            raise ValueError("unexpected EOF while reading MCP payload")

        message = json.loads(payload.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError("MCP payload must be JSON object")
        return message

    def _write_message(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self.stdout.write(header)
        self.stdout.write(body)
        self.stdout.flush()

def main() -> None:
    """启动 MCP Server。"""

    logger = get_logger(__name__)
    try:
        settings = load_settings("config/settings.yaml")
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        raise SystemExit(1) from exc

    logger.info("mcp server starting: %s", settings.app["name"])
    server = McpServer()
    try:
        raise SystemExit(server.serve_forever())
    except Exception as exc:  # noqa: BLE001
        logger.error("mcp server fatal error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
