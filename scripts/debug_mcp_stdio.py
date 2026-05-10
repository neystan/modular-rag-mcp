"""Debug stdin delivered by an MCP client over stdio."""

from __future__ import annotations

from pathlib import Path
import os
import select
import sys
import time


def main() -> None:
    log_path = Path("logs/mcp_stdio_debug.bin")
    meta_path = Path("logs/mcp_stdio_debug.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    deadline = time.monotonic() + 15
    chunks: list[bytes] = []
    fd = sys.stdin.buffer.fileno()
    while time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.5)
        if not readable:
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\r\n\r\n" in b"".join(chunks) or b"\n" in chunk:
            break

    data = b"".join(chunks)
    log_path.write_bytes(data)
    meta_path.write_text(
        "\n".join(
            [
                f"started_at={started_at}",
                f"pid={os.getpid()}",
                f"cwd={Path.cwd()}",
                f"bytes={len(data)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"captured {len(data)} bytes to {log_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
