"""Dashboard 启动入口。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "observability" / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
