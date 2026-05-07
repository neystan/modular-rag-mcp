"""Dashboard 启动入口。"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "observability" / "dashboard" / "app.py"
    uv_path = shutil.which("uv")
    if uv_path:
        command = [uv_path, "run", "streamlit", "run", str(app_path), *sys.argv[1:]]
    else:
        command = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
