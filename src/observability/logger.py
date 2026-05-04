"""基础 stderr logger。"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "modular_rag_mcp") -> logging.Logger:
    """返回写入 stderr 的项目 logger。"""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
