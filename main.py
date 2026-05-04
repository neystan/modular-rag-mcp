"""MCP Server 启动入口。"""

from __future__ import annotations

from core.settings import SettingsError, load_settings
from observability.logger import get_logger


def main() -> None:
    """加载配置并执行最小启动校验。"""

    logger = get_logger(__name__)
    try:
        settings = load_settings("config/settings.yaml")
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        raise SystemExit(1) from exc

    logger.info("配置加载成功: %s", settings.app["name"])


if __name__ == "__main__":
    main()
