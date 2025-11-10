from __future__ import annotations

import logging
import sys
from typing import Optional

from .context import get_request_id


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def configure_logging(app_env: str = "development") -> None:
    level = logging.DEBUG if app_env.lower() == "development" else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger_name = name or "medios"
    logger = logging.getLogger(logger_name)
    logger.addFilter(RequestIdFilter())
    return logger


__all__ = ["configure_logging", "get_logger"]
