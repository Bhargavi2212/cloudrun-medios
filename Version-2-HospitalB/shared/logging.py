"""
Logging configuration helpers with emoji support.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from logging.config import dictConfig

EMOJI_LEVEL_MAP: Mapping[int, str] = {
    logging.DEBUG: "🔍",
    logging.INFO: "✅",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🔥",
}


def configure_logging(service_name: str, level: str = "INFO") -> None:
    """
    Configure structured logging for a service.

    Args:
        service_name: Name of the service emitting logs.
        level: Log level name (e.g., INFO, DEBUG).
    """

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "emoji": {
                    "format": "%(asctime)s %(levelname)s %(emoji)s "
                    "[%(name)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "filters": {
                "emoji_level": {
                    "()": "shared.logging.EmojiLevelFilter",
                    "service_name": service_name,
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "emoji",
                    "filters": ["emoji_level"],
                }
            },
            "root": {
                "handlers": ["default"],
                "level": level.upper(),
            },
        }
    )


class EmojiLevelFilter(logging.Filter):
    """
    Logging filter that injects emoji glyphs based on log severity.
    """

    def __init__(self, service_name: str) -> None:
        super().__init__(name=service_name)
        self.service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Inject the emoji attribute into the log record.

        Args:
            record: Log record to mutate.

        Returns:
            Always True to keep the record.
        """

        emoji = EMOJI_LEVEL_MAP.get(record.levelno, "i")
        record.emoji = emoji
        return True
