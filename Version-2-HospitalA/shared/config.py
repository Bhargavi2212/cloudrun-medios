"""
Configuration helpers shared across Medi OS services.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from pydantic_settings import BaseSettings, SettingsConfigDict

_SettingsT = TypeVar("_SettingsT", bound="BaseAppSettings")


class BaseAppSettings(BaseSettings):
    """
    Base class for application settings.

    Attributes:
        environment: Deployment environment identifier (e.g., development, staging, production).
        debug: Flag indicating whether debug instrumentation should be enabled.
        log_level: Default log level name for the service logger.
    """

    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def root_path(self) -> Path:
        """
        Return the absolute path to the project's root directory.
        """

        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings(settings_cls: type[_SettingsT]) -> _SettingsT:
    """
    Instantiate and cache the provided settings class.

    Args:
        settings_cls: Subclass of `BaseAppSettings` to instantiate.

    Returns:
        Cached settings instance for the given class.
    """

    return settings_cls()
