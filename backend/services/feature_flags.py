"""Helpers for evaluating feature flags defined in configuration."""

from __future__ import annotations

from typing import Dict, Optional

from .config import get_settings


class FeatureFlagService:
    def __init__(self, flags: Optional[Dict[str, bool]] = None) -> None:
        self._flags = flags if flags is not None else get_settings().feature_flags

    def is_enabled(self, key: str, default: bool = False) -> bool:
        return self._flags.get(key, default)

    def all_flags(self) -> Dict[str, bool]:
        return dict(self._flags)


feature_flags = FeatureFlagService()

