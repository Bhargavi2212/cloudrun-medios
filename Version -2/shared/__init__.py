"""
Shared utilities for Medi OS Version -2.

This package centralizes reusable helpers (configuration, logging, exceptions)
that are consumed by each microservice.
"""

from __future__ import annotations

from shared.config import BaseAppSettings, get_settings
from shared.logging import configure_logging

__all__ = [
    "BaseAppSettings",
    "configure_logging",
    "get_settings",
]
