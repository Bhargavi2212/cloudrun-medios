"""
Summarizer package for longitudinal patient timeline generation.

Modules are organized by responsibility (data loading, visit grouping,
timeline construction, reasoning, API integration, etc.).
"""

from importlib import import_module
from typing import Any


def get_version() -> str:
    """Return the package version if defined, otherwise ``"0.1.0"``."""
    try:
        pkg = import_module("summarizer_agent")  # reuse existing version if available
        return getattr(pkg, "__version__", "0.1.0")
    except ModuleNotFoundError:
        return "0.1.0"


__all__ = ["get_version"]

