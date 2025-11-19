"""
Database package for Medi OS Version -2.

This package will expose asynchronous SQLAlchemy models, sessions, and
repository utilities shared across hospital services.
"""

from __future__ import annotations

from importlib import import_module


def ensure_loaded() -> None:
    """
    Trigger import side effects for database models.

    This helper can be called during application startup to guarantee that
    SQLAlchemy metadata is fully populated before migrations or table creation.
    """

    import_module("database.models")
