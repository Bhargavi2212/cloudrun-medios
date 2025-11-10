"""Pydantic response schema for consistent API responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class StandardResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    is_stub: bool = False
    warning: Optional[str] = None


__all__ = ["StandardResponse"]

