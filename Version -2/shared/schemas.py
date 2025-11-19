"""
Shared Pydantic schemas for API responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StandardResponse(BaseModel):
    """
    Standard API response wrapper.
    """

    success: bool = Field(..., description="Whether the operation succeeded.")
    data: Any | None = Field(None, description="Response data payload.")
    error: str | None = Field(None, description="Error message if operation failed.")
    warning: str | None = Field(None, description="Optional warning message.")
