"""
Configuration models for federation components.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field, field_validator
from pydantic_core import PydanticUndefined

from database.session import DatabaseSettings


class AggregatorSettings(DatabaseSettings):
    """
    Settings for the model aggregator service.
    """

    version: str = Field(default="0.1.0", alias="FEDERATION_VERSION")
    shared_secret: str = Field(..., alias="FEDERATION_SHARED_SECRET")

    # Store as string to prevent pydantic-settings from auto-parsing as JSON
    cors_allow_origins_str: str | None = Field(
        default=None,
        alias="FEDERATION_CORS_ORIGINS",
        description="Comma-separated list of trusted origins.",
        exclude=True,  # Don't include in serialization
    )

    @field_validator("cors_allow_origins_str", mode="before")
    @classmethod
    def _normalize_cors_str(cls, value: Any) -> str | None:
        """Normalize the CORS string value."""
        if value is None or value is PydanticUndefined:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # If already a list, join it
            return ",".join(str(v) for v in value)
        return None

    @computed_field
    @property
    def cors_allow_origins(self) -> list[str]:
        """
        Parse CORS origins from environment variable.
        Handles both JSON array strings and comma-separated strings.
        """
        value = self.cors_allow_origins_str
        if not value:
            # Default to localhost origins if not specified
            return ["http://localhost:5173", "http://127.0.0.1:5173"]

        # Try to parse as JSON first (for array strings)
        if value.strip().startswith("["):
            try:
                import json

                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # Otherwise, treat as comma-separated string
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        return (
            origins if origins else ["http://localhost:5173", "http://127.0.0.1:5173"]
        )
