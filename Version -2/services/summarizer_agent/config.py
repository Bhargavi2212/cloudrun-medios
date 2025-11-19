"""
Configuration for the summarizer-agent service.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field, field_validator
from pydantic_core import PydanticUndefined

from database.session import DatabaseSettings


class SummarizerAgentSettings(DatabaseSettings):
    """
    Settings for summarizer-agent.
    """

    version: str = Field(default="0.1.0", alias="SUMMARIZER_AGENT_VERSION")
    model_version: str = Field(
        default="summary_v0", alias="SUMMARIZER_AGENT_MODEL_VERSION"
    )

    # Store as string to prevent pydantic-settings from auto-parsing as JSON
    cors_allow_origins_str: str | None = Field(
        default=None,
        alias="SUMMARIZER_AGENT_CORS_ORIGINS",
        description="Trusted origins for CORS.",
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
        if value is None:
            return ["http://localhost:5173", "http://127.0.0.1:5173"]

        # Try to parse as JSON first (for JSON array strings)
        import json

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(origin).strip() for origin in parsed if origin]
        except (json.JSONDecodeError, ValueError):
            pass

        # If not JSON, treat as comma-separated string
        if not value.strip():
            return ["http://localhost:5173", "http://127.0.0.1:5173"]
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    gemini_api_key: str | None = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Google Gemini API key for document processing.",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        alias="GEMINI_MODEL",
        description="Gemini model to use for document processing.",
    )
    storage_root: str = Field(
        default="./storage",
        alias="SUMMARIZER_AGENT_STORAGE_ROOT",
        description="Root directory for stored files (must match manage-agent storage).",
    )
