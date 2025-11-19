"""
Configuration for the Data Orchestration Layer (DOL) service.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_core import PydanticUndefined

from database.session import DatabaseSettings


class PeerConfig(BaseModel):
    """
    Configuration for federated peer endpoints.
    """

    name: str = Field(..., description="Logical name for the peer hospital.")
    base_url: str = Field(..., description="Base URL of the peer DOL service.")
    api_key: str | None = Field(
        default=None,
        description="Shared secret or API key for authenticating with the peer.",
    )


class DOLSettings(DatabaseSettings):
    """
    Settings for the DOL service.
    """

    version: str = Field(default="0.1.0", alias="DOL_VERSION")
    hospital_id: str = Field(
        ..., alias="DOL_HOSPITAL_ID", description="Identifier for this hospital."
    )
    shared_secret: str = Field(
        ...,
        alias="DOL_SHARED_SECRET",
        description="Shared secret for peer authentication.",
    )

    # Store as string to prevent pydantic-settings from auto-parsing as JSON
    cors_allow_origins_str: str | None = Field(
        default=None,
        alias="DOL_CORS_ORIGINS",
        description="Comma-separated list of trusted origins.",
        exclude=True,  # Don't include in serialization
    )

    peers: list[PeerConfig] = Field(
        default_factory=list,
        alias="DOL_PEERS",
        description="JSON array describing peer endpoints.",
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

    @field_validator("peers", mode="before")
    @classmethod
    def _parse_peers(cls, value: str | list[PeerConfig]) -> list[PeerConfig]:
        if isinstance(value, str) and value.strip():
            import json

            data = json.loads(value)
            return [PeerConfig(**item) for item in data]
        if isinstance(value, list):
            return [
                PeerConfig(**item) if not isinstance(item, PeerConfig) else item
                for item in value
            ]
        return []
