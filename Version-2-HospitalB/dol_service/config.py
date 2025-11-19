"""
Configuration for the Data Orchestration Layer (DOL) service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

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
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        alias="DOL_CORS_ORIGINS",
        description="Comma-separated list of allowed origins.",
    )
    peers: list[PeerConfig] = Field(
        default_factory=list,
        alias="DOL_PEERS",
        description="JSON array describing peer endpoints.",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

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
