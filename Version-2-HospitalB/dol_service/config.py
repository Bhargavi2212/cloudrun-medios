"""
Configuration for the Data Orchestration Layer (DOL) service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

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

    # Store as string to avoid JSON parsing, then convert to list
    cors_origins_str: str | None = Field(
        default=None,
        alias="DOL_CORS_ORIGINS",
        description="Comma-separated list of allowed origins.",
        exclude=True,
    )

    cors_allow_origins: list[str] = Field(
        default_factory=list,
        description="Parsed list of allowed CORS origins.",
    )

    peers: list[PeerConfig] = Field(
        default_factory=list,
        alias="DOL_PEERS",
        description="JSON array describing peer endpoints.",
    )

    @model_validator(mode="after")
    def _parse_cors_origins(self) -> DOLSettings:
        """Parse CORS origins string into list after model initialization."""
        if self.cors_origins_str:
            self.cors_allow_origins = [
                origin.strip()
                for origin in self.cors_origins_str.split(",")
                if origin.strip()
            ]
        return self

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
