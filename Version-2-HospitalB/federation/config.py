"""
Configuration models for federation components.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from shared.config import BaseAppSettings


class AggregatorSettings(BaseAppSettings):
    """
    Settings for the model aggregator service.
    """

    version: str = Field(default="0.1.0", alias="FEDERATION_VERSION")
    shared_secret: str = Field(..., alias="FEDERATION_SHARED_SECRET")

    # Store as string to avoid JSON parsing, then convert to list
    # Using a non-private field name to avoid Pydantic validation issues
    cors_origins_str: str | None = Field(
        default=None,
        alias="FEDERATION_CORS_ORIGINS",
        description="Comma-separated list of allowed origins.",
        exclude=True,
    )

    cors_allow_origins: list[str] = Field(
        default_factory=list,
        description="Parsed list of allowed CORS origins.",
    )

    @model_validator(mode="after")
    def _parse_cors_origins(self) -> AggregatorSettings:
        """Parse CORS origins string into list after model initialization."""
        if self.cors_origins_str:
            self.cors_allow_origins = [
                origin.strip()
                for origin in self.cors_origins_str.split(",")
                if origin.strip()
            ]
        return self
