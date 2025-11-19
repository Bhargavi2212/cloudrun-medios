"""
Configuration models for federation components.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from shared.config import BaseAppSettings


class AggregatorSettings(BaseAppSettings):
    """
    Settings for the model aggregator service.
    """

    version: str = Field(default="0.1.0", alias="FEDERATION_VERSION")
    shared_secret: str = Field(..., alias="FEDERATION_SHARED_SECRET")
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        alias="FEDERATION_CORS_ORIGINS",
        description="Comma-separated list of allowed origins.",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
