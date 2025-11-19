"""
Configuration for the scribe-agent service.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from database.session import DatabaseSettings


class ScribeAgentSettings(DatabaseSettings):
    """
    Settings for the scribe-agent service.
    """

    version: str = Field(default="0.1.0", alias="SCRIBE_AGENT_VERSION")
    model_version: str = Field(default="scribe_v0", alias="SCRIBE_AGENT_MODEL_VERSION")
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        alias="SCRIBE_AGENT_CORS_ORIGINS",
        description="Comma-separated list of allowed origins for CORS.",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
