"""
Configuration for the scribe-agent service.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from database.session import DatabaseSettings


class ScribeAgentSettings(DatabaseSettings):
    """
    Settings for the scribe-agent service.
    """

    version: str = Field(default="0.1.0", alias="SCRIBE_AGENT_VERSION")
    model_version: str = Field(default="scribe_v0", alias="SCRIBE_AGENT_MODEL_VERSION")

    # Store as string to avoid JSON parsing, then convert to list
    # Using a non-private field name to avoid Pydantic validation issues
    cors_origins_str: str | None = Field(
        default=None,
        alias="SCRIBE_AGENT_CORS_ORIGINS",
        description="Comma-separated list of allowed origins for CORS.",
        exclude=True,
    )

    cors_allow_origins: list[str] = Field(
        default_factory=list,
        description="Parsed list of allowed CORS origins.",
    )

    @model_validator(mode="after")
    def _parse_cors_origins(self) -> ScribeAgentSettings:
        """Parse CORS origins string into list after model initialization."""
        if self.cors_origins_str:
            self.cors_allow_origins = [
                origin.strip()
                for origin in self.cors_origins_str.split(",")
                if origin.strip()
            ]
        return self
