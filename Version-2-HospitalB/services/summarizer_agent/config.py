"""
Configuration for the summarizer-agent service.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from database.session import DatabaseSettings


class SummarizerAgentSettings(DatabaseSettings):
    """
    Settings for summarizer-agent.
    """

    version: str = Field(default="0.1.0", alias="SUMMARIZER_AGENT_VERSION")
    model_version: str = Field(
        default="summary_v0", alias="SUMMARIZER_AGENT_MODEL_VERSION"
    )

    # Store as string to avoid JSON parsing, then convert to list
    # Using a non-private field name to avoid Pydantic validation issues
    cors_origins_str: str | None = Field(
        default=None,
        alias="SUMMARIZER_AGENT_CORS_ORIGINS",
        description="Comma-separated list of allowed origins for CORS.",
        exclude=True,
    )

    cors_allow_origins: list[str] = Field(
        default_factory=list,
        description="Parsed list of allowed CORS origins.",
    )

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
        description=(
            "Root directory for stored files " "(must match manage-agent storage)."
        ),
    )

    @model_validator(mode="after")
    def _parse_cors_origins(self) -> SummarizerAgentSettings:
        """Parse CORS origins string into list after model initialization."""
        if self.cors_origins_str:
            self.cors_allow_origins = [
                origin.strip()
                for origin in self.cors_origins_str.split(",")
                if origin.strip()
            ]
        return self
