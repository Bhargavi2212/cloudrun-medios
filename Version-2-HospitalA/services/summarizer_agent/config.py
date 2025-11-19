"""
Configuration for the summarizer-agent service.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from database.session import DatabaseSettings


class SummarizerAgentSettings(DatabaseSettings):
    """
    Settings for summarizer-agent.
    """

    version: str = Field(default="0.1.0", alias="SUMMARIZER_AGENT_VERSION")
    model_version: str = Field(
        default="summary_v0", alias="SUMMARIZER_AGENT_MODEL_VERSION"
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        alias="SUMMARIZER_AGENT_CORS_ORIGINS",
        description="Trusted origins for CORS.",
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

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
