"""
Configuration for the manage-agent service.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from database.session import DatabaseSettings


class ManageAgentSettings(DatabaseSettings):
    """
    Settings tailored to the manage-agent service.
    """

    version: str = Field(default="0.1.0", alias="MANAGE_AGENT_VERSION")
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        alias="MANAGE_AGENT_CORS_ORIGINS",
        description="Comma-separated list of trusted origins.",
    )
    model_version: str = Field(default="triage_v0", alias="MANAGE_AGENT_MODEL_VERSION")
    dol_base_url: str | None = Field(
        default=None,
        alias="MANAGE_AGENT_DOL_URL",
        description="Base URL of the DOL service (e.g., http://localhost:8004).",
    )
    dol_shared_secret: str | None = Field(
        default=None,
        alias="MANAGE_AGENT_DOL_SECRET",
        description="Shared secret used when contacting the DOL.",
    )
    dol_hospital_id: str = Field(
        default="hospital-a",
        alias="MANAGE_AGENT_HOSPITAL_ID",
        description="Identifier for this hospital when communicating with DOL.",
    )
    storage_root: str = Field(
        default="./storage",
        alias="MANAGE_AGENT_STORAGE_ROOT",
        description="Root directory for storing uploaded files.",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | list[str]) -> list[str]:
        """
        Allow origins to be supplied as comma-separated string or list.
        """

        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
