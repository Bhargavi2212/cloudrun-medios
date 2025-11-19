"""
Configuration for the manage-agent service.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field, field_validator
from pydantic_core import PydanticUndefined

from database.session import DatabaseSettings


class ManageAgentSettings(DatabaseSettings):
    """
    Settings tailored to the manage-agent service.
    """

    version: str = Field(default="0.1.0", alias="MANAGE_AGENT_VERSION")

    # Store as string to prevent pydantic-settings from auto-parsing as JSON
    cors_allow_origins_str: str | None = Field(
        default=None,
        alias="MANAGE_AGENT_CORS_ORIGINS",
        description="Comma-separated list of trusted origins.",
        exclude=True,  # Don't include in serialization
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
        if value is None:
            return ["http://localhost:5173", "http://127.0.0.1:5173"]

        # Try to parse as JSON first (for JSON array strings)
        import json

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(origin).strip() for origin in parsed if origin]
        except (json.JSONDecodeError, ValueError):
            pass

        # If not JSON, treat as comma-separated string
        if not value.strip():
            return ["http://localhost:5173", "http://127.0.0.1:5173"]
        return [origin.strip() for origin in value.split(",") if origin.strip()]

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
    orchestrator_base_url: str | None = Field(
        default=None,
        alias="ORCHESTRATOR_BASE_URL",
        description="Base URL for the orchestrator/DOL service.",
    )
    orchestrator_shared_secret: str | None = Field(
        default=None,
        alias="ORCHESTRATOR_SHARED_SECRET",
        description="Shared secret used when pushing snapshots to the orchestrator.",
    )
    hospital_name: str = Field(
        default="City Hospital",
        alias="MANAGE_AGENT_HOSPITAL_NAME",
        description="Friendly hospital name used during registry.",
    )
    manage_public_url: str | None = Field(
        default=None,
        alias="MANAGE_AGENT_PUBLIC_URL",
        description="Externally reachable Manage agent URL.",
    )
    scribe_public_url: str | None = Field(
        default=None,
        alias="SCRIBE_AGENT_PUBLIC_URL",
        description="Externally reachable Scribe agent URL.",
    )
    summarizer_public_url: str | None = Field(
        default=None,
        alias="SUMMARIZER_AGENT_PUBLIC_URL",
        description="Externally reachable Summarizer agent URL.",
    )
    dol_public_url: str | None = Field(
        default=None,
        alias="DOL_AGENT_PUBLIC_URL",
        description="Externally reachable DOL URL for peer lookups.",
    )
    federation_base_url: str | None = Field(
        default=None,
        alias="FEDERATION_BASE_URL",
        description="Base URL for the federated aggregator service.",
    )
    federation_shared_secret: str | None = Field(
        default=None,
        alias="FEDERATION_SHARED_SECRET",
        description="Shared secret used when submitting model updates.",
    )
    triage_features_path: str = Field(
        default="data/processed/X_train_final.csv",
        alias="TRIAGE_FEATURES_PATH",
        description="Path to the triage training feature matrix.",
    )
    triage_labels_path: str = Field(
        default="data/processed/y_train.csv",
        alias="TRIAGE_LABELS_PATH",
        description="Path to the triage training labels.",
    )
    federated_training_enabled: bool = Field(
        default=True,
        alias="FEDERATED_TRAINING_ENABLED",
        description="Whether the background federated training loop should run.",
    )
    federated_training_interval_seconds: int = Field(
        default=3600,
        alias="FEDERATED_TRAINING_INTERVAL_SECONDS",
        description="Interval between autonomous federated training rounds.",
    )
