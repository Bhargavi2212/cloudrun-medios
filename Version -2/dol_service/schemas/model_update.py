"""
Model update pass-through schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelUpdateRequest(BaseModel):
    """
    Incoming model update from a peer hospital.
    """

    model_name: str = Field(..., description="Model identifier (e.g., manage-triage).")
    round_id: int = Field(..., description="Federated learning round identifier.")
    payload: dict[str, Any] = Field(
        ..., description="Serialized model update or metadata."
    )


class ModelUpdateResponse(BaseModel):
    """
    Acknowledgement payload for received model update.
    """

    status: str = Field(..., description="Acknowledgement status string.")
    model_name: str = Field(..., description="Model identifier.")
    round_id: int = Field(..., description="Federated round identifier.")
