"""
Pydantic schemas for federation payloads.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelUpdate(BaseModel):
    """
    Payload representing a model update from a hospital.
    """

    model_name: str = Field(..., description="Model identifier (e.g., manage-triage).")
    round_id: int = Field(..., description="Federated learning round identifier.")
    hospital_id: str = Field(..., description="Identifier for the submitting hospital.")
    weights: dict[str, list[float]] = Field(
        ..., description="Layer weights or gradients."
    )
    num_samples: int = Field(
        ..., gt=0, description="Number of training samples used for this update."
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional metadata required to reconstruct the model "
            "(shape, classes, etc.)."
        ),
    )


class ModelUpdateAck(BaseModel):
    """
    Acknowledgement returned to hospitals after submitting an update.
    """

    status: str = Field(..., description="Status message.")
    model_name: str = Field(..., description="Model identifier.")
    round_id: int = Field(..., description="Federated learning round identifier.")
    contributor_count: int = Field(
        ..., description="Number of updates aggregated in this round."
    )


class GlobalModel(BaseModel):
    """
    Response payload representing the aggregated global model.
    """

    model_name: str = Field(..., description="Model identifier.")
    round_id: int = Field(..., description="Latest round aggregated.")
    weights: dict[str, list[float]] = Field(..., description="Aggregated weights.")
    contributor_count: int = Field(
        ..., description="Number of contributors to the round."
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata associated with the aggregated model.",
    )
