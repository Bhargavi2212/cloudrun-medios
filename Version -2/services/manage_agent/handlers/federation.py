"""
Federated learning management endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from services.manage_agent.dependencies import get_federated_trainer
from services.manage_agent.services.federated_trainer import FederatedTrainer

router = APIRouter(prefix="/manage/federation", tags=["federation"])


@router.post(
    "/train",
    summary="Trigger a local federated training round",
)
async def trigger_training(
    trainer: FederatedTrainer | None = Depends(get_federated_trainer),
) -> dict[str, float | int | str]:
    """
    Train a hospital-specific model and submit its weights to the aggregator.
    """

    if trainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Federated trainer not configured.",
        )

    result = await trainer.run_round()
    return {
        "status": "completed",
        "hospital_id": trainer.settings.dol_hospital_id,
        **result,
    }
