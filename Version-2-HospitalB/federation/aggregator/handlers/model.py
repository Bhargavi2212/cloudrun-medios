"""
Model aggregation endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from federation.dependencies import get_aggregator
from federation.schemas import GlobalModel, ModelUpdate, ModelUpdateAck
from federation.security.auth import verify_shared_secret
from federation.services.aggregator_service import AggregatorService

router = APIRouter(prefix="/federation", tags=["federation"])


@router.post(
    "/submit",
    response_model=ModelUpdateAck,
    summary="Submit model update",
)
async def submit_model_update(
    payload: ModelUpdate,
    hospital_id: str = Depends(verify_shared_secret),
    aggregator: AggregatorService = Depends(get_aggregator),
) -> ModelUpdateAck:
    """
    Accept a model update from a hospital and perform aggregation.
    """

    if payload.hospital_id != hospital_id and payload.hospital_id != "unknown":
        # enforce alignment between header and payload
        payload = payload.model_copy(update={"hospital_id": hospital_id})

    aggregated = aggregator.submit_update(payload)
    return ModelUpdateAck(
        status="accepted",
        model_name=aggregated.model_name,
        round_id=aggregated.round_id,
        contributor_count=aggregated.contributor_count,
    )


@router.get(
    "/global-model/{model_name}",
    response_model=GlobalModel,
    summary="Fetch latest global model",
)
async def get_global_model(
    model_name: str,
    aggregator: AggregatorService = Depends(get_aggregator),
) -> GlobalModel:
    """
    Retrieve the latest aggregated model weights.
    """

    global_model = aggregator.get_global_model(model_name)
    if global_model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not aggregated yet."
        )
    return global_model
