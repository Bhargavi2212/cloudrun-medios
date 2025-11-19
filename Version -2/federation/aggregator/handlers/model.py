"""
Model aggregation endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from federation.dependencies import get_aggregator
from federation.schemas import GlobalModel, ModelUpdate, ModelUpdateAck
from federation.security.auth import verify_shared_secret
from federation.services.aggregator_service import AggregatorService
from federation.services.model_round_service import ModelRoundService

router = APIRouter(prefix="/federation", tags=["federation"])


@router.post(
    "/submit",
    response_model=ModelUpdateAck,
    summary="Submit model update",
)
async def submit_model_update(
    payload: ModelUpdate,
    hospital_id: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
    aggregator: AggregatorService = Depends(get_aggregator),
) -> ModelUpdateAck:
    """
    Accept a model update from a hospital and perform aggregation.
    """

    if payload.hospital_id != hospital_id and payload.hospital_id != "unknown":
        # enforce alignment between header and payload
        payload = payload.model_copy(update={"hospital_id": hospital_id})

    aggregated = aggregator.submit_update(payload)
    model_rounds = ModelRoundService(session)
    await model_rounds.record_round(aggregated)
    await session.commit()
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
    session: AsyncSession = Depends(get_session),
    aggregator: AggregatorService = Depends(get_aggregator),
) -> GlobalModel:
    """
    Retrieve the latest aggregated model weights.
    """

    global_model = aggregator.get_global_model(model_name)
    if global_model is not None:
        return global_model

    model_rounds = ModelRoundService(session)
    record = await model_rounds.latest(model_name)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not aggregated yet."
        )
    return GlobalModel(
        model_name=record.model_name,
        round_id=record.round_id,
        weights=record.weights,
        contributor_count=record.contributor_count,
        metadata=record.round_metadata,
    )


@router.get(
    "/status/{model_name}",
    response_model=list[GlobalModel],
    summary="Return recent aggregation rounds",
)
async def get_model_status(
    model_name: str,
    limit: int = Query(
        10, ge=1, le=50, description="Number of recent rounds to return."
    ),
    session: AsyncSession = Depends(get_session),
) -> list[GlobalModel]:
    """
    Provide telemetry for dashboards by returning the latest aggregated rounds.
    """

    model_rounds = ModelRoundService(session)
    records = await model_rounds.list_recent(model_name, limit=limit)
    return [
        GlobalModel(
            model_name=record.model_name,
            round_id=record.round_id,
            weights=record.weights,
            contributor_count=record.contributor_count,
            metadata=record.round_metadata,
        )
        for record in records
    ]
