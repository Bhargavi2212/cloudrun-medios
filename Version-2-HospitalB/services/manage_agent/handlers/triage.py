"""
Triage classification endpoints.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.dependencies import get_triage_engine
from services.manage_agent.schemas import TriageRequest, TriageResponse

router = APIRouter(prefix="/manage", tags=["triage"])


@router.post(
    "/classify",
    response_model=TriageResponse,
    summary="Classify triage acuity",
    description="Generate an acuity score based on vital signs.",
)
async def classify_patient(
    payload: TriageRequest,
    engine: TriageEngine = Depends(get_triage_engine),
) -> TriageResponse:
    """
    Run the triage engine against incoming vital signs.
    """

    result = await engine.classify(payload)
    return TriageResponse.model_validate(asdict(result))
