"""
SOAP generation endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.scribe_agent.core.soap import ScribeEngine
from services.scribe_agent.dependencies import get_scribe_engine
from services.scribe_agent.schemas import SoapGenerateRequest, SoapResponse
from services.scribe_agent.services.soap_service import SoapService

router = APIRouter(prefix="/scribe", tags=["soap"])


@router.post(
    "/generate-soap",
    response_model=SoapResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate SOAP note",
    description="Generate a SOAP note from a dialogue transcript and persist it.",
)
async def generate_soap_note(
    payload: SoapGenerateRequest,
    session: AsyncSession = Depends(get_session),
    engine: ScribeEngine = Depends(get_scribe_engine),
) -> SoapResponse:
    """
    Generate and store a SOAP note for the provided encounter.
    """

    generation = await engine.generate(payload)
    service = SoapService(session)
    note = await service.save_generated_note(
        encounter_id=payload.encounter_id,
        subjective=generation.subjective,
        objective=generation.objective,
        assessment=generation.assessment,
        plan=generation.plan,
        model_version=generation.model_version,
        confidence_score=0.5,
    )
    return SoapResponse.model_validate(note, from_attributes=True)
