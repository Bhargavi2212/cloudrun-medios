from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from ...services.error_response import StandardResponse
from ...services.summarizer_service import (
    MedicalSummarizer,
    PatientNotFoundError,
    SummarizerError,
)

router = APIRouter(prefix="/summarizer", tags=["summarizer"])
summarizer_service = MedicalSummarizer()


@router.get("/health", response_model=StandardResponse)
async def summarizer_health() -> StandardResponse:
    return StandardResponse(
        success=True,
        data={
            "enabled": not summarizer_service.is_stub,
            "is_stub": summarizer_service.is_stub,
        },
        is_stub=summarizer_service.is_stub,
    )


@router.get("/{subject_id}", response_model=StandardResponse)
async def summarise_subject(
    subject_id: int,
    visit_limit: Optional[int] = Query(None, ge=1, le=200),
    force_refresh: bool = Query(False),
) -> StandardResponse:
    try:
        result = summarizer_service.summarize_patient(
            subject_id,
            visit_limit=visit_limit,
            force_refresh=force_refresh,
        )
        return StandardResponse(
            success=True,
            data={
                "subject_id": result.subject_id,
                "summary_markdown": result.summary_markdown,
                "timeline": result.timeline,
                "metrics": result.metrics,
                "cached": result.cached,
            },
            is_stub=result.is_stub,
        )
    except PatientNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SummarizerError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

