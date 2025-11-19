"""
Clinical timeline management endpoints.
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from shared.database import get_db_session, ClinicalEventRepository, ClinicalEventType
from schemas import (
    ClinicalEventCreateRequest,
    ClinicalEventResponse,
    TimelineResponse
)

router = APIRouter()


@router.post("/events", response_model=ClinicalEventResponse)
async def create_clinical_event(
    event_data: ClinicalEventCreateRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Add a new clinical event to patient timeline."""
    try:
        event_repo = ClinicalEventRepository(db)
        
        clinical_event = await event_repo.create_clinical_event(
            patient_id=event_data.patient_id,
            event_type=event_data.event_type,
            clinical_summary=event_data.clinical_summary,
            timestamp=event_data.timestamp,
            structured_data=event_data.structured_data,
            ai_generated_insights=event_data.ai_generated_insights,
            confidence_score=event_data.confidence_score
        )
        
        return ClinicalEventResponse.from_attributes(clinical_event)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create clinical event: {str(e)}"
        )


@router.get("/{patient_id}", response_model=TimelineResponse)
async def get_patient_timeline(
    patient_id: str,
    event_type: Optional[ClinicalEventType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session)
):
    """Get complete clinical timeline for a patient."""
    event_repo = ClinicalEventRepository(db)
    
    events = await event_repo.get_patient_timeline(
        patient_id=patient_id,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    return TimelineResponse(
        patient_id=patient_id,
        total_events=len(events),
        events=[ClinicalEventResponse.from_attributes(event) for event in events],
        filters_applied={
            "event_type": event_type.value if event_type else None,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "limit": limit
        }
    )


@router.get("/events/{event_id}", response_model=ClinicalEventResponse)
async def get_clinical_event(
    event_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get specific clinical event by ID."""
    result = await db.execute(
        select(ClinicalEvent).where(ClinicalEvent.event_id == event_id)
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clinical event not found"
        )
    
    return ClinicalEventResponse.from_orm(event)


@router.get("/{patient_id}/summary")
async def get_timeline_summary(
    patient_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get summary statistics for patient timeline."""
    event_repo = ClinicalEventRepository(db)
    
    summary = await event_repo.get_timeline_summary(patient_id)
    
    if summary["total_events"] == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No timeline data found for patient"
        )
    
    return summary