from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import crud
from backend.database.schemas import PatientCreate, PatientRead
from backend.database.session import get_db_session
from backend.security.dependencies import require_roles
from backend.security.permissions import UserRole
from backend.services.error_response import StandardResponse

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=StandardResponse)
def list_patients(
    *,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=25, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> StandardResponse:
    skip = (page - 1) * size
    patients = crud.list_patients(session, skip=skip, limit=size)
    total = crud.count_patients(session)
    return StandardResponse(
        success=True,
        data={
            "items": [PatientRead.model_validate(patient).model_dump() for patient in patients],
            "total": total,
            "page": page,
            "size": size,
        },
    )


@router.get("/search", response_model=StandardResponse)
def search_patients(
    *,
    q: str = Query(..., min_length=2, description="Search string for patient name or MRN."),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db_session),
) -> StandardResponse:
    matches = crud.search_patients(session, q, limit=limit)
    return StandardResponse(
        success=True,
        data=[PatientRead.model_validate(patient).model_dump() for patient in matches],
    )


@router.post(
    "",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles(UserRole.RECEPTIONIST, UserRole.NURSE, UserRole.ADMIN))],
    status_code=status.HTTP_201_CREATED,
)
def create_patient(
    payload: PatientCreate,
    session: Session = Depends(get_db_session),
) -> StandardResponse:
    mrn = payload.mrn or f"MRN-{uuid4().hex[:8].upper()}"

    existing = crud.get_patient_by_mrn(session, mrn)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Patient with this MRN already exists.",
        )

    patient = crud.create_patient(
        session,
        mrn=mrn,
        first_name=payload.first_name,
        last_name=payload.last_name,
        date_of_birth=payload.date_of_birth,
        sex=payload.sex,
        contact_phone=payload.contact_phone,
        contact_email=payload.contact_email,
    )

    return StandardResponse(
        success=True,
        data=PatientRead.model_validate(patient).model_dump(),
    )

