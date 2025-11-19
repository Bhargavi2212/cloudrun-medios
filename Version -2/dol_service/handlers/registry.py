"""
Registry endpoints for orchestrator metadata.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from dol_service.schemas.registry import HospitalRegistration, HospitalRegistryRead
from dol_service.security.auth import verify_shared_secret
from dol_service.services.registry_service import RegistryService

router = APIRouter(prefix="/api/dol", tags=["dol"])


@router.get(
    "/registry",
    response_model=list[HospitalRegistryRead],
    summary="List registered hospitals",
)
async def list_registry(
    session: AsyncSession = Depends(get_session),
) -> list[HospitalRegistryRead]:
    """
    Return the hospitals known to the orchestrator.
    """

    service = RegistryService(session)
    entries = await service.list_hospitals()
    return [HospitalRegistryRead.model_validate(entry) for entry in entries]


@router.post(
    "/registry",
    response_model=HospitalRegistryRead,
    summary="Register or heartbeat a hospital",
)
async def register_hospital(
    payload: HospitalRegistration,
    _: str = Depends(verify_shared_secret),
    session: AsyncSession = Depends(get_session),
) -> HospitalRegistryRead:
    """
    Create or update a hospital registry entry.
    """

    service = RegistryService(session)
    entry = await service.upsert(payload)
    await session.commit()
    return HospitalRegistryRead.model_validate(entry)
