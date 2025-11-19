"""
Patient profile management endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from shared.database import get_db_session, ProfileRepository
from schemas import (
    ProfileCreateRequest,
    ProfileResponse,
    ProfileImportRequest,
    ProfileExportResponse
)

router = APIRouter()


@router.post("/", response_model=ProfileResponse)
async def create_profile(
    profile_data: ProfileCreateRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new patient profile."""
    try:
        profile_repo = ProfileRepository(db)
        
        profile = await profile_repo.create_profile(
            patient_id=profile_data.patient_id,
            first_name=profile_data.first_name,
            last_name=profile_data.last_name,
            date_of_birth=profile_data.date_of_birth,
            biological_sex=profile_data.biological_sex,
            active_medications=profile_data.active_medications,
            known_allergies=profile_data.known_allergies,
            chronic_conditions=profile_data.chronic_conditions,
            emergency_contacts=profile_data.emergency_contacts
        )
        
        return ProfileResponse.from_attributes(profile)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create profile: {str(e)}"
        )


@router.get("/{patient_id}", response_model=ProfileResponse)
async def get_profile(
    patient_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get patient profile by ID."""
    profile_repo = ProfileRepository(db)
    profile = await profile_repo.get_by_id(patient_id)
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    return ProfileResponse.from_attributes(profile)


@router.get("/", response_model=List[ProfileResponse])
async def list_profiles(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session)
):
    """List all patient profiles."""
    profile_repo = ProfileRepository(db)
    profiles = await profile_repo.get_all(skip=skip, limit=limit, order_by="last_updated")
    
    return [ProfileResponse.from_attributes(profile) for profile in profiles]


@router.post("/import")
async def import_profile(
    import_data: ProfileImportRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Import a portable profile from external source."""
    try:
        # TODO: Implement profile import logic
        # - Decrypt and verify profile data
        # - Validate cryptographic signatures
        # - Merge with existing local records
        # - Handle data conflicts
        
        return {
            "status": "success",
            "message": "Profile import functionality coming soon",
            "patient_id": import_data.encrypted_data[:20] + "..."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to import profile: {str(e)}"
        )


@router.get("/{patient_id}/export", response_model=ProfileExportResponse)
async def export_profile(
    patient_id: str,
    format: str = "json",
    db: AsyncSession = Depends(get_db_session)
):
    """Export patient profile for portable use."""
    # Get profile
    result = await db.execute(
        select(PortableProfile).where(PortableProfile.patient_id == patient_id)
    )
    profile = result.scalar_one_or_none()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    try:
        # TODO: Implement profile export logic
        # - Filter out hospital metadata
        # - Apply cryptographic signatures
        # - Generate QR codes or encrypted files
        # - Ensure privacy compliance
        
        return ProfileExportResponse(
            patient_id=patient_id,
            export_format=format,
            encrypted_data="placeholder_encrypted_data",
            qr_code_data="placeholder_qr_data" if format == "qr" else None,
            integrity_hash="placeholder_hash",
            export_timestamp="2024-01-01T00:00:00Z"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export profile: {str(e)}"
        )