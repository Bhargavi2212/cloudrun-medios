"""
Federated patient management API routes.

This module handles federated patient profile operations including
import, export, and secure communication between hospitals.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..services.privacy_service import PrivacyFilterService
from ..services.profile_service import PortableProfileService
from ..services.crypto_service import CryptographicService
from ..config import get_config
from ..schemas.patient import (
    PortablePatientProfile,
    ProfileImportRequest,
    ProfileExportRequest,
    ProfileImportResponse,
    ProfileExportResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()
config = get_config()

# Initialize services
privacy_service = PrivacyFilterService(config.HOSPITAL_ID)
profile_service = PortableProfileService(config.HOSPITAL_ID)
crypto_service = CryptographicService(config.CRYPTO_CONFIG)


@router.post("/patient/import", response_model=ProfileImportResponse)
async def import_patient_profile(
    import_request: ProfileImportRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Import a portable patient profile from another hospital.
    
    This endpoint handles the secure import of patient profiles while
    maintaining privacy and verifying cryptographic integrity.
    """
    try:
        logger.info(f"Importing patient profile: {import_request.profile_id}")
        
        # Verify cryptographic integrity
        verification_result = await crypto_service.verify_profile_integrity(
            import_request.encrypted_profile_data,
            import_request.signature
        )
        
        if not verification_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Profile integrity verification failed: {verification_result.error}"
            )
        
        # Decrypt and parse profile
        decrypted_profile = await crypto_service.decrypt_profile(
            import_request.encrypted_profile_data,
            import_request.encryption_key
        )
        
        # Apply privacy filtering
        filtered_profile = await privacy_service.filter_imported_profile(decrypted_profile)
        
        # Import into local system
        import_result = await profile_service.import_profile(
            filtered_profile,
            import_request.merge_strategy
        )
        
        # Schedule background tasks
        background_tasks.add_task(
            _log_profile_import,
            import_request.profile_id,
            import_result.local_patient_id,
            verification_result.source_hospital_fingerprint
        )
        
        return ProfileImportResponse(
            success=True,
            local_patient_id=import_result.local_patient_id,
            timeline_entries_imported=import_result.timeline_entries_imported,
            conflicts_detected=import_result.conflicts_detected,
            verification_status=verification_result.status,
            import_timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Profile import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/patient/export", response_model=ProfileExportResponse)
async def export_patient_profile(
    export_request: ProfileExportRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Export a portable patient profile for transfer to another hospital.
    
    This endpoint creates a privacy-filtered, cryptographically signed
    portable profile that can be safely transferred between hospitals.
    """
    try:
        logger.info(f"Exporting patient profile: {export_request.patient_id}")
        
        # Get patient profile from local system
        local_profile = await profile_service.get_patient_profile(
            export_request.patient_id
        )
        
        if not local_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {export_request.patient_id} not found"
            )
        
        # Apply privacy filtering to remove hospital metadata
        filtered_profile = await privacy_service.filter_exported_profile(
            local_profile,
            export_request.privacy_level
        )
        
        # Create portable profile
        portable_profile = await profile_service.create_portable_profile(
            filtered_profile,
            export_request.export_format,
            export_request.include_full_timeline
        )
        
        # Apply cryptographic signing
        signed_profile = await crypto_service.sign_profile(
            portable_profile,
            config.HOSPITAL_ID
        )
        
        # Encrypt if requested
        if export_request.encrypt_profile:
            encrypted_profile = await crypto_service.encrypt_profile(
                signed_profile,
                export_request.recipient_public_key
            )
        else:
            encrypted_profile = signed_profile
        
        return ProfileExportResponse(
            success=True,
            profile_id=portable_profile.profile_id,
            encrypted_profile_data=encrypted_profile.data,
            signature=encrypted_profile.signature,
            export_format=export_request.export_format,
            privacy_level=export_request.privacy_level,
            timeline_entries_count=len(portable_profile.clinical_timeline),
            export_timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow().replace(year=datetime.utcnow().year + 1)  # 1 year expiry
        )
        
    except Exception as e:
        logger.error(f"Profile export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/patient/{patient_id}/status")
async def get_patient_status(
    patient_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get patient profile status and metadata.
    
    Returns information about the patient's profile without exposing
    sensitive clinical data or hospital-identifying information.
    """
    try:
        status = await profile_service.get_patient_status(patient_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found"
            )
        
        # Filter out hospital-identifying information
        filtered_status = await privacy_service.filter_patient_status(status)
        
        return {
            "patient_id": patient_id,
            "profile_exists": True,
            "timeline_entries_count": filtered_status.timeline_entries_count,
            "last_updated": filtered_status.last_updated,
            "profile_version": filtered_status.profile_version,
            "has_active_medications": filtered_status.has_active_medications,
            "has_known_allergies": filtered_status.has_known_allergies,
            "emergency_contact_available": filtered_status.emergency_contact_available,
            # Hospital metadata excluded for privacy
            "privacy_compliant": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get patient status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/patient/{patient_id}/verify")
async def verify_patient_profile(
    patient_id: str,
    verification_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Verify patient profile integrity and authenticity.
    
    Checks cryptographic signatures and validates profile data
    without exposing sensitive information.
    """
    try:
        verification_result = await crypto_service.verify_patient_profile(
            patient_id,
            verification_data
        )
        
        return {
            "patient_id": patient_id,
            "verification_status": verification_result.status,
            "is_authentic": verification_result.is_authentic,
            "signature_valid": verification_result.signature_valid,
            "timeline_integrity": verification_result.timeline_integrity,
            "last_verification": datetime.utcnow(),
            "verification_details": {
                "signatures_checked": verification_result.signatures_checked,
                "tamper_evidence": verification_result.tamper_evidence,
                "chain_of_custody": verification_result.chain_of_custody
            }
        }
        
    except Exception as e:
        logger.error(f"Profile verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/privacy/compliance")
async def check_privacy_compliance(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Check privacy compliance status of the DOL service.
    
    Returns information about privacy guarantees and compliance
    without exposing any patient data.
    """
    try:
        compliance_status = await privacy_service.check_compliance_status()
        
        return {
            "hospital_id": config.HOSPITAL_ID,
            "privacy_compliant": compliance_status.is_compliant,
            "compliance_checks": {
                "hospital_metadata_filtering": compliance_status.metadata_filtering_active,
                "differential_privacy_enabled": compliance_status.differential_privacy_enabled,
                "cryptographic_signing": compliance_status.crypto_signing_active,
                "audit_logging_active": compliance_status.audit_logging_active,
                "patient_data_sovereignty": compliance_status.patient_sovereignty_maintained
            },
            "privacy_guarantees": [
                "Zero hospital metadata in exported profiles",
                "Cryptographic integrity verification",
                "Patient-controlled data access",
                "Differential privacy for federated learning",
                "Complete audit trails without PHI exposure"
            ],
            "last_compliance_check": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Privacy compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")


async def _log_profile_import(
    profile_id: str,
    local_patient_id: str,
    source_hospital_fingerprint: str
):
    """
    Background task to log profile import for audit purposes.
    
    Logs the import without exposing patient data or hospital identities.
    """
    try:
        audit_entry = {
            "event_type": "profile_import",
            "profile_id": profile_id,
            "local_patient_id": local_patient_id,
            "source_hospital_fingerprint": source_hospital_fingerprint,  # Fingerprint, not hospital name
            "timestamp": datetime.utcnow(),
            "hospital_id": config.HOSPITAL_ID
        }
        
        # Log to audit system (implementation depends on audit service)
        logger.info(f"Profile import audit: {audit_entry}")
        
    except Exception as e:
        logger.error(f"Failed to log profile import: {e}")