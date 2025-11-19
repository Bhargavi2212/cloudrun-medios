"""
Profile management service with business logic.
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import (
    ProfileRepository,
    ClinicalEventRepository,
    LocalRecordRepository,
    PortableProfile,
    ClinicalEvent
)


class ProfileService:
    """Service for managing patient profiles with business logic."""
    
    def __init__(self, session: AsyncSession, hospital_id: str):
        """
        Initialize profile service.
        
        Args:
            session: Database session
            hospital_id: Hospital identifier
        """
        self.session = session
        self.hospital_id = hospital_id
        self.profile_repo = ProfileRepository(session)
        self.event_repo = ClinicalEventRepository(session)
        self.local_repo = LocalRecordRepository(session)
    
    async def create_patient_profile(
        self,
        patient_id: str,
        demographics: Dict[str, Any],
        medical_data: Optional[Dict[str, Any]] = None
    ) -> PortableProfile:
        """
        Create a new patient profile with integrity validation.
        
        Args:
            patient_id: Universal patient ID
            demographics: Patient demographic information
            medical_data: Initial medical data
            
        Returns:
            Created portable profile
        """
        # Validate patient ID format
        if not self._validate_patient_id(patient_id):
            raise ValueError(f"Invalid patient ID format: {patient_id}")
        
        # Check if profile already exists
        existing_profile = await self.profile_repo.get_by_id(patient_id)
        if existing_profile:
            raise ValueError(f"Profile already exists for patient: {patient_id}")
        
        # Extract demographics
        first_name = demographics.get("first_name")
        last_name = demographics.get("last_name")
        date_of_birth = demographics.get("date_of_birth")
        biological_sex = demographics.get("biological_sex")
        emergency_contacts = demographics.get("emergency_contacts", {})
        
        # Extract medical data
        medical_data = medical_data or {}
        active_medications = medical_data.get("active_medications", {})
        known_allergies = medical_data.get("known_allergies", {})
        chronic_conditions = medical_data.get("chronic_conditions", {})
        
        # Calculate integrity hash
        integrity_hash = self._calculate_profile_hash({
            "patient_id": patient_id,
            "demographics": demographics,
            "medical_data": medical_data
        })
        
        # Create profile
        profile = await self.profile_repo.create_profile(
            patient_id=patient_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            biological_sex=biological_sex,
            active_medications=active_medications,
            known_allergies=known_allergies,
            chronic_conditions=chronic_conditions,
            emergency_contacts=emergency_contacts,
            integrity_hash=integrity_hash
        )
        
        # Create corresponding local record
        await self.local_repo.create_local_record(
            portable_patient_id=patient_id,
            hospital_id=self.hospital_id
        )
        
        return profile
    
    async def import_patient_profile(
        self,
        encrypted_profile_data: str,
        verification_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a portable patient profile from external source.
        
        Args:
            encrypted_profile_data: Encrypted profile data
            verification_key: Key for signature verification
            
        Returns:
            Import result with status and patient ID
        """
        try:
            # TODO: Implement actual decryption and verification
            # For now, simulate successful import
            
            # Placeholder: Extract patient ID from encrypted data
            patient_id = f"MED-imported-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Check if profile already exists locally
            existing_local = await self.local_repo.get_by_portable_id(
                portable_patient_id=patient_id,
                hospital_id=self.hospital_id
            )
            
            if existing_local:
                # Profile exists, merge data
                return {
                    "status": "merged",
                    "patient_id": patient_id,
                    "message": "Profile merged with existing local record"
                }
            else:
                # New profile, create local record
                await self.local_repo.create_local_record(
                    portable_patient_id=patient_id,
                    hospital_id=self.hospital_id
                )
                
                return {
                    "status": "imported",
                    "patient_id": patient_id,
                    "message": "Profile imported successfully"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "patient_id": None,
                "message": f"Import failed: {str(e)}"
            }
    
    async def export_patient_profile(
        self,
        patient_id: str,
        export_format: str = "json",
        privacy_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Export patient profile for portable use.
        
        Args:
            patient_id: Patient identifier
            export_format: Export format (json, qr, fhir)
            privacy_level: Privacy level (minimal, standard, comprehensive)
            
        Returns:
            Export result with encrypted data
        """
        # Get profile with timeline
        profile = await self.profile_repo.get_profile_with_timeline(patient_id)
        if not profile:
            raise ValueError(f"Profile not found: {patient_id}")
        
        # Apply privacy filtering
        filtered_data = self._apply_privacy_filter(profile, privacy_level)
        
        # Generate export data
        export_data = {
            "patient_id": patient_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "export_format": export_format,
            "privacy_level": privacy_level,
            "profile_data": filtered_data
        }
        
        # TODO: Implement actual encryption and signing
        encrypted_data = self._encrypt_profile_data(export_data)
        
        # Generate QR code data if requested
        qr_code_data = None
        if export_format == "qr":
            qr_code_data = self._generate_qr_code(encrypted_data)
        
        # Calculate integrity hash
        integrity_hash = self._calculate_profile_hash(export_data)
        
        return {
            "patient_id": patient_id,
            "export_format": export_format,
            "encrypted_data": encrypted_data,
            "qr_code_data": qr_code_data,
            "integrity_hash": integrity_hash,
            "export_timestamp": export_data["export_timestamp"]
        }
    
    async def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get comprehensive patient summary.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Patient summary with profile and timeline data
        """
        # Get profile
        profile = await self.profile_repo.get_by_id(patient_id)
        if not profile:
            raise ValueError(f"Profile not found: {patient_id}")
        
        # Get timeline summary
        timeline_summary = await self.event_repo.get_timeline_summary(patient_id)
        
        # Get recent events
        recent_events = await self.event_repo.get_recent_events(patient_id, days=30)
        
        # Get local record
        local_record = await self.local_repo.get_by_portable_id(
            portable_patient_id=patient_id,
            hospital_id=self.hospital_id
        )
        
        return {
            "patient_id": patient_id,
            "profile": {
                "first_name": profile.first_name,
                "last_name": profile.last_name,
                "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
                "biological_sex": profile.biological_sex,
                "active_medications": profile.active_medications,
                "known_allergies": profile.known_allergies,
                "chronic_conditions": profile.chronic_conditions
            },
            "timeline_summary": timeline_summary,
            "recent_events": len(recent_events),
            "local_record": {
                "hospital_mrn": local_record.hospital_mrn if local_record else None,
                "department": local_record.department if local_record else None,
                "sync_status": local_record.profile_sync_status.value if local_record else None
            }
        }
    
    def _validate_patient_id(self, patient_id: str) -> bool:
        """Validate patient ID format (MED-{uuid4})."""
        import re
        pattern = r'^MED-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, patient_id))
    
    def _calculate_profile_hash(self, profile_data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of profile data."""
        # Convert to JSON string with sorted keys for consistent hashing
        json_str = json.dumps(profile_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _apply_privacy_filter(self, profile: PortableProfile, privacy_level: str) -> Dict[str, Any]:
        """Apply privacy filtering to profile data."""
        # Base data (always included)
        filtered_data = {
            "patient_id": profile.patient_id,
            "profile_version": profile.profile_version,
            "last_updated": profile.last_updated.isoformat()
        }
        
        if privacy_level in ["standard", "comprehensive"]:
            # Include demographics
            filtered_data.update({
                "first_name": profile.first_name,
                "last_name": profile.last_name,
                "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
                "biological_sex": profile.biological_sex
            })
        
        if privacy_level == "comprehensive":
            # Include full medical data
            filtered_data.update({
                "active_medications": profile.active_medications,
                "known_allergies": profile.known_allergies,
                "chronic_conditions": profile.chronic_conditions,
                "emergency_contacts": profile.emergency_contacts
            })
            
            # Include clinical timeline (privacy-filtered)
            if profile.clinical_events:
                filtered_data["clinical_timeline"] = [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "clinical_summary": event.clinical_summary,
                        "structured_data": event.structured_data,
                        "ai_generated_insights": event.ai_generated_insights
                        # Note: No hospital metadata included
                    }
                    for event in profile.clinical_events
                ]
        
        return filtered_data
    
    def _encrypt_profile_data(self, profile_data: Dict[str, Any]) -> str:
        """Encrypt profile data for transport."""
        # TODO: Implement actual encryption (AES-256)
        # For now, return base64 encoded JSON
        import base64
        json_str = json.dumps(profile_data)
        return base64.b64encode(json_str.encode()).decode()
    
    def _generate_qr_code(self, encrypted_data: str) -> str:
        """Generate QR code data for profile."""
        # TODO: Implement actual QR code generation
        # For now, return placeholder
        return f"QR_CODE_DATA:{encrypted_data[:50]}..."