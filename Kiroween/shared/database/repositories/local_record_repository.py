"""
Repository for hospital-local patient records.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models import LocalPatientRecord, ProfileSyncStatus
from .base_repository import BaseRepository


class LocalRecordRepository(BaseRepository[LocalPatientRecord]):
    """Repository for hospital-local patient record operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(LocalPatientRecord, session)
    
    async def create_local_record(
        self,
        portable_patient_id: str,
        hospital_id: str,
        hospital_mrn: Optional[str] = None,
        admission_date: Optional[datetime] = None,
        discharge_date: Optional[datetime] = None,
        attending_physician: Optional[str] = None,
        department: Optional[str] = None,
        room_number: Optional[str] = None,
        insurance_info: Optional[Dict[str, Any]] = None,
        billing_codes: Optional[Dict[str, Any]] = None,
        detailed_clinical_notes: Optional[Dict[str, Any]] = None,
        complete_lab_results: Optional[Dict[str, Any]] = None,
        imaging_studies: Optional[Dict[str, Any]] = None,
        profile_sync_status: ProfileSyncStatus = ProfileSyncStatus.SYNCED
    ) -> LocalPatientRecord:
        """
        Create a new local patient record.
        
        Args:
            portable_patient_id: Links to MED-{uuid4} in portable profile
            hospital_id: Hospital identifier
            hospital_mrn: Hospital medical record number
            admission_date: Hospital admission date
            discharge_date: Hospital discharge date
            attending_physician: Attending physician name
            department: Hospital department
            room_number: Hospital room number
            insurance_info: Insurance and billing information
            billing_codes: ICD-10, CPT, and other billing codes
            detailed_clinical_notes: Complete clinical notes
            complete_lab_results: Complete lab results
            imaging_studies: Imaging studies and reports
            profile_sync_status: Profile synchronization status
            
        Returns:
            Created local patient record
        """
        return await self.create(
            portable_patient_id=portable_patient_id,
            hospital_id=hospital_id,
            hospital_mrn=hospital_mrn,
            admission_date=admission_date,
            discharge_date=discharge_date,
            attending_physician=attending_physician,
            department=department,
            room_number=room_number,
            insurance_info=insurance_info or {},
            billing_codes=billing_codes or {},
            detailed_clinical_notes=detailed_clinical_notes or {},
            complete_lab_results=complete_lab_results or {},
            imaging_studies=imaging_studies or {},
            profile_sync_status=profile_sync_status
        )
    
    async def get_by_portable_id(
        self,
        portable_patient_id: str,
        hospital_id: str
    ) -> Optional[LocalPatientRecord]:
        """
        Get local record by portable patient ID and hospital.
        
        Args:
            portable_patient_id: Portable patient identifier
            hospital_id: Hospital identifier
            
        Returns:
            Local patient record or None
        """
        result = await self.session.execute(
            select(LocalPatientRecord)
            .where(
                and_(
                    LocalPatientRecord.portable_patient_id == portable_patient_id,
                    LocalPatientRecord.hospital_id == hospital_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_by_hospital_mrn(
        self,
        hospital_mrn: str,
        hospital_id: str
    ) -> Optional[LocalPatientRecord]:
        """
        Get local record by hospital MRN.
        
        Args:
            hospital_mrn: Hospital medical record number
            hospital_id: Hospital identifier
            
        Returns:
            Local patient record or None
        """
        result = await self.session.execute(
            select(LocalPatientRecord)
            .where(
                and_(
                    LocalPatientRecord.hospital_mrn == hospital_mrn,
                    LocalPatientRecord.hospital_id == hospital_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_hospital_patients(
        self,
        hospital_id: str,
        skip: int = 0,
        limit: int = 100,
        active_only: bool = False
    ) -> List[LocalPatientRecord]:
        """
        Get all patients for a specific hospital.
        
        Args:
            hospital_id: Hospital identifier
            skip: Number of records to skip
            limit: Maximum records to return
            active_only: Only return patients without discharge date
            
        Returns:
            List of local patient records
        """
        query = select(LocalPatientRecord).where(LocalPatientRecord.hospital_id == hospital_id)
        
        if active_only:
            query = query.where(LocalPatientRecord.discharge_date.is_(None))
        
        query = query.offset(skip).limit(limit).order_by(LocalPatientRecord.admission_date.desc())
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_department(
        self,
        hospital_id: str,
        department: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[LocalPatientRecord]:
        """
        Get patients by hospital department.
        
        Args:
            hospital_id: Hospital identifier
            department: Department name
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of patients in department
        """
        result = await self.session.execute(
            select(LocalPatientRecord)
            .where(
                and_(
                    LocalPatientRecord.hospital_id == hospital_id,
                    LocalPatientRecord.department == department
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(LocalPatientRecord.admission_date.desc())
        )
        return list(result.scalars().all())
    
    async def get_by_physician(
        self,
        hospital_id: str,
        attending_physician: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[LocalPatientRecord]:
        """
        Get patients by attending physician.
        
        Args:
            hospital_id: Hospital identifier
            attending_physician: Physician name
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of patients under physician care
        """
        result = await self.session.execute(
            select(LocalPatientRecord)
            .where(
                and_(
                    LocalPatientRecord.hospital_id == hospital_id,
                    LocalPatientRecord.attending_physician == attending_physician
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(LocalPatientRecord.admission_date.desc())
        )
        return list(result.scalars().all())
    
    async def update_sync_status(
        self,
        local_record_id: str,
        sync_status: ProfileSyncStatus,
        last_import: Optional[datetime] = None,
        last_export: Optional[datetime] = None
    ) -> Optional[LocalPatientRecord]:
        """
        Update profile synchronization status.
        
        Args:
            local_record_id: Local record identifier
            sync_status: New synchronization status
            last_import: Last profile import timestamp
            last_export: Last profile export timestamp
            
        Returns:
            Updated local record or None
        """
        update_data = {"profile_sync_status": sync_status}
        
        if last_import:
            update_data["last_profile_import"] = last_import
        if last_export:
            update_data["last_profile_export"] = last_export
        
        return await self.update(local_record_id, **update_data)
    
    async def get_sync_pending_records(
        self,
        hospital_id: str,
        limit: int = 50
    ) -> List[LocalPatientRecord]:
        """
        Get records that need profile synchronization.
        
        Args:
            hospital_id: Hospital identifier
            limit: Maximum records to return
            
        Returns:
            List of records needing sync
        """
        result = await self.session.execute(
            select(LocalPatientRecord)
            .where(
                and_(
                    LocalPatientRecord.hospital_id == hospital_id,
                    LocalPatientRecord.profile_sync_status.in_([
                        ProfileSyncStatus.PENDING,
                        ProfileSyncStatus.CONFLICT,
                        ProfileSyncStatus.ERROR
                    ])
                )
            )
            .order_by(LocalPatientRecord.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_hospital_statistics(self, hospital_id: str) -> Dict[str, Any]:
        """
        Get statistics for a hospital.
        
        Args:
            hospital_id: Hospital identifier
            
        Returns:
            Dictionary with hospital statistics
        """
        # Get total patients
        total_patients = await self.session.execute(
            select(func.count(LocalPatientRecord.local_record_id))
            .where(LocalPatientRecord.hospital_id == hospital_id)
        )
        total_count = total_patients.scalar() or 0
        
        # Get active patients (not discharged)
        active_patients = await self.session.execute(
            select(func.count(LocalPatientRecord.local_record_id))
            .where(
                and_(
                    LocalPatientRecord.hospital_id == hospital_id,
                    LocalPatientRecord.discharge_date.is_(None)
                )
            )
        )
        active_count = active_patients.scalar() or 0
        
        # Get department counts
        dept_counts = await self.session.execute(
            select(
                LocalPatientRecord.department,
                func.count(LocalPatientRecord.local_record_id).label('count')
            )
            .where(LocalPatientRecord.hospital_id == hospital_id)
            .group_by(LocalPatientRecord.department)
        )
        department_stats = {dept: count for dept, count in dept_counts.all() if dept}
        
        return {
            "hospital_id": hospital_id,
            "total_patients": total_count,
            "active_patients": active_count,
            "discharged_patients": total_count - active_count,
            "department_counts": department_stats
        }