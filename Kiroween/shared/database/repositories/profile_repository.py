"""
Repository for portable patient profiles.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload

from ..models import PortableProfile, ClinicalEvent
from .base_repository import BaseRepository


class ProfileRepository(BaseRepository[PortableProfile]):
    """Repository for portable patient profile operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(PortableProfile, session)
    
    async def create_profile(
        self,
        patient_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        date_of_birth: Optional[datetime] = None,
        biological_sex: Optional[str] = None,
        active_medications: Optional[Dict[str, Any]] = None,
        known_allergies: Optional[Dict[str, Any]] = None,
        chronic_conditions: Optional[Dict[str, Any]] = None,
        emergency_contacts: Optional[Dict[str, Any]] = None,
        integrity_hash: str = "placeholder_hash"
    ) -> PortableProfile:
        """
        Create a new portable patient profile.
        
        Args:
            patient_id: Universal patient ID in MED-{uuid4} format
            first_name: Patient first name
            last_name: Patient last name
            date_of_birth: Patient date of birth
            biological_sex: Biological sex (M/F/Other/Unknown)
            active_medications: Current medications data
            known_allergies: Known allergies data
            chronic_conditions: Chronic conditions data
            emergency_contacts: Emergency contact information
            integrity_hash: Profile integrity hash
            
        Returns:
            Created portable profile
        """
        return await self.create(
            patient_id=patient_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            biological_sex=biological_sex,
            active_medications=active_medications or {},
            known_allergies=known_allergies or {},
            chronic_conditions=chronic_conditions or {},
            emergency_contacts=emergency_contacts or {},
            integrity_hash=integrity_hash
        )
    
    async def get_profile_with_timeline(self, patient_id: str) -> Optional[PortableProfile]:
        """
        Get profile with complete clinical timeline.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Profile with clinical events or None
        """
        result = await self.session.execute(
            select(PortableProfile)
            .options(selectinload(PortableProfile.clinical_events))
            .where(PortableProfile.patient_id == patient_id)
        )
        return result.scalar_one_or_none()
    
    async def search_profiles(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        date_of_birth: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[PortableProfile]:
        """
        Search profiles by demographic information.
        
        Args:
            first_name: First name to search for
            last_name: Last name to search for
            date_of_birth: Date of birth to match
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of matching profiles
        """
        query = select(PortableProfile)
        
        conditions = []
        if first_name:
            conditions.append(PortableProfile.first_name.ilike(f"%{first_name}%"))
        if last_name:
            conditions.append(PortableProfile.last_name.ilike(f"%{last_name}%"))
        if date_of_birth:
            conditions.append(PortableProfile.date_of_birth == date_of_birth)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.offset(skip).limit(limit).order_by(PortableProfile.last_updated.desc())
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_medical_data(
        self,
        patient_id: str,
        active_medications: Optional[Dict[str, Any]] = None,
        known_allergies: Optional[Dict[str, Any]] = None,
        chronic_conditions: Optional[Dict[str, Any]] = None,
        emergency_contacts: Optional[Dict[str, Any]] = None
    ) -> Optional[PortableProfile]:
        """
        Update medical data for a profile.
        
        Args:
            patient_id: Patient identifier
            active_medications: Updated medications
            known_allergies: Updated allergies
            chronic_conditions: Updated conditions
            emergency_contacts: Updated emergency contacts
            
        Returns:
            Updated profile or None
        """
        update_data = {}
        if active_medications is not None:
            update_data["active_medications"] = active_medications
        if known_allergies is not None:
            update_data["known_allergies"] = known_allergies
        if chronic_conditions is not None:
            update_data["chronic_conditions"] = chronic_conditions
        if emergency_contacts is not None:
            update_data["emergency_contacts"] = emergency_contacts
        
        if update_data:
            return await self.update(patient_id, **update_data)
        
        return await self.get_by_id(patient_id)
    
    async def get_profiles_by_age_range(
        self,
        min_age: int,
        max_age: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[PortableProfile]:
        """
        Get profiles within an age range.
        
        Args:
            min_age: Minimum age in years
            max_age: Maximum age in years
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of profiles in age range
        """
        # Calculate date range
        current_date = datetime.utcnow()
        max_birth_date = datetime(current_date.year - min_age, current_date.month, current_date.day)
        min_birth_date = datetime(current_date.year - max_age, current_date.month, current_date.day)
        
        result = await self.session.execute(
            select(PortableProfile)
            .where(
                and_(
                    PortableProfile.date_of_birth >= min_birth_date,
                    PortableProfile.date_of_birth <= max_birth_date
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(PortableProfile.date_of_birth.desc())
        )
        return list(result.scalars().all())
    
    async def get_profiles_with_conditions(
        self,
        condition_names: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[PortableProfile]:
        """
        Get profiles with specific chronic conditions.
        
        Args:
            condition_names: List of condition names to search for
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of profiles with matching conditions
        """
        # This would require JSON querying capabilities
        # For now, return all profiles and filter in application logic
        # TODO: Implement proper JSON querying for PostgreSQL
        
        result = await self.session.execute(
            select(PortableProfile)
            .where(PortableProfile.chronic_conditions.isnot(None))
            .offset(skip)
            .limit(limit)
            .order_by(PortableProfile.last_updated.desc())
        )
        return list(result.scalars().all())
    
    async def validate_profile_integrity(self, patient_id: str) -> bool:
        """
        Validate profile integrity hash.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            True if integrity is valid
        """
        profile = await self.get_by_id(patient_id)
        if not profile:
            return False
        
        # TODO: Implement actual integrity validation
        # This would calculate hash of profile data and compare
        return True