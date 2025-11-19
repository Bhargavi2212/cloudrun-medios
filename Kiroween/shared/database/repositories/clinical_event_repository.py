"""
Repository for clinical events and timeline management.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc

from ..models import ClinicalEvent, ClinicalEventType
from .base_repository import BaseRepository


class ClinicalEventRepository(BaseRepository[ClinicalEvent]):
    """Repository for clinical event operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(ClinicalEvent, session)
    
    async def create_clinical_event(
        self,
        patient_id: str,
        event_type: ClinicalEventType,
        clinical_summary: str,
        timestamp: Optional[datetime] = None,
        structured_data: Optional[Dict[str, Any]] = None,
        ai_generated_insights: Optional[str] = None,
        confidence_score: Optional[float] = None,
        cryptographic_signature: str = "placeholder_signature",
        signing_key_fingerprint: str = "placeholder_fingerprint"
    ) -> ClinicalEvent:
        """
        Create a new clinical event.
        
        Args:
            patient_id: Patient identifier
            event_type: Type of clinical event
            clinical_summary: Clinical summary text
            timestamp: When the event occurred (defaults to now)
            structured_data: Structured clinical data
            ai_generated_insights: AI-generated insights
            confidence_score: AI confidence score (0.0-1.0)
            cryptographic_signature: Digital signature
            signing_key_fingerprint: Signing key fingerprint
            
        Returns:
            Created clinical event
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return await self.create(
            patient_id=patient_id,
            timestamp=timestamp,
            event_type=event_type,
            clinical_summary=clinical_summary,
            structured_data=structured_data or {},
            ai_generated_insights=ai_generated_insights,
            confidence_score=confidence_score,
            cryptographic_signature=cryptographic_signature,
            signing_key_fingerprint=signing_key_fingerprint
        )
    
    async def get_patient_timeline(
        self,
        patient_id: str,
        event_type: Optional[ClinicalEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ClinicalEvent]:
        """
        Get clinical timeline for a patient with filters.
        
        Args:
            patient_id: Patient identifier
            event_type: Filter by event type
            start_date: Filter events after this date
            end_date: Filter events before this date
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of clinical events in chronological order
        """
        query = select(ClinicalEvent).where(ClinicalEvent.patient_id == patient_id)
        
        # Apply filters
        if event_type:
            query = query.where(ClinicalEvent.event_type == event_type)
        if start_date:
            query = query.where(ClinicalEvent.timestamp >= start_date)
        if end_date:
            query = query.where(ClinicalEvent.timestamp <= end_date)
        
        # Order by timestamp (most recent first) and apply pagination
        query = query.order_by(desc(ClinicalEvent.timestamp)).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_events(
        self,
        patient_id: str,
        days: int = 30,
        limit: int = 10
    ) -> List[ClinicalEvent]:
        """
        Get recent clinical events for a patient.
        
        Args:
            patient_id: Patient identifier
            days: Number of days to look back
            limit: Maximum number of events
            
        Returns:
            List of recent clinical events
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(ClinicalEvent)
            .where(
                and_(
                    ClinicalEvent.patient_id == patient_id,
                    ClinicalEvent.timestamp >= cutoff_date
                )
            )
            .order_by(desc(ClinicalEvent.timestamp))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_events_by_type(
        self,
        patient_id: str,
        event_type: ClinicalEventType,
        limit: int = 50
    ) -> List[ClinicalEvent]:
        """
        Get all events of a specific type for a patient.
        
        Args:
            patient_id: Patient identifier
            event_type: Type of events to retrieve
            limit: Maximum number of events
            
        Returns:
            List of events of specified type
        """
        result = await self.session.execute(
            select(ClinicalEvent)
            .where(
                and_(
                    ClinicalEvent.patient_id == patient_id,
                    ClinicalEvent.event_type == event_type
                )
            )
            .order_by(desc(ClinicalEvent.timestamp))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_timeline_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for patient timeline.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with timeline statistics
        """
        # Get event counts by type
        result = await self.session.execute(
            select(
                ClinicalEvent.event_type,
                func.count(ClinicalEvent.event_id).label('count'),
                func.min(ClinicalEvent.timestamp).label('first_event'),
                func.max(ClinicalEvent.timestamp).label('last_event')
            )
            .where(ClinicalEvent.patient_id == patient_id)
            .group_by(ClinicalEvent.event_type)
        )
        
        event_stats = result.all()
        
        if not event_stats:
            return {
                "patient_id": patient_id,
                "total_events": 0,
                "event_counts": {},
                "first_event": None,
                "last_event": None,
                "timeline_span_days": 0
            }
        
        # Calculate summary
        total_events = sum(stat.count for stat in event_stats)
        event_counts = {stat.event_type.value: stat.count for stat in event_stats}
        
        # Get overall timeline span
        first_event = min(stat.first_event for stat in event_stats)
        last_event = max(stat.last_event for stat in event_stats)
        timeline_span = (last_event - first_event).days if first_event and last_event else 0
        
        return {
            "patient_id": patient_id,
            "total_events": total_events,
            "event_counts": event_counts,
            "first_event": first_event.isoformat() if first_event else None,
            "last_event": last_event.isoformat() if last_event else None,
            "timeline_span_days": timeline_span
        }
    
    async def search_events_by_content(
        self,
        patient_id: str,
        search_term: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[ClinicalEvent]:
        """
        Search clinical events by content.
        
        Args:
            patient_id: Patient identifier
            search_term: Term to search for in clinical summary
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of matching clinical events
        """
        result = await self.session.execute(
            select(ClinicalEvent)
            .where(
                and_(
                    ClinicalEvent.patient_id == patient_id,
                    ClinicalEvent.clinical_summary.ilike(f"%{search_term}%")
                )
            )
            .order_by(desc(ClinicalEvent.timestamp))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_events_with_ai_insights(
        self,
        patient_id: str,
        min_confidence: float = 0.7,
        limit: int = 20
    ) -> List[ClinicalEvent]:
        """
        Get events that have AI-generated insights above confidence threshold.
        
        Args:
            patient_id: Patient identifier
            min_confidence: Minimum confidence score
            limit: Maximum number of events
            
        Returns:
            List of events with high-confidence AI insights
        """
        result = await self.session.execute(
            select(ClinicalEvent)
            .where(
                and_(
                    ClinicalEvent.patient_id == patient_id,
                    ClinicalEvent.ai_generated_insights.isnot(None),
                    ClinicalEvent.confidence_score >= min_confidence
                )
            )
            .order_by(desc(ClinicalEvent.confidence_score))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def verify_event_signatures(self, patient_id: str) -> Dict[str, bool]:
        """
        Verify cryptographic signatures for all events of a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary mapping event IDs to verification status
        """
        events = await self.get_patient_timeline(patient_id, limit=1000)
        
        verification_results = {}
        for event in events:
            # TODO: Implement actual signature verification
            # For now, assume all signatures are valid
            verification_results[event.event_id] = True
        
        return verification_results