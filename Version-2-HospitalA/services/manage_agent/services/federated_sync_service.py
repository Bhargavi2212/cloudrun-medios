"""
Utilities to synchronize local patient data with the orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dol_service.core.privacy_filter import redact_metadata, sanitize_timeline
from dol_service.services.profile_service import ProfileService
from services.manage_agent.services.orchestrator_client import OrchestratorClient

logger = logging.getLogger(__name__)


class FederatedSyncService:
    """
    Builds sanitized patient snapshots and pushes them to the orchestrator.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        hospital_id: str,
        client: OrchestratorClient | None,
    ) -> None:
        self._session_factory = session_factory
        self._hospital_id = hospital_id
        self._client = client

    async def sync_patient(self, patient_id: UUID) -> None:
        """
        Build a snapshot for the patient and push it to the orchestrator.
        """

        if self._client is None:
            return

        async with self._session_factory() as session:
            profile_service = ProfileService(session, self._hospital_id)
            profile = await profile_service.build_profile(patient_id)
            if profile is None:
                logger.debug(
                    "No local data found for patient %s; skipping federation sync.",
                    patient_id,
                )
                return

        snapshot = self._build_snapshot(profile)
        await self._client.push_snapshot(patient_id, snapshot)

    def _build_snapshot(self, profile: dict[str, Any]) -> dict[str, Any]:
        """
        Convert the local profile into the orchestrator snapshot schema.
        """

        sanitized_patient = redact_metadata(profile["patient"])
        sanitized_timeline = sanitize_timeline(profile["timeline"])
        sanitized_summaries = [
            redact_metadata(summary) for summary in profile["summaries"]
        ]

        patient_block = {
            "patient_id": sanitized_patient.get("id"),
            "mrn": sanitized_patient.get("mrn"),
            "first_name": sanitized_patient.get("first_name"),
            "last_name": sanitized_patient.get("last_name"),
            "dob": sanitized_patient.get("dob"),
            "sex": sanitized_patient.get("sex"),
            "contact_info": sanitized_patient.get("contact_info"),
        }

        timeline_payload = [
            {
                "external_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "event_timestamp": event.get("timestamp"),
                "encounter_id": event.get("encounter_id"),
                "summary_id": event.get("summary_id"),
                "content": event.get("content", {}),
            }
            for event in sanitized_timeline
        ]

        return {
            "patient": patient_block,
            "summaries": sanitized_summaries,
            "timeline": timeline_payload,
        }
