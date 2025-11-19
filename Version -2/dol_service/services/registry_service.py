"""
Service helpers for orchestrator hospital registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import HospitalRegistry
from dol_service.schemas.registry import HospitalRegistration


class RegistryService:
    """
    Persist and retrieve hospital registry entries.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, payload: HospitalRegistration) -> HospitalRegistry:
        """
        Create or update a registry record.
        """

        entry = await self.session.get(HospitalRegistry, payload.hospital_id)
        if entry is None:
            entry = HospitalRegistry(id=payload.hospital_id)

        entry.name = payload.name
        entry.manage_url = str(payload.manage_url)
        entry.scribe_url = str(payload.scribe_url) if payload.scribe_url else None
        entry.summarizer_url = (
            str(payload.summarizer_url) if payload.summarizer_url else None
        )
        entry.dol_url = str(payload.dol_url) if payload.dol_url else None
        entry.status = "online"
        entry.capabilities = payload.capabilities or None
        entry.last_seen_at = datetime.now(tz=UTC)

        self.session.add(entry)
        await self.session.flush()
        return entry

    async def list_hospitals(self) -> Sequence[HospitalRegistry]:
        """
        Return the known hospitals ordered by last heartbeat.
        """

        stmt = select(HospitalRegistry).order_by(HospitalRegistry.name.asc())
        result = await self.session.execute(stmt)
        return result.scalars().all()
