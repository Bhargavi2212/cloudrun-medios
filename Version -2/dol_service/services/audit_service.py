"""
Audit logging utilities for DOL access events.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import AuditLog


class AuditService:
    """
    Persist audit events for federated profile access.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = AsyncCRUDRepository[AuditLog](session, AuditLog)

    async def log_access(
        self, *, patient_id: UUID, requester: str, action: str
    ) -> None:
        """
        Record an access event in the audit log.
        """

        payload = {
            "entity_type": "patient_profile",
            "entity_id": patient_id,
            "action": action,
            "performed_by": requester,
            "details": None,
        }
        await self.repository.create(payload)
        await self.session.commit()
