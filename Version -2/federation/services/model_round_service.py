"""
Persistence helper for aggregated model rounds.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import FederatedModelRound
from federation.schemas import GlobalModel


class ModelRoundService:
    """
    Store and retrieve aggregated model rounds for telemetry purposes.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_round(self, payload: GlobalModel) -> FederatedModelRound:
        """
        Upsert the aggregated round for the provided model name + round id.
        """

        stmt = select(FederatedModelRound).where(
            FederatedModelRound.model_name == payload.model_name,
            FederatedModelRound.round_id == payload.round_id,
        )
        result = await self._session.execute(stmt)
        record: FederatedModelRound | None = result.scalar_one_or_none()
        if record is None:
            record = FederatedModelRound(
                model_name=payload.model_name,
                round_id=payload.round_id,
                weights=payload.weights,
                contributor_count=payload.contributor_count,
                round_metadata=payload.metadata,
            )
            self._session.add(record)
        else:
            record.weights = payload.weights
            record.contributor_count = payload.contributor_count
            record.round_metadata = payload.metadata
        return record

    async def latest(self, model_name: str) -> FederatedModelRound | None:
        """
        Retrieve the most recent round for the requested model.
        """

        stmt = (
            select(FederatedModelRound)
            .where(FederatedModelRound.model_name == model_name)
            .order_by(FederatedModelRound.round_id.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_recent(
        self, model_name: str, *, limit: int = 10
    ) -> Sequence[FederatedModelRound]:
        """
        Return the latest N rounds for telemetry dashboards.
        """

        stmt = (
            select(FederatedModelRound)
            .where(FederatedModelRound.model_name == model_name)
            .order_by(FederatedModelRound.round_id.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()
