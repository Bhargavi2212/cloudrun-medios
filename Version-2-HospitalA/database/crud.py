"""
Generic CRUD repositories for asynchronous SQLAlchemy usage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

TModel = TypeVar("TModel")


class AsyncCRUDRepository(Generic[TModel]):
    """
    Base repository implementing common CRUD patterns.
    """

    def __init__(self, session: AsyncSession, model: type[TModel]) -> None:
        self.session = session
        self.model = model

    async def get(self, identifier: Any) -> TModel | None:
        """
        Retrieve an instance by primary key.
        """

        return await self.session.get(self.model, identifier)

    async def list(self, *, limit: int = 100, offset: int = 0) -> Sequence[TModel]:
        """
        Return a paginated list of instances.
        """

        stmt = select(self.model).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create(self, data: dict[str, Any]) -> TModel:
        """
        Persist a new instance using the provided data mapping.
        """

        instance = self.model(**data)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update(self, instance: TModel, data: dict[str, Any]) -> TModel:
        """
        Apply partial updates to an instance.
        """

        for field, value in data.items():
            setattr(instance, field, value)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, instance: TModel) -> None:
        """
        Delete an instance.
        """

        await self.session.delete(instance)
        await self.session.flush()
