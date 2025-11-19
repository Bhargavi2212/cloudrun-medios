"""
Bootstrap utility to prepare a hospital database with schema and seed data.
"""

from __future__ import annotations

import asyncio
import os

from database import ensure_loaded
from database.base import Base
from database.seeds import seed_demo_data
from database.session import (
    dispose_engine,
    get_engine,
    get_session_factory,
    init_engine,
)


async def bootstrap(database_url: str | None) -> None:
    """
    Apply migrations (metadata create) and seed demo content.
    """

    init_engine(database_url=database_url)
    engine = get_engine()
    ensure_loaded()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = get_session_factory()
    async with session_factory() as session:
        await seed_demo_data(session)

    await dispose_engine()


if __name__ == "__main__":
    url = os.getenv("DATABASE_URL")
    asyncio.run(bootstrap(url))
