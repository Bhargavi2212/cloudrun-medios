"""
CLI entry point to seed the database with demo data.
"""

from __future__ import annotations

import logging
import sys

from database import ensure_loaded
from database.base import Base
from database.seeds import seed_demo_data
from database.session import (
    dispose_engine,
    get_engine,
    get_session_factory,
    init_engine,
    run_sync,
)
from shared.exceptions import ConfigurationError
from shared.logging import configure_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Seed database with synthetic demo data.
    """

    try:
        init_engine()
    except ConfigurationError as exc:
        logger.error("❌ DATABASE_URL is not configured: %s", exc)
        raise

    engine = get_engine()
    ensure_loaded()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = get_session_factory()
    async with session_factory() as session:
        await seed_demo_data(session)

    await dispose_engine()


if __name__ == "__main__":
    configure_logging("seed")
    try:
        run_sync(main())
    except ConfigurationError:
        sys.exit(1)
