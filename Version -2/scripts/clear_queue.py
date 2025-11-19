"""
Utility script to wipe the live queue (encounters + triage observations).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import delete

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Imports after sys.path manipulation are intentional
from database.models import (  # noqa: E402
    Encounter,
    TriageObservation,
)
from database.session import (  # noqa: E402
    get_session_factory,
    init_engine,
)


async def clear_queue() -> None:
    """
    Remove all encounter-related rows so every dashboard sees an empty queue.
    """

    init_engine()
    session_factory = get_session_factory()

    async with session_factory() as session:
        await session.execute(delete(TriageObservation))
        await session.execute(delete(Encounter))
        await session.commit()


if __name__ == "__main__":
    asyncio.run(clear_queue())
