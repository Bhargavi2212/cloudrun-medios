"""Database engine and session management."""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..services.config import get_settings

# Check if we're in test mode
TESTING = os.getenv("TESTING", "").lower() == "true"

if TESTING:
    # Use SQLite in-memory for tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    # Use production database from settings
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
        pool_recycle=settings.database_pool_recycle_seconds,
        echo=settings.database_echo,
    )

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


@contextlib.contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Iterator[Session]:
    with get_session() as session:
        yield session
