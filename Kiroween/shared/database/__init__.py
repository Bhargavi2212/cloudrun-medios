"""
Database models and utilities for portable patient profiles.

This module provides:
- Async SQLAlchemy models for portable patient profiles
- Database connection management
- CRUD repositories for data operations
- Alembic migration utilities
- Privacy-preserving data structures
"""

from .models import (
    PortableProfile,
    ClinicalEvent,
    LocalPatientRecord,
    ProfileSignature,
    ClinicalEventType,
    ProfileSyncStatus,
)
from .connection import DatabaseManager, get_database_url, get_db_session
from .base import Base
from .repositories import (
    ProfileRepository,
    ClinicalEventRepository,
    LocalRecordRepository,
)

__all__ = [
    "PortableProfile",
    "ClinicalEvent", 
    "LocalPatientRecord",
    "ProfileSignature",
    "ClinicalEventType",
    "ProfileSyncStatus",
    "DatabaseManager",
    "get_database_url",
    "get_db_session",
    "Base",
    "ProfileRepository",
    "ClinicalEventRepository",
    "LocalRecordRepository",
]