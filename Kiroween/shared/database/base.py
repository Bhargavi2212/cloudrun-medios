"""
Base database configuration and utilities.

This module provides the base SQLAlchemy configuration for all database models
with proper async support and medical data validation.
"""

from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase
from typing import Any


class Base(DeclarativeBase):
    """Base class for all database models with medical data conventions."""
    
    # Naming convention for constraints to ensure consistent database schema
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


# Type alias for better type hints
DatabaseModel = Any