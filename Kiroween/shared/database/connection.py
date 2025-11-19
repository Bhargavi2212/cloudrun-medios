"""
Database connection management for multi-hospital deployment.

This module provides async database connection management with proper
connection pooling and health checks for production deployment.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from .base import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages async database connections for hospital services.
    
    Provides connection pooling, health checks, and proper async session management
    for production deployment across multiple hospitals.
    """
    
    def __init__(self, database_url: str, hospital_id: str):
        """
        Initialize database manager for a specific hospital.
        
        Args:
            database_url: PostgreSQL connection URL
            hospital_id: Unique identifier for this hospital (e.g., 'hospital-a')
            
        Raises:
            ValueError: If database_url is invalid or hospital_id is empty
        """
        if not database_url:
            raise ValueError("Database URL cannot be empty")
        if not hospital_id:
            raise ValueError("Hospital ID cannot be empty")
            
        self.database_url = database_url
        self.hospital_id = hospital_id
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        
        logger.info(f"🏥 Initializing database manager for {hospital_id}")
        
    async def initialize(self) -> None:
        """
        Initialize database engine and session factory.
        
        Creates async engine with proper connection pooling for production use.
        """
        try:
            # Parse database URL to validate format
            parsed_url = urlparse(self.database_url)
            if parsed_url.scheme not in ['postgresql', 'postgresql+asyncpg']:
                # Convert to async URL if needed
                if parsed_url.scheme == 'postgresql':
                    self.database_url = self.database_url.replace(
                        'postgresql://', 'postgresql+asyncpg://', 1
                    )
            
            # Create async engine with production-ready settings
            self._engine = create_async_engine(
                self.database_url,
                # Connection pool settings for production
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                # Echo SQL in development only
                echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
                # Async-specific settings
                future=True,
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            logger.info(f"✅ Database engine initialized for {self.hospital_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database for {self.hospital_id}: {e}")
            raise
            
    async def create_tables(self) -> None:
        """
        Create all database tables if they don't exist.
        
        This is used for initial setup and testing. In production,
        use Alembic migrations instead.
        """
        if not self._engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info(f"✅ Database tables created for {self.hospital_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables for {self.hospital_id}: {e}")
            raise
            
    async def health_check(self) -> bool:
        """
        Perform database health check.
        
        Returns:
            True if database is healthy, False otherwise
        """
        if not self._engine:
            return False
            
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
                
        except Exception as e:
            logger.warning(f"⚠️ Database health check failed for {self.hospital_id}: {e}")
            return False
            
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with proper error handling.
        
        Yields:
            AsyncSession: Database session for queries
            
        Raises:
            RuntimeError: If database not initialized
        """
        if not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
                
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self._engine:
            await self._engine.dispose()
            logger.info(f"🔒 Database connections closed for {self.hospital_id}")


def get_database_url(hospital_id: str) -> str:
    """
    Get database URL for a specific hospital.
    
    Args:
        hospital_id: Hospital identifier (e.g., 'hospital-a')
        
    Returns:
        Database URL for the hospital
        
    Raises:
        ValueError: If hospital_id is invalid or DATABASE_URL not configured
    """
    if not hospital_id:
        raise ValueError("Hospital ID cannot be empty")
        
    # Try hospital-specific URL first
    hospital_url_key = f"DATABASE_URL_{hospital_id.upper().replace('-', '_')}"
    database_url = os.getenv(hospital_url_key)
    
    if not database_url:
        # Fall back to generic DATABASE_URL
        database_url = os.getenv("DATABASE_URL")
        
    if not database_url:
        raise ValueError(
            f"Database URL not configured. Set {hospital_url_key} or DATABASE_URL"
        )
        
    return database_url


# Global database manager instance (set by each service)
db_manager: Optional[DatabaseManager] = None


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database session.
    
    Yields:
        AsyncSession: Database session for request handling
    """
    if not db_manager:
        raise RuntimeError("Database manager not initialized")
        
    async with db_manager.get_session() as session:
        yield session