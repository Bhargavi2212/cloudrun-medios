"""
Manage Agent Service - Patient Profile Management

This service handles patient profile import/export, timeline management,
and core patient data operations for hospital systems.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from shared.database import DatabaseManager, get_db_session
from .config import get_settings
from .routers import profiles, timeline, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global database manager
db_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global db_manager
    
    settings = get_settings()
    
    # Initialize database
    db_manager = DatabaseManager(
        database_url=settings.database_url,
        hospital_id=settings.hospital_id
    )
    
    try:
        await db_manager.initialize()
        await db_manager.create_tables()
        logger.info(f"Manage Agent initialized for {settings.hospital_id}")
        yield
    finally:
        if db_manager:
            await db_manager.close()
            logger.info("Manage Agent shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Medi OS - Manage Agent",
    description="Patient profile management service for portable medical records",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
app.include_router(timeline.router, prefix="/api/timeline", tags=["timeline"])


@app.get("/")
async def root():
    """Root endpoint with service information."""
    settings = get_settings()
    return {
        "service": "Medi OS Manage Agent",
        "version": "2.0.0",
        "hospital_id": settings.hospital_id,
        "status": "operational",
        "description": "Patient profile management service"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "service": "manage-agent"
        }
    )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )