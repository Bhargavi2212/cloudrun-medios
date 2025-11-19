"""
Scribe Agent Service - Clinical Documentation

This service handles clinical documentation, SOAP note generation,
and medical dialogue processing using AI models.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import get_settings
from .routers import health, documentation, dialogue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    logger.info(f"Scribe Agent initialized for {settings.hospital_id}")
    yield
    logger.info("Scribe Agent shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Medi OS - Scribe Agent",
    description="Clinical documentation and SOAP note generation service",
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
app.include_router(documentation.router, prefix="/api/documentation", tags=["documentation"])
app.include_router(dialogue.router, prefix="/api/dialogue", tags=["dialogue"])


@app.get("/")
async def root():
    """Root endpoint with service information."""
    settings = get_settings()
    return {
        "service": "Medi OS Scribe Agent",
        "version": "2.0.0",
        "hospital_id": settings.hospital_id,
        "status": "operational",
        "description": "Clinical documentation and SOAP note generation service",
        "capabilities": [
            "SOAP note generation",
            "Medical dialogue processing",
            "Clinical documentation",
            "Voice-to-text transcription"
        ]
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "service": "scribe-agent"
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