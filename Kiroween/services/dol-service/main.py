"""
Data Orchestration Layer (DOL) Service for Medi OS Kiroween Edition v2.0.

This service handles federated patient profile management, privacy filtering,
and secure communication between hospitals while maintaining patient data sovereignty.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from .config import get_settings
from .routers import federated_patient, timeline, model_update, peer_registry, peer_registry
from .middleware.auth import AuthMiddleware
from .middleware.audit import AuditMiddleware
from .services.privacy_filter import PrivacyFilterService
from .services.crypto_service import CryptographicService
from .services.peer_registry import PeerRegistryService
from .services.audit_storage import AuditStorageService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    # Initialize services
    logger.info(f"Starting DOL Service for hospital: {settings.hospital_id}")
    logger.info("Initializing privacy filtering and cryptographic services...")
    
    # Initialize privacy filter service
    app.state.privacy_filter = PrivacyFilterService(settings.hospital_id)
    
    # Initialize cryptographic service
    app.state.crypto_service = CryptographicService(
        hospital_id=settings.hospital_id,
        private_key_path=settings.private_key_path,
        public_key_path=settings.public_key_path
    )
    
    # Initialize peer registry service
    app.state.peer_registry = PeerRegistryService(settings.hospital_id)
    
    # Initialize audit storage service
    app.state.audit_storage = AuditStorageService(settings.hospital_id)
    
    logger.info("DOL Service initialization complete")
    
    yield
    
    # Cleanup
    logger.info("Shutting down DOL Service")


# Create FastAPI app
app = FastAPI(
    title="Medi OS DOL Service",
    description="Data Orchestration Layer for privacy-preserving federated patient profiles",
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

# Add custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(AuditMiddleware)

# Include routers
app.include_router(
    federated_patient.router,
    prefix="/api/federated/patient",
    tags=["Federated Patient Profiles"]
)

app.include_router(
    timeline.router,
    prefix="/api/timeline",
    tags=["Clinical Timeline"]
)

app.include_router(
    model_update.router,
    prefix="/api/model_update",
    tags=["Federated Learning"]
)

app.include_router(
    peer_registry.router,
    prefix="/api/peer_registry",
    tags=["Peer Registry"]
)

app.include_router(
    peer_registry.router,
    prefix="/api/registry",
    tags=["Peer Registry & Audit"]
)

app.include_router(
    peer_registry.router,
    prefix="/api/peer_registry",
    tags=["Peer Registry"]
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "dol-service",
        "version": "2.0.0",
        "hospital_id": settings.hospital_id,
        "timestamp": "2024-11-17T00:00:00Z"
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    settings = get_settings()
    return {
        "service": "Medi OS Data Orchestration Layer",
        "version": "2.0.0",
        "description": "Privacy-preserving federated patient profile management",
        "hospital_id": settings.hospital_id,
        "features": [
            "Federated patient profile import/export",
            "Privacy-first data filtering",
            "Cryptographic profile integrity",
            "Secure hospital-to-hospital communication",
            "Federated learning coordination"
        ]
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error in DOL service: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )