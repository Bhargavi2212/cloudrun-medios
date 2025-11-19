"""
FastAPI dependencies for DOL Service.

This module provides dependency injection for services and utilities
used across DOL API endpoints.
"""

from fastapi import Request, HTTPException
from typing import Optional

from .services.privacy_filter import PrivacyFilterService
from .services.crypto_service import CryptographicService
from .services.peer_registry import PeerRegistryService
from .services.audit_storage import AuditStorageService
from .config import get_settings


def get_privacy_filter(request: Request) -> PrivacyFilterService:
    """Get privacy filter service instance."""
    if not hasattr(request.app.state, 'privacy_filter'):
        settings = get_settings()
        request.app.state.privacy_filter = PrivacyFilterService(settings.hospital_id)
    
    return request.app.state.privacy_filter


def get_crypto_service(request: Request) -> CryptographicService:
    """Get cryptographic service instance."""
    if not hasattr(request.app.state, 'crypto_service'):
        settings = get_settings()
        request.app.state.crypto_service = CryptographicService(
            hospital_id=settings.hospital_id,
            private_key_path=settings.private_key_path,
            public_key_path=settings.public_key_path
        )
    
    return request.app.state.crypto_service


def get_current_hospital_id(request: Request) -> str:
    """Get current hospital ID."""
    settings = get_settings()
    return settings.hospital_id


def verify_hospital_authorization(request: Request) -> bool:
    """Verify hospital authorization for sensitive operations."""
    # TODO: Implement actual authorization logic
    # This would check JWT tokens, API keys, or mTLS certificates
    
    # For demo purposes, always return True
    return True


def get_audit_context(request: Request) -> dict:
    """Get audit context for logging."""
    return {
        "request_id": getattr(request.state, 'request_id', 'unknown'),
        "user_agent": request.headers.get("user-agent", "unknown"),
        "remote_addr": request.client.host if request.client else "unknown",
        "timestamp": "2024-11-17T00:00:00Z"
    }


def get_peer_registry(request: Request) -> PeerRegistryService:
    """Get peer registry service instance."""
    if not hasattr(request.app.state, 'peer_registry'):
        settings = get_settings()
        request.app.state.peer_registry = PeerRegistryService(settings.hospital_id)
    
    return request.app.state.peer_registry


def get_audit_storage(request: Request) -> AuditStorageService:
    """Get audit storage service instance."""
    if not hasattr(request.app.state, 'audit_storage'):
        settings = get_settings()
        request.app.state.audit_storage = AuditStorageService(settings.hospital_id)
    
    return request.app.state.audit_storage