"""
Configuration for DOL Service.

This module manages configuration settings for the Data Orchestration Layer,
including hospital identification, security settings, and federated learning parameters.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os


class DOLSettings(BaseSettings):
    """DOL Service configuration settings."""
    
    # Hospital identification
    hospital_id: str = Field(
        default="hospital_001",
        description="Unique identifier for this hospital"
    )
    
    hospital_name: str = Field(
        default="General Hospital",
        description="Human-readable hospital name (for internal use only)"
    )
    
    # Service configuration
    port: int = Field(default=8003, description="Port for DOL service")
    debug: bool = Field(default=True, description="Enable debug mode")
    
    # Database configuration
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/medi_os",
        description="Database connection URL"
    )
    
    # Cryptographic settings
    private_key_path: str = Field(
        default="./keys/hospital_private.pem",
        description="Path to hospital's private key for signing"
    )
    
    public_key_path: str = Field(
        default="./keys/hospital_public.pem", 
        description="Path to hospital's public key"
    )
    
    # Privacy settings
    privacy_level: str = Field(
        default="standard",
        description="Privacy level: minimal, standard, maximum"
    )
    
    enable_differential_privacy: bool = Field(
        default=True,
        description="Enable differential privacy for federated learning"
    )
    
    # Federated learning settings
    federated_coordinator_url: Optional[str] = Field(
        default=None,
        description="URL of federated learning coordinator"
    )
    
    enable_federated_learning: bool = Field(
        default=True,
        description="Enable federated learning participation"
    )
    
    # Peer hospital registry
    peer_hospitals: List[str] = Field(
        default_factory=list,
        description="List of trusted peer hospital IDs"
    )
    
    # Security settings
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key for authentication"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    jwt_expiration_hours: int = Field(
        default=24,
        description="JWT token expiration time in hours"
    )
    
    # Audit settings
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging for compliance"
    )
    
    audit_log_path: str = Field(
        default="./logs/dol_audit.log",
        description="Path to audit log file"
    )
    
    # Profile export settings
    max_profile_size_mb: int = Field(
        default=50,
        description="Maximum portable profile size in MB"
    )
    
    supported_export_formats: List[str] = Field(
        default_factory=lambda: ["json", "fhir", "qr_code"],
        description="Supported profile export formats"
    )
    
    # Timeline settings
    max_timeline_entries: int = Field(
        default=10000,
        description="Maximum timeline entries per patient"
    )
    
    timeline_retention_days: int = Field(
        default=365 * 10,  # 10 years
        description="Timeline data retention period in days"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "DOL_"


# Global settings instance
_settings: Optional[DOLSettings] = None


def get_settings() -> DOLSettings:
    """Get DOL service settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = DOLSettings()
    return _settings


def reload_settings() -> DOLSettings:
    """Reload settings from environment."""
    global _settings
    _settings = DOLSettings()
    return _settings