"""
Configuration management for Manage Agent service.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service configuration
    hospital_id: str = os.getenv("HOSPITAL_ID", "default-hospital")
    port: int = int(os.getenv("PORT", "8001"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Database configuration
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://medi_user:medi_password@localhost:5432/medi_os"
    )
    
    # Security configuration
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # External service URLs
    scribe_agent_url: str = os.getenv("SCRIBE_AGENT_URL", "http://localhost:8002")
    summarizer_agent_url: str = os.getenv("SUMMARIZER_AGENT_URL", "http://localhost:8003")
    dol_service_url: str = os.getenv("DOL_SERVICE_URL", "http://localhost:8004")
    
    # Logging configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()