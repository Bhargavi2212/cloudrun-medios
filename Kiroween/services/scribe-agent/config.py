"""
Configuration management for Scribe Agent service.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service configuration
    hospital_id: str = os.getenv("HOSPITAL_ID", "default-hospital")
    port: int = int(os.getenv("SCRIBE_PORT", "8002"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # AI Model configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gemini-1.5-pro")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Data paths
    training_data_path: str = os.getenv("TRAINING_DATA_PATH", "../../Data/Data - Ai Scribe")
    
    # External service URLs
    manage_agent_url: str = os.getenv("MANAGE_AGENT_URL", "http://localhost:8001")
    summarizer_agent_url: str = os.getenv("SUMMARIZER_AGENT_URL", "http://localhost:8003")
    
    # Security configuration
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # Logging configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()