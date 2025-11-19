"""
Configuration management for Summarizer Agent service.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service configuration
    hospital_id: str = os.getenv("HOSPITAL_ID", "default-hospital")
    port: int = int(os.getenv("SUMMARIZER_PORT", "8003"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # AI Model configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gemini-1.5-pro")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))  # Lower for more consistent summaries
    
    # Summarization settings
    max_timeline_events: int = int(os.getenv("MAX_TIMELINE_EVENTS", "50"))
    summary_length: str = os.getenv("SUMMARY_LENGTH", "medium")  # short, medium, long
    
    # Data paths
    training_data_path: str = os.getenv("TRAINING_DATA_PATH", "../../Data/Data - Ai summarizer")
    
    # External service URLs
    manage_agent_url: str = os.getenv("MANAGE_AGENT_URL", "http://localhost:8001")
    scribe_agent_url: str = os.getenv("SCRIBE_AGENT_URL", "http://localhost:8002")
    
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