"""
Configuration helpers for Medi OS backend services.

Important environment variables:
    APP_ENV                     Application environment ("development", "staging", "production")
    APP_DEBUG                   Enable debug mode ("true"/"false", default false)

    HF_TOKEN                    HuggingFace API token for authenticated downloads (optional)
    GEMINI_API_KEY              Google Generative AI key (required for live note generation)
    MODEL_SIZE                  Whisper model size (tiny, base, small, medium, large)
    DATA_PATH                   Directory for uploaded audio files
    WHISPER_CACHE_DIR           Directory to cache whisper weights (default: ./models)
    GEMINI_MODEL                Gemini model name (default: gemini-2.0-flash-exp)
    GEMINI_TEMPERATURE          Sampling temperature (default: 0.2)
    GEMINI_MAX_TOKENS           Max output tokens for note generation (default: 1000)

    DATABASE_URL                SQLAlchemy URL for PostgreSQL (e.g. postgresql+psycopg2://user:pass@host/db)
    DATABASE_POOL_SIZE          Primary connection pool size (default 5)
    DATABASE_MAX_OVERFLOW       Additional connections beyond pool size (default 10)
    DATABASE_POOL_RECYCLE       Seconds before recycling connections (default 1800)
    DATABASE_ECHO               Enable SQL echo logging ("true"/"false", default false)

    JWT_ACCESS_SECRET           Secret used to sign access tokens (required)
    JWT_REFRESH_SECRET          Secret used to sign refresh tokens (required)
    JWT_ALGORITHM               Signing algorithm (default HS256)
    JWT_ACCESS_EXPIRES_MIN      Minutes before access token expiry (default 15)
    JWT_REFRESH_EXPIRES_MIN     Minutes before refresh token expiry (default 43200 ~30 days)

    STORAGE_BACKEND             Storage backend to use ("local", "gcs")
    STORAGE_LOCAL_PATH          Directory for local storage backend (default: ./storage)
    STORAGE_GCS_BUCKET          Google Cloud Storage bucket name (when using GCS backend)
    STORAGE_GCS_CREDENTIALS     Path to GCP service account JSON (optional if using ADC)
    STORAGE_SIGNED_URL_EXPIRY   Signed URL expiry in seconds (default: 3600)

    SECRET_MANAGER_ENABLED      Enable Cloud Secret Manager integration ("true"/"false", default false)
    SECRET_MANAGER_PROJECT_ID   GCP project ID for Secret Manager (required if SECRET_MANAGER_ENABLED=true)
    SECRET_MANAGER_ENVIRONMENT  Environment suffix for secrets (dev/staging/prod, optional)
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Try to import Google Cloud Secret Manager (optional dependency)
try:
    from google.cloud import secretmanager  # type: ignore
except ImportError:
    secretmanager = None  # pragma: no cover

BOOL_TRUE = {"1", "true", "on", "yes"}

logger = logging.getLogger(__name__)


def _json_or_raw(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class ScribeSettings(BaseSettings):
    """Global settings for Medi OS services.

    Supports environment tiers (development, staging, production) and
    optional Cloud Secret Manager integration for secure secret management.
    """

    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")

    # Secret Manager configuration
    secret_manager_enabled: bool = Field(default=False, alias="SECRET_MANAGER_ENABLED")
    secret_manager_project_id: Optional[str] = Field(
        default=None, alias="SECRET_MANAGER_PROJECT_ID"
    )
    secret_manager_environment: Optional[str] = Field(
        default=None, alias="SECRET_MANAGER_ENVIRONMENT"
    )

    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    model_size: str = Field(default="base", alias="MODEL_SIZE")
    data_path: Path = Field(default=Path("./uploads"), alias="DATA_PATH")
    whisper_cache_dir: Path = Field(default=Path("./models"), alias="WHISPER_CACHE_DIR")
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp", alias="GEMINI_MODEL"
    )  # Use latest model (gemini-1.5-pro and gemini-1.5-flash were deprecated)
    gemini_temperature: float = Field(default=0.2, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=1000, alias="GEMINI_MAX_TOKENS")

    summarizer_enabled: bool = Field(default=True, alias="SUMMARIZER_ENABLED")
    summarizer_data_glob: Optional[str] = Field(
        default=None, alias="SUMMARIZER_DATA_GLOB"
    )
    summarizer_codes_path: Optional[Path] = Field(
        default=None, alias="SUMMARIZER_CODES_PATH"
    )
    summarizer_max_cache_entries: int = Field(
        default=32, alias="SUMMARIZER_MAX_CACHE_ENTRIES"
    )
    summarizer_stop_gap_days: int = Field(default=90, alias="SUMMARIZER_STOP_GAP_DAYS")
    summarizer_slow_threshold: float = Field(
        default=5.0, alias="SUMMARIZER_SLOW_THRESHOLD"
    )
    summarizer_temperature: float = Field(default=0.2, alias="SUMMARIZER_TEMPERATURE")
    summarizer_max_tokens: int = Field(default=3000, alias="SUMMARIZER_MAX_TOKENS")
    summarizer_use_fake_llm: bool = Field(
        default=False, alias="SUMMARIZER_USE_FAKE_LLM"
    )
    summarizer_cache_ttl_minutes: int = Field(
        default=360, alias="SUMMARIZER_CACHE_TTL_MINUTES"
    )

    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/medios",
        alias="DATABASE_URL",
    )
    database_pool_size: int = Field(default=5, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, alias="DATABASE_MAX_OVERFLOW")
    database_pool_recycle_seconds: int = Field(
        default=1800, alias="DATABASE_POOL_RECYCLE"
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")

    jwt_access_secret: str = Field(
        default="change-me-access", alias="JWT_ACCESS_SECRET"
    )
    jwt_refresh_secret: str = Field(
        default="change-me-refresh", alias="JWT_REFRESH_SECRET"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_expires_minutes: int = Field(default=15, alias="JWT_ACCESS_EXPIRES_MIN")
    jwt_refresh_expires_minutes: int = Field(
        default=60 * 24 * 30, alias="JWT_REFRESH_EXPIRES_MIN"
    )

    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")
    storage_local_path: Path = Field(
        default=Path("./storage"), alias="STORAGE_LOCAL_PATH"
    )
    storage_gcs_bucket: Optional[str] = Field(default=None, alias="STORAGE_GCS_BUCKET")
    storage_gcs_credentials: Optional[Path] = Field(
        default=None, alias="STORAGE_GCS_CREDENTIALS"
    )
    storage_signed_url_expiry_seconds: int = Field(
        default=3600, alias="STORAGE_SIGNED_URL_EXPIRY"
    )

    # Retention policy configuration (in days)
    storage_retention_default_days: int = Field(
        default=365, alias="STORAGE_RETENTION_DEFAULT_DAYS"
    )
    storage_retention_audio_days: int = Field(
        default=365, alias="STORAGE_RETENTION_AUDIO_DAYS"
    )
    storage_retention_document_days: int = Field(
        default=2555, alias="STORAGE_RETENTION_DOCUMENT_DAYS"
    )  # 7 years for medical records
    storage_retention_enabled: bool = Field(
        default=True, alias="STORAGE_RETENTION_ENABLED"
    )
    storage_retention_cleanup_interval_hours: int = Field(
        default=24, alias="STORAGE_RETENTION_CLEANUP_INTERVAL_HOURS"
    )

    rate_limit_enabled: bool = Field(default=False, alias="RATE_LIMIT_ENABLED")
    rate_limit_default_per_minute: int = Field(
        default=120, alias="RATE_LIMIT_DEFAULT_PER_MINUTE"
    )
    rate_limit_burst_multiplier: float = Field(
        default=3.0, alias="RATE_LIMIT_BURST_MULTIPLIER"
    )

    feature_flags_raw: Optional[str] = Field(default=None, alias="FEATURE_FLAGS")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, exclude=True)

    triage_model_dir: Path = Field(
        default=Path("medi-os/services/manage-agent/models"),
        alias="TRIAGE_MODEL_DIR",
    )
    triage_metadata_file: str = Field(
        default="xgboost_lightgbm_metadata.pkl", alias="TRIAGE_METADATA_FILE"
    )
    triage_baseline_metadata_file: str = Field(
        default="baseline_metadata.pkl", alias="TRIAGE_BASELINE_METADATA_FILE"
    )
    triage_stacking_model: str = Field(
        default="final_stacking_ensemble.pkl", alias="TRIAGE_STACKING_MODEL"
    )
    triage_xgb_model: str = Field(
        default="final_xgboost_full_features.pkl", alias="TRIAGE_XGB_MODEL"
    )
    triage_lgbm_model: str = Field(
        default="final_lightgbm_full_features.pkl", alias="TRIAGE_LGBM_MODEL"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
        settings_json_loads=_json_or_raw,
    )

    @field_validator(
        "data_path",
        "whisper_cache_dir",
        "storage_local_path",
        "storage_gcs_credentials",
        "summarizer_codes_path",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value: Optional[str | Path]) -> Optional[Path]:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        return Path(value).expanduser()

    @field_validator("summarizer_enabled", "summarizer_use_fake_llm", mode="before")
    @classmethod
    def _coerce_bool_fields(cls, value):
        return cls._coerce_bool(value)

    @field_validator("model_size", mode="before")
    @classmethod
    def _validate_model_size(cls, value: str) -> str:
        allowed = {"tiny", "base", "small", "medium", "large", "large-v1"}
        if value not in allowed:
            return "base"
        return value

    @field_validator("storage_backend", mode="before")
    @classmethod
    def _normalize_storage_backend(cls, value: str) -> str:
        return (value or "local").strip().lower()

    @field_validator("app_env", mode="before")
    @classmethod
    def _validate_app_env(cls, value: str) -> str:
        """Validate and normalize environment name."""
        if value is None:
            return "development"
        env = str(value).strip().lower()
        valid_envs = {"development", "dev", "staging", "stage", "production", "prod"}
        if env not in valid_envs:
            logger.warning(f"Invalid APP_ENV '{value}', defaulting to 'development'")
            return "development"
        # Normalize to standard names
        if env in ("dev", "development"):
            return "development"
        if env in ("stage", "staging"):
            return "staging"
        if env in ("prod", "production"):
            return "production"
        return env

    @field_validator("secret_manager_enabled", mode="before")
    @classmethod
    def _coerce_secret_manager_enabled(cls, value) -> bool:
        return cls._coerce_bool(value)

    @field_validator("storage_retention_enabled", mode="before")
    @classmethod
    def _coerce_storage_retention_enabled(cls, value) -> bool:
        return cls._coerce_bool(value)

    @model_validator(mode="after")
    def _ensure_directories(self) -> "ScribeSettings":
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.whisper_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.storage_backend == "local":
            self.storage_local_path.mkdir(parents=True, exist_ok=True)
        self.feature_flags = self._parse_feature_flags(self.feature_flags_raw)
        self.triage_model_dir = self.triage_model_dir.expanduser()

        # Set secret manager environment from app_env if not explicitly set
        if self.secret_manager_enabled and not self.secret_manager_environment:
            self.secret_manager_environment = self.app_env

        # Validate Secret Manager configuration
        if self.secret_manager_enabled:
            if not self.secret_manager_project_id:
                logger.warning(
                    "SECRET_MANAGER_ENABLED is true but SECRET_MANAGER_PROJECT_ID is not set. "
                    "Secret Manager will be disabled."
                )
                self.secret_manager_enabled = False
            elif secretmanager is None:
                logger.warning(
                    "SECRET_MANAGER_ENABLED is true but google-cloud-secret-manager is not installed. "
                    "Install it with: pip install google-cloud-secret-manager"
                )
                self.secret_manager_enabled = False

        return self

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        value_str = str(value).strip().lower()
        if value_str in BOOL_TRUE:
            return True
        return False

    @classmethod
    def _parse_feature_flags(cls, value) -> Dict[str, bool]:
        if value in (None, "", {}, []):
            return {}
        if isinstance(value, dict):
            return {str(k): cls._coerce_bool(v) for k, v in value.items()}
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return {str(k): cls._coerce_bool(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass
            flags: Dict[str, bool] = {}
            for item in value.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    key, raw = item.split("=", 1)
                    flags[key.strip()] = cls._coerce_bool(raw.strip())
                else:
                    flags[item] = True
            return flags
        raise ValueError("Unsupported feature flag format")


@lru_cache(maxsize=1)
def get_settings() -> ScribeSettings:
    """Return cached scribe settings instance."""
    return ScribeSettings()


class SecretManager:
    """Cloud Secret Manager integration for secure secret retrieval."""

    def __init__(self, project_id: str, environment: Optional[str] = None):
        """Initialize Secret Manager client.

        Args:
            project_id: GCP project ID
            environment: Optional environment suffix (dev/staging/prod)
        """
        if secretmanager is None:
            raise ImportError(
                "google-cloud-secret-manager is not installed. "
                "Install it with: pip install google-cloud-secret-manager"
            )
        self.project_id = project_id
        self.environment = environment
        self.client = secretmanager.SecretManagerServiceClient()
        self._cache: Dict[str, str] = {}

    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """Get a secret from Cloud Secret Manager.

        Args:
            secret_name: Name of the secret (without environment suffix)
            version: Secret version (default: "latest")

        Returns:
            Secret value as string, or None if not found

        Examples:
            >>> sm = SecretManager(project_id="my-project", environment="prod")
            >>> api_key = sm.get_secret("gemini-api-key")
            # Fetches secret: projects/my-project/secrets/gemini-api-key-prod/versions/latest
        """
        # Build secret name with environment suffix if provided
        if self.environment:
            full_secret_name = f"{secret_name}-{self.environment}"
        else:
            full_secret_name = secret_name

        # Check cache first
        cache_key = f"{full_secret_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build secret resource name
        name = (
            f"projects/{self.project_id}/secrets/{full_secret_name}/versions/{version}"
        )

        try:
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            # Cache the secret
            self._cache[cache_key] = secret_value
            logger.debug(
                f"Retrieved secret '{full_secret_name}' from Cloud Secret Manager"
            )
            return secret_value
        except Exception as exc:
            logger.warning(
                f"Failed to retrieve secret '{full_secret_name}' from Cloud Secret Manager: {exc}"
            )
            return None


# Global Secret Manager instance (lazy initialization)
_secret_manager_instance: Optional[SecretManager] = None


def get_secret_manager() -> Optional[SecretManager]:
    """Get or create Secret Manager instance.

    Returns:
        SecretManager instance if enabled and configured, None otherwise
    """
    global _secret_manager_instance

    settings = get_settings()
    if not settings.secret_manager_enabled or not settings.secret_manager_project_id:
        return None

    if _secret_manager_instance is None:
        try:
            _secret_manager_instance = SecretManager(
                project_id=settings.secret_manager_project_id,
                environment=settings.secret_manager_environment,
            )
            logger.info(
                f"Initialized Secret Manager for project '{settings.secret_manager_project_id}' "
                f"(environment: {settings.secret_manager_environment or 'default'})"
            )
        except Exception as exc:
            logger.error(f"Failed to initialize Secret Manager: {exc}")
            return None

    return _secret_manager_instance


def get_secret(
    secret_name: str, env_var: Optional[str] = None, version: str = "latest"
) -> Optional[str]:
    """Get secret from Cloud Secret Manager or environment variable.

    This is a convenience function that:
    1. Tries to get the secret from Cloud Secret Manager (if enabled)
    2. Falls back to environment variable (if provided)
    3. Returns None if neither is available

    Args:
        secret_name: Name of the secret in Secret Manager
        env_var: Optional environment variable name as fallback
        version: Secret version (default: "latest")

    Returns:
        Secret value, or None if not found

    Examples:
        >>> # Get secret from Secret Manager or GEMINI_API_KEY env var
        >>> api_key = get_secret("gemini-api-key", env_var="GEMINI_API_KEY")

        >>> # Get secret from Secret Manager only
        >>> db_password = get_secret("database-password")
    """
    settings = get_settings()

    # Try Secret Manager first if enabled
    if settings.secret_manager_enabled and settings.secret_manager_project_id:
        sm = get_secret_manager()
        if sm:
            secret_value = sm.get_secret(secret_name, version=version)
            if secret_value:
                return secret_value

    # Fallback to environment variable if provided
    if env_var:
        return os.getenv(env_var)

    return None


__all__ = [
    "ScribeSettings",
    "get_settings",
    "SecretManager",
    "get_secret_manager",
    "get_secret",
]
