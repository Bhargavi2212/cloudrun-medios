"""
Model management helpers for the AI Scribe pipeline.

Responsible for downloading/caching Whisper weights on-demand and exposing a
safe accessor that other modules can rely on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover
    whisper = None

from .config import get_settings

logger = logging.getLogger(__name__)

_WHISPER_MODEL: Optional[whisper.Whisper] = None
_WHISPER_ERROR: Optional[str] = None


def _model_cache_path(model_size: str, cache_dir: Path) -> Path:
    """Return expected cache path for a whisper model."""
    return cache_dir / model_size


def initialize_models() -> Tuple[bool, Optional[str]]:
    """
    Ensure required models are available.

    Returns:
        Tuple[bool, Optional[str]]: success flag and optional error message.
    """
    global _WHISPER_MODEL, _WHISPER_ERROR

    settings = get_settings()
    cache_dir = settings.whisper_cache_dir
    model_size = settings.model_size

    if whisper is None:
        error_message = (
            "Whisper dependency not installed. Install openai-whisper to enable transcription."
        )
        logger.warning(error_message)
        _WHISPER_MODEL = None
        _WHISPER_ERROR = error_message
        return False, error_message

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = _model_cache_path(model_size, cache_dir)

    try:
        logger.info("Loading Whisper model '%s' (cache: %s)", model_size, cache_dir)
        model = whisper.load_model(model_size, download_root=str(cache_dir))
        _WHISPER_MODEL = model
        _WHISPER_ERROR = None
        logger.info("Whisper model '%s' ready", model_size)
        # Touch the cache path so we can detect availability even if weights are stored elsewhere.
        model_path.mkdir(parents=True, exist_ok=True)
        return True, None
    except Exception as exc:  # pragma: no cover - network issues during download
        logger.exception("Failed to load Whisper model: %s", exc)
        _WHISPER_MODEL = None
        _WHISPER_ERROR = str(exc)
        return False, _WHISPER_ERROR


def get_whisper_model():
    """
    Return the cached Whisper model instance (initialising if required).

    Raises:
        RuntimeError: if the model could not be initialised.
    """
    global _WHISPER_MODEL, _WHISPER_ERROR
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    success, error = initialize_models()
    if success and _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    raise RuntimeError(error or "Whisper model not available")

