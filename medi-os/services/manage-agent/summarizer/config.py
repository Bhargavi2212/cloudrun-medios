"""
Configuration management for the summarizer service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from summarizer.data_loader import DataLoaderConfig
from summarizer.errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    data_glob: str
    codes_path: Path
    max_cache_entries: int
    gemini_api_key: Optional[str]
    gemini_model: str
    summarizer_temperature: float
    summarizer_max_tokens: int
    stop_gap_days: int
    slow_request_threshold: float
    use_fake_llm: bool


def _parse_int(value: Optional[str], *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer value: {value}") from exc


def _parse_float(value: Optional[str], *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid float value: {value}") from exc


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    base_config = DataLoaderConfig()
    data_glob = os.getenv("EHRSHOT_DATA_GLOB", base_config.data_glob)
    codes_path = Path(os.getenv("EHRSHOT_CODES_PATH", str(base_config.codes_path)))
    max_cache_entries = _parse_int(
        os.getenv("SUMMARIZER_CACHE_ENTRIES"), default=base_config.max_cache_entries
    )
    stop_gap_days = _parse_int(
        os.getenv("SUMMARIZER_STOP_GAP_DAYS"), default=90
    )
    slow_request_threshold = _parse_float(
        os.getenv("SUMMARIZER_SLOW_THRESHOLD"), default=5.0
    )
    summarizer_temperature = _parse_float(
        os.getenv("SUMMARY_TEMPERATURE"), default=0.2
    )
    summarizer_max_tokens = _parse_int(
        os.getenv("SUMMARY_MAX_TOKENS"), default=3000
    )
    use_fake_llm = os.getenv("USE_FAKE_LLM", "false").lower() in {"1", "true", "yes"}

    if not codes_path.exists():
        raise ConfigurationError(f"Codes parquet not found at {codes_path}")

    return Settings(
        data_glob=data_glob,
        codes_path=codes_path,
        max_cache_entries=max_cache_entries,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro"),
        summarizer_temperature=summarizer_temperature,
        summarizer_max_tokens=summarizer_max_tokens,
        stop_gap_days=stop_gap_days,
        slow_request_threshold=slow_request_threshold,
        use_fake_llm=use_fake_llm,
    )


__all__ = ["Settings", "load_settings"]

