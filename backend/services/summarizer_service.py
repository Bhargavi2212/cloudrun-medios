from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Protocol

from .config import get_settings
from .telemetry import record_service_metric

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Optional import of the standalone summarizer package
# --------------------------------------------------------------------------- #

SUMMARIZER_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "medi-os" / "services" / "manage-agent"
)
if SUMMARIZER_PACKAGE_ROOT.exists():
    sys.path.insert(0, str(SUMMARIZER_PACKAGE_ROOT))

try:  # pragma: no cover - import side effects exercised in integration tests
    from summarizer.config import \
        Settings as SummarizerSettings  # type: ignore[import-not-found]
    from summarizer.data_loader import \
        DataLoaderConfig  # type: ignore[import-not-found]
    from summarizer.errors import (  # type: ignore[import-not-found]
        ConfigurationError, PatientNotFoundError, SummarizerError)
    from summarizer.summarizer import \
        SummarizerService as CoreSummarizer  # type: ignore[import-not-found]
    from summarizer.summarizer import SummaryResult
except Exception:  # pragma: no cover - summarizer optional
    SummarizerSettings = None  # type: ignore[assignment]

    class SummarizerError(Exception):
        pass

    class ConfigurationError(SummarizerError):
        pass

    class PatientNotFoundError(SummarizerError):
        pass

    CoreSummarizer = None  # type: ignore[assignment]
    SummaryResult = Any  # type: ignore[assignment]
    DataLoaderConfig = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Cache protocol and implementation
# --------------------------------------------------------------------------- #


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[dict]: ...

    def set(self, key: str, value: dict) -> None: ...

    def clear(self) -> None: ...


class InMemoryCache(CacheBackend):
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = max(ttl_seconds, 0)
        self._store: Dict[str, dict] = {}

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if self.ttl and perf_counter() - entry["created_at"] > self.ttl:
            self._store.pop(key, None)
            return None
        return entry["payload"]

    def set(self, key: str, value: dict) -> None:
        self._store[key] = {"payload": value, "created_at": perf_counter()}

    def clear(self) -> None:
        self._store.clear()


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


@dataclass
class SummarizerResponse:
    subject_id: int
    summary_markdown: str
    timeline: Dict[str, Any]
    metrics: Dict[str, Any]
    cached: bool
    is_stub: bool


class MedicalSummarizer:
    """Orchestrates the summarizer pipeline with caching and fallbacks."""

    def __init__(
        self,
        *,
        core_service: Optional[CoreSummarizer] = None,
        cache_backend: Optional[CacheBackend] = None,
    ) -> None:
        self.settings = get_settings()
        ttl_seconds = max(self.settings.summarizer_cache_ttl_minutes, 0) * 60
        self.cache: CacheBackend = cache_backend or InMemoryCache(
            ttl_seconds=ttl_seconds
        )
        self.core: Optional[CoreSummarizer] = core_service
        self.is_stub: bool = False

        if self.core is None:
            self.core = self._initialise_core()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def summarize_patient(
        self,
        subject_id: int,
        *,
        visit_limit: Optional[int] = None,
        force_refresh: bool = False,
    ) -> SummarizerResponse:
        cache_key = self._cache_key(subject_id, visit_limit)
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return SummarizerResponse(
                    subject_id=subject_id,
                    summary_markdown=cached["summary_markdown"],
                    timeline=cached["timeline"],
                    metrics=cached["metrics"],
                    cached=True,
                    is_stub=cached.get("is_stub", False),
                )

        if self.core is None or self.is_stub:
            logger.warning("Summarizer core unavailable; returning stub summary.")
            payload = self._stub_summary(subject_id)
            if not force_refresh:
                self.cache.set(cache_key, payload)
            return SummarizerResponse(
                subject_id=subject_id,
                summary_markdown=payload["summary_markdown"],
                timeline=payload["timeline"],
                metrics=payload["metrics"],
                cached=False,
                is_stub=True,
            )

        start_ts = perf_counter()
        try:
            result = self.core.summarize(subject_id=subject_id, visit_limit=visit_limit)
        except PatientNotFoundError:
            raise
        except SummarizerError as exc:
            logger.exception("Summarizer pipeline failed: %s", exc)
            raise
        finally:
            latency_ms = (perf_counter() - start_ts) * 1000.0
            try:
                record_service_metric(
                    service_name="summarizer",
                    metric_name="request_latency_ms",
                    metric_value=latency_ms,
                    metadata={"subject_id": str(subject_id)},
                )
            except Exception:  # pragma: no cover - telemetry failures shouldn't bubble
                logger.debug(
                    "Unable to record summarizer latency metric", exc_info=True
                )

        payload = self._serialise_result(result)
        payload["cached"] = False
        payload["is_stub"] = False
        payload["fingerprint"] = self._timeline_fingerprint(payload["timeline"])
        self.cache.set(cache_key, payload)

        return SummarizerResponse(
            subject_id=subject_id,
            summary_markdown=payload["summary_markdown"],
            timeline=payload["timeline"],
            metrics=payload["metrics"],
            cached=False,
            is_stub=False,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _initialise_core(self) -> Optional[CoreSummarizer]:
        if not self.settings.summarizer_enabled:
            self.is_stub = True
            return None

        if SummarizerSettings is None or CoreSummarizer is None:
            logger.warning("Summarizer package unavailable; running in stub mode.")
            self.is_stub = True
            return None

        try:
            loader_defaults = DataLoaderConfig() if DataLoaderConfig else None
            data_glob = self.settings.summarizer_data_glob or (
                loader_defaults.data_glob if loader_defaults else None
            )
            codes_path = self.settings.summarizer_codes_path or (
                loader_defaults.codes_path if loader_defaults else None
            )
            if data_glob is None or codes_path is None:
                raise ConfigurationError("Summarizer dataset paths are not configured.")

            summarizer_settings = SummarizerSettings(
                data_glob=data_glob,
                codes_path=Path(codes_path),
                max_cache_entries=self.settings.summarizer_max_cache_entries,
                gemini_api_key=self.settings.gemini_api_key,
                gemini_model=self.settings.gemini_model,
                summarizer_temperature=self.settings.summarizer_temperature,
                summarizer_max_tokens=self.settings.summarizer_max_tokens,
                stop_gap_days=self.settings.summarizer_stop_gap_days,
                slow_request_threshold=self.settings.summarizer_slow_threshold,
                use_fake_llm=self.settings.summarizer_use_fake_llm
                or not bool(self.settings.gemini_api_key),
            )
            return CoreSummarizer(settings=summarizer_settings)
        except ConfigurationError as exc:
            logger.warning("Summarizer configuration error: %s", exc)
            self.is_stub = True
            return None
        except Exception as exc:  # pragma: no cover - unexpected failure
            logger.exception("Failed to initialise summarizer core: %s", exc)
            self.is_stub = True
            return None

    def _serialise_result(self, result: SummaryResult) -> dict:
        return {
            "subject_id": getattr(result, "subject_id", None),
            "summary_markdown": getattr(result, "markdown_summary", ""),
            "timeline": getattr(result, "structured_timeline", {}),
            "metrics": getattr(result, "metrics", {}),
        }

    def _stub_summary(self, subject_id: int) -> dict:
        return {
            "subject_id": subject_id,
            "summary_markdown": (
                "# Summarizer Unavailable\n"
                "Gemini summarization is not configured. Provide a valid GEMINI_API_KEY and "
                "dataset paths to enable full functionality."
            ),
            "timeline": {},
            "metrics": {},
            "cached": False,
            "is_stub": True,
        }

    @staticmethod
    def _cache_key(subject_id: int, visit_limit: Optional[int]) -> str:
        return f"{subject_id}:{visit_limit if visit_limit is not None else 'all'}"

    @staticmethod
    def _timeline_fingerprint(timeline: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps(timeline, default=str, sort_keys=True).encode(
                "utf-8"
            )
        except TypeError:
            serialized = repr(timeline).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


__all__ = [
    "MedicalSummarizer",
    "SummarizerResponse",
    "PatientNotFoundError",
    "SummarizerError",
]
