"""Rate limiting configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import get_settings


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool
    default_per_minute: int
    burst_multiplier: float

    @property
    def burst_limit(self) -> int:
        return int(self.default_per_minute * self.burst_multiplier)


def get_rate_limit_config() -> RateLimitConfig:
    settings = get_settings()
    return RateLimitConfig(
        enabled=settings.rate_limit_enabled,
        default_per_minute=settings.rate_limit_default_per_minute,
        burst_multiplier=settings.rate_limit_burst_multiplier,
    )
