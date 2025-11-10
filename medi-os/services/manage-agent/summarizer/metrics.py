"""
Lightweight in-memory metrics collector for the summarizer service.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    name: str
    duration: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
        }


class MetricsRecorder:
    """Thread-safe recorder for timings and counters."""

    def __init__(self, *, slow_threshold: float = 5.0) -> None:
        self._timings: List[TimingRecord] = []
        self._counters: Counter[str] = Counter()
        self._lock = threading.Lock()
        self.slow_threshold = slow_threshold

    @contextmanager
    def time(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            duration = perf_counter() - start
            self.record_timing(name, duration)

    def record_timing(self, name: str, duration: float) -> None:
        record = TimingRecord(name=name, duration=duration)
        with self._lock:
            self._timings.append(record)
        if duration >= self.slow_threshold:
            logger.warning("Slow operation detected", extra={"name": name, "duration": duration})

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def snapshot(self) -> dict:
        with self._lock:
            timings = [record.to_dict() for record in self._timings[-50:]]
            counters = dict(self._counters)
        return {"timings": timings, "counters": counters}


__all__ = ["MetricsRecorder"]

