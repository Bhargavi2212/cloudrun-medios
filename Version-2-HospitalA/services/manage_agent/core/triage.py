"""
Triage logic stubs for the manage-agent service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class TriagePayload(Protocol):
    """
    Protocol describing the inputs needed for triage classification.
    """

    hr: int
    rr: int
    sbp: int
    dbp: int
    temp_c: float
    spo2: int
    pain: int


@dataclass
class TriageResult:
    """
    Structured response from the triage engine.
    """

    acuity_level: int
    model_version: str
    explanation: str


class TriageEngine:
    """
    Lightweight heuristic-based triage engine placeholder.
    """

    def __init__(self, *, model_version: str) -> None:
        self.model_version = model_version

    async def classify(self, payload: TriagePayload) -> TriageResult:
        """
        Produce a triage score using simple heuristics.
        """

        severity_score = 0
        if payload.sbp < 90 or payload.spo2 < 92:
            severity_score += 2
        if payload.hr > 120 or payload.rr > 24 or payload.temp_c > 39.0:
            severity_score += 1
        if payload.pain >= 7:
            severity_score += 1

        acuity = 5 - min(severity_score, 4)
        explanation = (
            "Heuristic triage score based on vitals. "
            "Replace with trained model via federated learning."
        )

        logger.info(
            "✅ Generated triage acuity=%s (model=%s)", acuity, self.model_version
        )
        return TriageResult(
            acuity_level=acuity,
            model_version=self.model_version,
            explanation=explanation,
        )
