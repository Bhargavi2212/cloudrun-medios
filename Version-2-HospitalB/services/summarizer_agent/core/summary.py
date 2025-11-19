"""
Longitudinal summarization stubs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class SummaryPayload(Protocol):
    """
    Protocol describing fields needed for summary generation.
    """

    patient_id: str
    encounter_ids: list[str]
    highlights: list[str] | None


@dataclass
class SummaryResult:
    """
    Structured summarization result.
    """

    summary_text: str
    model_version: str
    confidence_score: float


class SummarizerEngine:
    """
    Stub summarizer engine that concatenates highlights into a cohesive summary.
    """

    def __init__(self, *, model_version: str) -> None:
        self.model_version = model_version

    async def summarize(self, payload: SummaryPayload) -> SummaryResult:
        """
        Produce a narrative summary for the patient.
        """

        highlights = payload.highlights or []
        bullet_points = (
            "; ".join(highlights) if highlights else "No highlights provided."
        )
        encounter_ids = [str(encounter_id) for encounter_id in payload.encounter_ids]
        summary_text = (
            f"Patient {payload.patient_id} recent encounters ({', '.join(encounter_ids)}): "
            f"{bullet_points}"
        )

        logger.info(
            "✅ Generated summary for patient=%s encounters=%s",
            payload.patient_id,
            encounter_ids,
        )
        return SummaryResult(
            summary_text=summary_text,
            model_version=self.model_version,
            confidence_score=0.6,
        )
