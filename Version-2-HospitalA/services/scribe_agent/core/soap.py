"""
SOAP note generation stubs for the scribe-agent service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class DialoguePayload(Protocol):
    """
    Protocol describing fields required to generate a SOAP note.
    """

    transcript: str
    encounter_id: str


@dataclass
class SoapGenerationResult:
    """
    Structured SOAP output produced by the engine.
    """

    subjective: str
    objective: str
    assessment: str
    plan: str
    model_version: str


class ScribeEngine:
    """
    Minimal stub for SOAP note generation, to be replaced with LLM inference.
    """

    def __init__(self, *, model_version: str) -> None:
        self.model_version = model_version

    async def generate(self, payload: DialoguePayload) -> SoapGenerationResult:
        """
        Produce a SOAP note from dialogue text.
        """

        transcript = payload.transcript.strip()
        summary_hint = (
            transcript.split("\n")[0][:120] if transcript else "No transcript supplied."
        )

        result = SoapGenerationResult(
            subjective=f"Patient states: {summary_hint}",
            objective="Vitals are stable; see triage observations for details.",
            assessment="Provisional diagnosis pending further evaluation.",
            plan=(
                "Document encounter, hand off to summarizer-agent for "
                "longitudinal update."
            ),
            model_version=self.model_version,
        )
        logger.info(
            "✅ Generated SOAP note for encounter=%s using model=%s",
            payload.encounter_id,
            self.model_version,
        )
        return result
