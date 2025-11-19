"""
Schema exports for the scribe-agent service.
"""

from services.scribe_agent.schemas.soap import SoapGenerateRequest, SoapResponse
from services.scribe_agent.schemas.transcript import (
    DialogueSegment,
    TranscriptCreate,
    TranscriptRead,
)

__all__ = [
    "DialogueSegment",
    "SoapGenerateRequest",
    "SoapResponse",
    "TranscriptCreate",
    "TranscriptRead",
]
