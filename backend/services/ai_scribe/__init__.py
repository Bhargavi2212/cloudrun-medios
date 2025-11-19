"""AI scribe service package exposing shared singletons."""

from .audio_gateway import AudioGatewayService
from .summarizer import GeminiSoapSummarizer
from .triage_bridge import NurseTriageBridge

audio_gateway_service = AudioGatewayService()
soap_summarizer = GeminiSoapSummarizer()
triage_bridge = NurseTriageBridge()

__all__ = ["audio_gateway_service", "soap_summarizer", "triage_bridge"]

