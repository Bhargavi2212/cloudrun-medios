from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

from backend.database import crud
from backend.database.models import ScribeSessionStatus
from backend.database.session import get_session
from backend.services.config import get_settings

logger = logging.getLogger(__name__)

FEW_SHOT_EXAMPLES = [
    {
        "transcript": "Doctor: Hello Ms. Perez, I see you are short of breath.\nPatient: Yes doctor, worsening tonight.",
        "soap": {
            "subjective": {
                "summary": "Patient reports acute shortness of breath that worsened tonight.",
                "details": [
                    "Symptoms began 2 hours ago",
                    "No chest pain, denies cough",
                    "History of asthma, home inhaler ineffective",
                ],
                "confidence": 0.78,
            },
            "objective": {
                "vitals": {"heart_rate": 110, "respiratory_rate": 30, "systolic_bp": 148, "temperature_c": 37.1},
                "exam": [
                    "Speaking in short sentences",
                    "Diffuse wheezing bilaterally",
                ],
                "confidence": 0.74,
            },
            "assessment": {
                "primary_impression": "Asthma exacerbation",
                "differential": ["Pneumonia", "Anxiety"],
                "confidence": 0.72,
            },
            "plan": {
                "diagnostics": ["Peak flow", "Portable chest X-ray"],
                "therapies": ["Nebulized albuterol/ipratropium", "IV steroids"],
                "disposition": "Monitor in ED, reassess q30m",
                "confidence": 0.76,
            },
            "entities": {
                "diagnoses": [{"name": "Asthma exacerbation", "confidence": 0.8}],
                "medications": [{"name": "Albuterol neb", "confidence": 0.73}],
                "labs": [],
                "orders": ["Chest X-ray"],
            },
        },
    },
    {
        "transcript": "Doctor: What brings you in?\nPatient: Severe abdominal pain since lunch, nausea and vomiting.",
        "soap": {
            "subjective": {
                "summary": "Sudden severe periumbilical abdominal pain after eating lunch.",
                "details": [
                    "Associated nausea/vomiting x3",
                    "Denies diarrhea",
                    "No prior abdominal surgeries",
                ],
                "confidence": 0.75,
            },
            "objective": {
                "vitals": {"heart_rate": 98, "respiratory_rate": 20, "systolic_bp": 132, "temperature_c": 37.9},
                "exam": [
                    "Abdomen soft but diffusely tender",
                    "Mild guarding, no rebound",
                ],
                "confidence": 0.7,
            },
            "assessment": {
                "primary_impression": "Acute gastroenteritis vs early appendicitis",
                "differential": ["Gastritis", "Gallbladder disease"],
                "confidence": 0.69,
            },
            "plan": {
                "diagnostics": ["CBC", "CMP", "Lipase", "CT abdomen/pelvis"],
                "therapies": ["IV fluids", "Ondansetron", "Ketorolac"],
                "disposition": "Pending CT and labs",
                "confidence": 0.71,
            },
            "entities": {
                "diagnoses": [{"name": "Possible appendicitis", "confidence": 0.65}],
                "medications": [
                    {"name": "Ondansetron 4mg IV", "confidence": 0.7},
                    {"name": "Ketorolac 15mg IV", "confidence": 0.67},
                ],
                "labs": ["CBC", "CMP", "Lipase"],
                "orders": ["CT abdomen/pelvis"],
            },
        },
    },
]


class GeminiSoapSummarizer:
    """Generates structured SOAP notes from transcripts using Gemini 2.0 Flash."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Optional[Any] = None
        self._stub_mode = False
        self._configure_client()

    def _configure_client(self) -> None:
        if not self.settings.gemini_api_key or genai is None:
            logger.warning("Gemini API key missing or google-generativeai unavailable; running summarizer in stub mode.")
            self._stub_mode = True
            return
        genai.configure(api_key=self.settings.gemini_api_key)
        self._model = genai.GenerativeModel(
            self.settings.gemini_model or "gemini-2.0-flash-exp",
        )

    async def summarize_session(self, session_id: str, *, specialty: str = "ED") -> dict:
        """Generate SOAP content and persist to database."""
        transcript, vitals, context = self._collect_session_data(session_id)
        payload = {
            "session_id": session_id,
            "transcript": transcript,
            "vitals": vitals,
            "context": context,
            "specialty": specialty,
        }
        if self._stub_mode or self._model is None:
            logger.info("SOAP summarizer running in stub mode.")
            content = self._build_stub_response(payload)
            return self._persist_note(session_id, content, is_stub=True)

        prompt = self._build_prompt(payload)
        attempt = 0
        response_text = ""
        latency_ms = 0
        while attempt < 3:
            attempt += 1
            try:
                loop = asyncio.get_running_loop()
                start = loop.time()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(
                        [prompt],
                        generation_config={
                            "temperature": self.settings.gemini_temperature,
                            "max_output_tokens": self.settings.gemini_max_tokens,
                        },
                    ),
                )
                latency_ms = int((loop.time() - start) * 1000)
                response_text = getattr(result, "text", "") or ""
                break
            except Exception as exc:  # pragma: no cover - external dependency
                logger.warning("Gemini call failed (attempt %s/3): %s", attempt, exc)
                if attempt >= 3:
                    raise
                await asyncio.sleep(2**attempt)

        content = self._parse_response(response_text) or self._build_stub_response(payload)
        return self._persist_note(session_id, content, latency_ms=latency_ms)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _collect_session_data(self, session_id: str) -> tuple[list[dict], list[dict], dict]:
        with get_session() as db:
            session = crud.get_scribe_session(db, session_id)
            if session is None:
                raise ValueError(f"Scribe session {session_id} not found.")
            segments = crud.list_transcript_segments(db, session_id, limit=2000)
            vitals = crud.list_scribe_vitals(db, session_id, limit=200)

        transcript_payload = [
            {
                "speaker": seg.speaker_label or "unknown",
                "text": seg.text,
                "startMs": seg.start_ms,
                "endMs": seg.end_ms,
                "confidence": seg.confidence,
            }
            for seg in segments
        ]
        vitals_payload = [
            {
                "recorded_at": entry.recorded_at.isoformat(),
                "source": entry.source,
                "heart_rate": entry.heart_rate,
                "respiratory_rate": entry.respiratory_rate,
                "systolic_bp": entry.systolic_bp,
                "diastolic_bp": entry.diastolic_bp,
                "temperature_c": float(entry.temperature_c) if entry.temperature_c is not None else None,
                "spo2": entry.oxygen_saturation,
                "pain": entry.pain_score,
            }
            for entry in vitals
        ]
        context = {
            "consultation_id": session.consultation_id,
            "patient_id": session.patient_id,
            "language": session.language or "en",
        }
        return transcript_payload, vitals_payload, context

    def _build_prompt(self, payload: dict) -> str:
        examples_str = "\n\n".join(
            textwrap.dedent(
                f"""Example Transcript:
{example['transcript']}

Example SOAP JSON:
{json.dumps(example['soap'], indent=2)}"""
            )
            for example in FEW_SHOT_EXAMPLES
        )
        instructions = textwrap.dedent(
            f"""
You are an emergency department AI scribe. Produce a structured SOAP note in JSON matching this schema:
{{
  "subjective": {{ "summary": str, "details": [str], "confidence": float }},
  "objective": {{ "vitals": {{"heart_rate": int, ...}}, "exam": [str], "confidence": float }},
  "assessment": {{ "primary_impression": str, "differential": [str], "confidence": float }},
  "plan": {{ "diagnostics": [str], "therapies": [str], "disposition": str, "confidence": float }},
  "entities": {{
      "diagnoses": [{{"name": str, "confidence": float}}],
      "medications": [{{"name": str, "confidence": float}}],
      "labs": [str],
      "orders": [str]
  }}
}}

Requirements:
- Specialty: {payload['specialty']}
- Include explicit confidence scores (0-1) per section and entity.
- If information is missing, set value to null and describe reason in details.
- Use vitals from manual entry when available; otherwise infer cautiously from transcript.

FEW SHOT EXAMPLES:
{examples_str}

PATIENT CONTEXT:
{json.dumps(payload['context'], indent=2)}

VITALS (chronological):
{json.dumps(payload['vitals'], indent=2)}

TRANSCRIPT (chronological):
{json.dumps(payload['transcript'], indent=2)}

Return ONLY JSON that conforms to the schema. Do not include prose or explanations.
"""
        )
        return instructions.strip()

    def _parse_response(self, response_text: str) -> Optional[dict]:
        if not response_text:
            return None
        candidate = response_text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            candidate = candidate.split("\n", 1)[-1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("Unable to parse Gemini response as JSON: %s", candidate[:200])
            return None

    def _build_stub_response(self, payload: dict) -> dict:
        transcript_preview = " ".join([seg["text"] for seg in payload["transcript"][:3]])
        return {
            "subjective": {
                "summary": transcript_preview or "No transcript available.",
                "details": [],
                "confidence": 0.0,
            },
            "objective": {"vitals": (payload["vitals"][-1] if payload["vitals"] else {}), "exam": [], "confidence": 0.0},
            "assessment": {"primary_impression": "Unavailable", "differential": [], "confidence": 0.0},
            "plan": {"diagnostics": [], "therapies": [], "disposition": "Pending", "confidence": 0.0},
            "entities": {"diagnoses": [], "medications": [], "labs": [], "orders": []},
        }

    def _persist_note(self, session_id: str, content: dict, *, latency_ms: int = 0, is_stub: bool = False) -> dict:
        with get_session() as db:
            session_obj = crud.get_scribe_session(db, session_id)
            soap = crud.create_soap_note(
                db,
                session_id=session_id,
                consultation_id=session_obj.consultation_id if session_obj else None,
                status="draft",
                model_name=self.settings.gemini_model if not is_stub else "stub",
                specialty="ED",
                content=content,
                raw_markdown=self._convert_to_markdown(content),
                confidence=self._extract_confidence(content),
                latency_ms=latency_ms,
            )
            if session_obj:
                target_status = ScribeSessionStatus.COMPLETED if not is_stub else ScribeSessionStatus.SUMMARIZING
                crud.update_scribe_session(db, session_obj, status=target_status)
        return {
            "note_id": soap.id,
            "content": content,
            "latency_ms": latency_ms,
            "stub": is_stub,
        }

    @staticmethod
    def _convert_to_markdown(content: dict) -> str:
        def section(title: str, body: list[str]) -> str:
            return f"## {title}\n" + ("\n".join(f"- {line}" for line in body) if body else "_Not documented_")

        subjective = section("Subjective", [content.get("subjective", {}).get("summary", "")])
        objective_lines = content.get("objective", {}).get("exam", [])
        assessment_lines = [content.get("assessment", {}).get("primary_impression", "")]
        plan_lines = content.get("plan", {}).get("therapies", [])
        return "\n\n".join([subjective, section("Objective", objective_lines), section("Assessment", assessment_lines), section("Plan", plan_lines)])

    @staticmethod
    def _extract_confidence(content: dict) -> dict:
        return {
            "subjective": content.get("subjective", {}).get("confidence"),
            "objective": content.get("objective", {}).get("confidence"),
            "assessment": content.get("assessment", {}).get("confidence"),
            "plan": content.get("plan", {}).get("confidence"),
        }


__all__ = ["GeminiSoapSummarizer"]

