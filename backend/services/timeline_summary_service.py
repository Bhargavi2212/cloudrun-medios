"""Timeline aggregation and summarisation for patient records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..database import crud
from ..database.models import Patient, SummaryType, TimelineEvent
from ..database.session import get_session
from .ai_models import AIModelsService
from .config import get_settings


class TimelineSummaryService:
    """Builds patient timelines from events and generates summarised narratives."""

    def __init__(
        self,
        *,
        ai_models: Optional[AIModelsService] = None,
    ) -> None:
        self.ai_models = ai_models or AIModelsService()
        self.settings = get_settings()
        self.cache_ttl_minutes = max(self.settings.summarizer_cache_ttl_minutes, 0)

    async def generate_patient_summary(
        self,
        patient_id: str,
        *,
        force_refresh: bool = False,
        visit_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a cached or freshly generated timeline summary for the patient."""
        with get_session() as session:
            patient = session.get(Patient, patient_id)
            if patient is None:
                raise ValueError(f"Patient {patient_id} not found")

            events = crud.list_timeline_events_for_patient(
                session,
                patient_id,
                include_pending=False,
            )
            if visit_limit is not None and visit_limit > 0:
                events = events[-visit_limit:]

            serialised_events = [self._event_to_dict(event) for event in events]
            timeline_payload = {
                "patient_id": patient_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "events": serialised_events,
            }
            timeline_hash = self._timeline_hash(timeline_payload)

            cache = crud.get_summary_cache(session, patient_id)
            if cache and not force_refresh and cache.timeline_hash == timeline_hash and cache.summary_id:
                summary = crud.get_patient_summary(session, cache.summary_id)
                if summary:
                    return {
                        "summary_id": summary.id,
                        "summary": summary.content,
                        "timeline": summary.timeline or timeline_payload,
                        "cached": True,
                        "generated_at": summary.created_at,
                        "model": summary.llm_model,
                        "token_usage": summary.token_usage or {},
                        "confidence": (summary.timeline.get("confidence") if summary.timeline else None),
                    }

        # Rebuild within a fresh session for summarisation
        events_data = serialised_events
        timeline_text = self._timeline_to_text(events_data, patient)
        summariser_metadata = {
            "patient_id": patient_id,
            "event_count": len(events_data),
        }
        result = await self.ai_models.summarize_document(timeline_text, summariser_metadata)

        summary_text = result.get("summary") or "No significant events available."
        highlights = result.get("highlights", [])
        confidence = float(result.get("confidence") or 0.0)
        model_name = result.get("model")
        token_usage = {
            "tokens_prompt": result.get("tokens_prompt", 0),
            "tokens_completion": result.get("tokens_completion", 0),
        }

        timeline_payload["highlights"] = highlights
        timeline_payload["confidence"] = confidence

        with get_session() as session:
            summary_row = crud.create_patient_summary(
                session,
                patient_id=patient_id,
                summary_type=(SummaryType.LLM if not result.get("is_stub") else SummaryType.MANUAL),
                content=summary_text,
                timeline=timeline_payload,
                llm_model=model_name,
                token_usage=token_usage,
                cost_cents=result.get("cost_cents"),
            )

            expires_at = None
            if self.cache_ttl_minutes:
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.cache_ttl_minutes)

            crud.upsert_summary_cache(
                session,
                patient_id,
                timeline_hash=timeline_hash,
                summary_id=summary_row.id,
                expires_at=expires_at,
            )

        return {
            "summary_id": summary_row.id,
            "summary": summary_text,
            "timeline": timeline_payload,
            "cached": False,
            "generated_at": summary_row.created_at,
            "model": model_name,
            "token_usage": token_usage,
            "confidence": confidence,
            "highlights": highlights,
        }

    @staticmethod
    def _event_to_dict(event: TimelineEvent) -> Dict[str, Any]:
        return {
            "id": event.id,
            "patient_id": event.patient_id,
            "consultation_id": event.consultation_id,
            "source_file_id": event.source_file_id,
            "event_type": (event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)),
            "status": (event.status.value if hasattr(event.status, "value") else str(event.status)),
            "title": event.title or "Timeline Update",
            "summary": event.summary or "",
            "data": event.data or {},
            "confidence": (float(event.confidence) if event.confidence is not None else None),
            "event_date": event.event_date.isoformat() if event.event_date else None,
            "notes": event.notes,
        }

    @staticmethod
    def _timeline_hash(timeline: Dict[str, Any]) -> str:
        canonical = json.dumps(timeline, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _timeline_to_text(events: List[Dict[str, Any]], patient: Optional[Patient]) -> str:
        header = []
        if patient:
            name = " ".join(filter(None, [patient.first_name, patient.last_name])).strip() or patient.mrn or patient.id
            header.append(f"Patient: {name}")
        header.append(f"Timeline generated at {datetime.now(timezone.utc).isoformat()}")
        lines = []
        for event in events:
            date_text = event.get("event_date") or "unspecified date"
            title = event.get("title") or event.get("event_type", "event")
            summary = event.get("summary") or ""
            status = event.get("status") or ""
            lines.append(f"{date_text} [{status}] {title}: {summary}")
        if not lines:
            lines.append("No timeline events recorded.")
        return "\n".join(header + [""] + lines)
