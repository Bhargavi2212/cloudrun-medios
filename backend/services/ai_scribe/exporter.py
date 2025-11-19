from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover
    FPDF = None

from backend.database.schemas import SoapNoteRead


def build_pdf(note: SoapNoteRead, *, session_info: Optional[dict] = None) -> bytes:
    if FPDF is None:
        return json.dumps(note.content or {}, indent=2).encode("utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Clinical SOAP Note", ln=True)
    pdf.set_font("Helvetica", "", 12)
    if session_info:
        pdf.cell(0, 8, f"Consultation ID: {session_info.get('consultation_id')}", ln=True)
        pdf.cell(0, 8, f"Patient ID: {session_info.get('patient_id')}", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().isoformat()}Z", ln=True)
    pdf.ln(4)

    content = note.content or {}
    _append_section(pdf, "Subjective", content.get("subjective", {}))
    _append_section(pdf, "Objective", content.get("objective", {}))
    _append_section(pdf, "Assessment", content.get("assessment", {}))
    _append_section(pdf, "Plan", content.get("plan", {}))

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


def _append_section(pdf: FPDF, title: str, section: Dict[str, Any]) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("Helvetica", "", 11)
    summary = section.get("summary") or section.get("primary_impression") or ""
    if summary:
        pdf.multi_cell(0, 6, summary)
    details = section.get("details") or section.get("exam") or section.get("therapies") or []
    for line in details:
        pdf.multi_cell(0, 6, f"- {line}")
    pdf.ln(2)


def build_fhir_document(note: SoapNoteRead, *, session_info: Optional[dict] = None) -> dict:
    content = note.content or {}
    document = {
        "resourceType": "DocumentReference",
        "status": "current",
        "type": {"text": "ED SOAP Note"},
        "subject": {"reference": f"Patient/{session_info.get('patient_id') if session_info else 'unknown'}"},
        "date": datetime.utcnow().isoformat() + "Z",
        "author": [{"reference": "Practitioner/AI-Scribe"}],
        "content": [
            {
                "attachment": {
                    "contentType": "application/json",
                    "data": base64.b64encode(json.dumps(content).encode("utf-8")).decode("ascii"),
                }
            }
        ],
    }
    if session_info and session_info.get("consultation_id"):
        document["context"] = {"encounter": [{"reference": f"Encounter/{session_info['consultation_id']}"}]}
    return document


__all__ = ["build_pdf", "build_fhir_document"]

