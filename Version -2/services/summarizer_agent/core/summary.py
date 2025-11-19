"""
Longitudinal summarization with Gemini AI integration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol
from uuid import UUID

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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
    structured_data: dict[str, Any] | None
    model_version: str
    confidence_score: float


class SummarizerEngine:
    """
    AI-powered summarizer engine using Gemini to generate narrative summaries.
    Falls back to stub if Gemini is unavailable.
    """

    def __init__(
        self,
        *,
        model_version: str,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-2.0-flash-exp",
    ) -> None:
        self.model_version = model_version
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self._model = None
        self._enabled = False

        if gemini_api_key and genai is not None:
            try:
                genai.configure(api_key=gemini_api_key)
                self._model = genai.GenerativeModel(gemini_model)
                self._enabled = True
                logger.info("SummarizerEngine initialized with Gemini %s", gemini_model)
            except Exception as e:
                logger.warning("Failed to initialize Gemini model: %s", e)
        else:
            if not gemini_api_key:
                logger.warning("Gemini API key not provided, using stub summarizer")
            if genai is None:
                logger.warning(
                    "google-generativeai not installed, using stub summarizer"
                )

    async def summarize(self, payload: SummaryPayload, session=None) -> SummaryResult:
        """
        Produce a structured timeline summary for the patient.
        Builds timeline from encounters, SOAP notes, and documents.
        Falls back to stub if session is unavailable.
        """
        import datetime as dt_module
        import sys
        import traceback

        # Force immediate file write to verify code execution
        with open("debug_summarizer.log", "a", encoding="utf-8") as f:
            f.write(f"[{dt_module.datetime.now()}] [SUMMARIZE] METHOD CALLED\n")
            f.write(f"  Patient: {payload.patient_id}\n")
            f.write(f"  Session: {session is not None}\n")
            f.write(f"  Gemini enabled: {self._enabled}\n")
            f.write(f"  Model exists: {self._model is not None}\n")
            f.flush()

        logger.info("[SUMMARIZE] Starting for patient=%s", payload.patient_id)
        logger.info(
            "[SUMMARIZE] Gemini enabled=%s, model=%s",
            self._enabled,
            self._model is not None,
        )
        sys.stderr.write(f"[SUMMARIZE] Starting for patient={payload.patient_id}\n")
        sys.stderr.write(
            f"[SUMMARIZE] Gemini enabled={self._enabled}, model={self._model is not None}\n"  # noqa: E501
        )
        sys.stderr.flush()

        if session is None:
            with open("debug_summarizer.log", "a", encoding="utf-8") as f:
                f.write(
                    f"[{dt_module.datetime.now()}] [SUMMARIZE] SESSION IS NONE - RETURNING STUB\n"  # noqa: E501
                )
                f.flush()
            logger.warning("No session provided, using stub summary")
            sys.stderr.write("[SUMMARIZE] No session, using stub\n")
            sys.stderr.flush()
            return self._stub_summary(payload)

        try:
            # Build structured timeline from database
            logger.info("[SUMMARIZE] Building structured timeline...")
            print(
                "[SUMMARIZE] Building structured timeline...",
                file=sys.stderr,
                flush=True,
            )
            structured_data = await self._build_structured_timeline(payload, session)
            logger.info(
                "[SUMMARIZE] Timeline built: %d entries",
                len(structured_data.get("timeline", [])),
            )
            print(
                f"[SUMMARIZE] Timeline built: {len(structured_data.get('timeline', []))} entries",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )

            #  Generate narrative summary text using Gemini (optional, for backward compatibility)
            summary_text = ""
            if self._enabled and self._model is not None:
                logger.info("[SUMMARIZE] Calling Gemini for narrative summary...")
                print(
                    "[SUMMARIZE] Calling Gemini for narrative summary...",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    context_text = await self._gather_patient_context(payload, session)
                    logger.info(
                        "[SUMMARIZE] Context gathered: %d chars", len(context_text)
                    )
                    print(
                        f"[SUMMARIZE] Context gathered: {len(context_text)} chars",
                        file=sys.stderr,
                        flush=True,
                    )
                    prompt = self._build_summary_prompt(payload, context_text)

                    def _run_gemini() -> str:
                        print(
                            "[SUMMARIZE] Gemini API call starting...",
                            file=sys.stderr,
                            flush=True,
                        )
                        response = self._model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.7,
                                "max_output_tokens": 3000,  # Increased for
                                # more comprehensive summaries
                            },
                        )
                        result = (response.text or "").strip()
                        print(
                            f"[SUMMARIZE] Gemini returned: {len(result)} chars",
                            file=sys.stderr,
                            flush=True,
                        )
                        return result

                    summary_text = await asyncio.to_thread(_run_gemini)
                    logger.info(
                        "[SUMMARIZE] Gemini summary generated: %d chars",
                        len(summary_text),
                    )
                    print(
                        f"[SUMMARIZE] Gemini summary generated: {len(summary_text)} chars",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[SUMMARIZE] Gemini failed: {e}", file=sys.stderr, flush=True
                    )
                    logger.warning(
                        "Failed to generate narrative summary with Gemini: %s",
                        e,
                        exc_info=True,
                    )
            else:
                print(
                    f"[SUMMARIZE] Gemini not enabled (enabled={self._enabled}, model={self._model is not None})",  # noqa: E501
                    file=sys.stderr,
                    flush=True,
                )

            # Fallback to simple text if Gemini failed
            if not summary_text:
                fallback_text = f"Patient timeline with {len(structured_data.get('timeline', []))} entries"  # noqa: E501
                print(
                    f"[SUMMARIZE] Using fallback text: {fallback_text}",
                    file=sys.stderr,
                    flush=True,
                )
                summary_text = fallback_text

            logger.info(
                "Generated structured timeline summary for patient=%s encounters=%s",
                payload.patient_id,
                payload.encounter_ids,
            )
            print(
                f"[SUMMARIZE] Summary complete: text_len={len(summary_text)}, timeline_entries={len(structured_data.get('timeline', []))}",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )
            return SummaryResult(
                summary_text=summary_text,
                structured_data=structured_data,
                model_version=f"{self.model_version}+gemini"
                if self._enabled
                else self.model_version,
                confidence_score=0.95 if structured_data else 0.6,
            )
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logger.error("[SUMMARIZE] EXCEPTION: %s", error_msg, exc_info=True)
            logger.error("[SUMMARIZE] TRACEBACK:\n%s", error_trace)
            print(f"[SUMMARIZE] EXCEPTION: {error_msg}", file=sys.stderr, flush=True)
            print(f"[SUMMARIZE] TRACEBACK:\n{error_trace}", file=sys.stderr, flush=True)

            with open("debug_summarizer.log", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(
                    f"[{dt_module.datetime.now()}] ERROR in summarize: {error_msg}\n"
                )
                f.write(f"TRACEBACK:\n{error_trace}\n")
                f.write(f"Patient ID: {payload.patient_id}\n")
                f.write(f"Encounter IDs: {payload.encounter_ids}\n")

            logger.error(
                "Failed to build structured timeline: %s", error_msg, exc_info=True
            )
            logger.error(
                "[SUMMARIZE] FALLING BACK TO STUB SUMMARY for patient=%s",
                payload.patient_id,
            )
            print(
                "[SUMMARIZE] FALLING BACK TO STUB SUMMARY", file=sys.stderr, flush=True
            )
            return self._stub_summary(payload)

    async def _gather_patient_context(self, payload: SummaryPayload, session) -> str:
        """
        Gather patient data from database to provide context for summary.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        from database.models import (
            Encounter,
            FileAsset,
            Patient,
        )

        # Handle both UUID and string types
        patient_id = (
            payload.patient_id
            if isinstance(payload.patient_id, UUID)
            else UUID(str(payload.patient_id))
        )
        encounter_ids = [
            eid if isinstance(eid, UUID) else UUID(str(eid))
            for eid in payload.encounter_ids
        ]

        # Fetch patient
        stmt = select(Patient).where(Patient.id == patient_id)
        result = await session.execute(stmt)
        patient = result.scalar_one_or_none()
        if not patient:
            return ""

        context_parts = [
            f"Patient: {patient.first_name} {patient.last_name}",
            f"MRN: {patient.mrn}",
        ]
        if patient.dob:
            context_parts.append(f"DOB: {patient.dob}")

        # Fetch encounters with related data
        stmt = (
            select(Encounter)
            .where(Encounter.patient_id == patient_id)
            .where(Encounter.id.in_(encounter_ids))
            .options(
                selectinload(Encounter.triage_observations),
                selectinload(Encounter.soap_notes),
            )
            .order_by(Encounter.arrival_ts.desc())
        )
        result = await session.execute(stmt)
        encounters = result.scalars().all()

        for encounter in encounters:
            context_parts.append(
                f"\n--- Encounter {encounter.id} ({encounter.arrival_ts}) ---"
            )

            # Triage observations (vitals, chief complaint)
            if encounter.triage_observations:
                for triage in encounter.triage_observations:
                    if triage.chief_complaint:
                        context_parts.append(
                            f"Chief Complaint: {triage.chief_complaint}"
                        )
                    if triage.vitals:
                        vitals_str = ", ".join(
                            [f"{k}: {v}" for k, v in triage.vitals.items() if v]
                        )
                        if vitals_str:
                            context_parts.append(f"Vitals: {vitals_str}")
                    if triage.notes:
                        context_parts.append(f"Notes: {triage.notes}")

            # SOAP notes
            if encounter.soap_notes:
                for soap in encounter.soap_notes:
                    soap_parts = []
                    if soap.subjective:
                        soap_parts.append(f"Subjective: {soap.subjective}")
                    if soap.objective:
                        soap_parts.append(f"Objective: {soap.objective}")
                    if soap.assessment:
                        soap_parts.append(f"Assessment: {soap.assessment}")
                    if soap.plan:
                        soap_parts.append(f"Plan: {soap.plan}")
                    if soap_parts:
                        context_parts.append("SOAP Note:\n" + "\n".join(soap_parts))

        # Fetch uploaded documents with their extracted text
        stmt = (
            select(FileAsset)
            .where(FileAsset.patient_id == patient_id)
            .where(FileAsset.encounter_id.in_(encounter_ids))
            .order_by(FileAsset.created_at.desc())
        )
        result = await session.execute(stmt)
        documents = result.scalars().all()
        if documents:
            doc_parts = []
            for doc in documents:
                doc_name = doc.original_filename or "Unknown"
                doc_type = doc.document_type or "document"
                doc_parts.append(f"- {doc_name} ({doc_type})")
                # Include extracted text if available
                if doc.raw_text and doc.raw_text.strip():
                    # Truncate long documents to first 2000 chars for context
                    text_preview = (
                        doc.raw_text[:2000] + "..."
                        if len(doc.raw_text) > 2000
                        else doc.raw_text
                    )
                    doc_parts.append(f"  Content: {text_preview}")
            context_parts.append("\nUploaded Documents:\n" + "\n".join(doc_parts))

        return "\n".join(context_parts)

    def _build_summary_prompt(self, payload: SummaryPayload, context_text: str) -> str:
        """
        Build the prompt for Gemini to generate a summary.
        """
        highlights = payload.highlights or []
        highlights_text = (
            "\n".join([f"- {h}" for h in highlights])
            if highlights
            else "None provided."
        )

        prompt = f"""You are an expert Medical AI assistant. Your task is to
generate a highly structured, chronological timeline summary of the patient's
medical history based on the provided context documents and encounters.

Patient Context:
{context_text if context_text else "No detailed context available."}

Recent Highlights:
{highlights_text}

**INSTRUCTIONS:**
1.  **Format**: Group events chronologically by **Date and Time**
    (e.g., `### **October 1, 2024 - 2:05 PM**`).
2.  **Structure**: Use numbered lists for distinct reports or encounters
    (e.g., `**1. Uric Acid Test (LAB141)**`).
3.  **Key Findings**: Use bold headers for findings (e.g., `* **Result:**`,
    `* **Key Finding:**`, `* **Clinical Note:**`).
4.  **Citations**: You MUST cite the source of information using the format
    `[Source: Document Name]` or `[Source: Encounter Date]` at the start or
    end of statements.
    *   Example: `[Source: Lab Report] Total Bilirubin was elevated at **2.1 mg/dL**`.
    *   If exact line numbers aren't available, cite the document title.
5.  **Content**:
    *   Include specific values and reference ranges where available.
    *   Note explicit clinical interpretations (e.g., "not surprising given...").
    *   Mention status (e.g., "non-fasting").
6.  **Tone**: Professional, objective, and clinical.

**EXAMPLE OUTPUT FORMAT:**

### **October 1, 2024 - 2:05 PM**

**1. Uric Acid Test**
* **Result:** 8.0 mg/dL [Source: Lab Report A].
* **Status:** Within reference range.

**2. Comprehensive Metabolic Panel**
* **Key Finding:** Total Bilirubin elevated at **2.1 mg/dL** [Source: Lab Report B].

**GENERATE THE SUMMARY NOW:**"""

        return prompt

    def _stub_summary(self, payload: SummaryPayload) -> SummaryResult:
        """
        Fallback stub summary when Gemini is unavailable.
        """
        highlights = payload.highlights or []
        bullet_points = (
            "; ".join(highlights) if highlights else "No highlights provided."
        )
        encounter_ids = [str(encounter_id) for encounter_id in payload.encounter_ids]
        summary_text = (
            f"Patient {payload.patient_id} recent encounters ({', '.join(encounter_ids)}): "  # noqa: E501
            f"{bullet_points}"
        )

        logger.info(
            "Generated stub summary for patient=%s encounters=%s",
            payload.patient_id,
            encounter_ids,
        )
        return SummaryResult(
            summary_text=summary_text,
            structured_data=None,
            model_version=self.model_version,
            confidence_score=0.6,
        )

    async def _build_structured_timeline(
        self, payload: SummaryPayload, session
    ) -> dict[str, Any]:
        """
        Build structured timeline format with patient info, alerts, and
        timeline entries.
        """
        import sys

        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        from database.models import (
            Encounter,
            FileAsset,
            Patient,
        )

        # Handle both UUID and string types
        patient_id = (
            payload.patient_id
            if isinstance(payload.patient_id, UUID)
            else UUID(str(payload.patient_id))
        )
        encounter_ids = [
            eid if isinstance(eid, UUID) else UUID(str(eid))
            for eid in payload.encounter_ids
        ]

        # Fetch patient
        stmt = select(Patient).where(Patient.id == patient_id)
        result = await session.execute(stmt)
        patient = result.scalar_one_or_none()
        if not patient:
            return {"error": "Patient not found"}

        # Calculate age
        age = None
        if patient.dob:
            today = date.today()
            age = (
                today.year
                - patient.dob.year
                - ((today.month, today.day) < (patient.dob.month, patient.dob.day))
            )

        # Fetch all encounters with related data
        stmt = (
            select(Encounter)
            .where(Encounter.patient_id == patient_id)
            .where(Encounter.id.in_(encounter_ids))
            .options(
                selectinload(Encounter.triage_observations),
                selectinload(Encounter.soap_notes),
                selectinload(Encounter.dialogue_transcripts),
            )
            .order_by(Encounter.arrival_ts.desc())
        )
        result = await session.execute(stmt)
        encounters = result.scalars().all()

        # Fetch all file assets for this patient
        # NOTE: Include file assets even if they're not linked to specific encounters
        # since documents can be uploaded without an encounter
        stmt = (
            select(FileAsset)
            .where(FileAsset.patient_id == patient_id)
            .where(
                (FileAsset.encounter_id.in_(encounter_ids))
                | (FileAsset.encounter_id.is_(None))
            )
            .order_by(FileAsset.created_at.desc())
        )
        result = await session.execute(stmt)
        file_assets = result.scalars().all()

        logger.info(
            "[TIMELINE] Found %d file assets for patient %s",
            len(file_assets),
            patient_id,
        )
        print(
            f"[TIMELINE] Found {len(file_assets)} file assets for patient {patient_id}",
            file=sys.stderr,
            flush=True,
        )

        # Debug: Check extraction_data for each file asset
        for asset in file_assets:
            has_extraction = bool(asset.extraction_data)
            has_raw_text = bool(asset.raw_text)
            logger.info(
                "[TIMELINE] File asset %s: has_extraction_data=%s, has_raw_text=%s, extraction_status=%s",  # noqa: E501
                asset.id,
                has_extraction,
                has_raw_text,
                asset.extraction_status,
            )
            print(
                f"[TIMELINE] File asset {asset.id}: has_extraction_data={has_extraction}, has_raw_text={has_raw_text}, extraction_status={asset.extraction_status}",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )

        # Build timeline entries
        timeline_entries = []
        alerts = {
            "allergies": [],
            "chronic_conditions": [],
            "recent_events": [],
            "warnings": [],
        }

        # Process encounters
        for encounter in encounters:
            try:
                triage = (
                    encounter.triage_observations[0]
                    if encounter.triage_observations
                    and len(encounter.triage_observations) > 0
                    else None
                )
                soap_note = (
                    encounter.soap_notes[0]
                    if encounter.soap_notes and len(encounter.soap_notes) > 0
                    else None
                )
                transcript = (
                    encounter.dialogue_transcripts[0]
                    if encounter.dialogue_transcripts
                    and len(encounter.dialogue_transcripts) > 0
                    else None
                )

                # Determine entry type and source
                entry_type = "visit"
                source_type = "manual_entry"
                confidence = 85.0
                original_file = None

                if soap_note:
                    source_type = "ai_scribe"
                    confidence = (
                        (soap_note.confidence_score * 100)
                        if soap_note.confidence_score
                        else 99.0
                    )
                    if transcript:
                        original_file = f"consultation_audio_{encounter.arrival_ts.strftime('%Y%m%d')}.webm"  # noqa: E501
                elif transcript:
                    source_type = "ai_scribe"
                    confidence = 95.0
                    original_file = f"consultation_audio_{encounter.arrival_ts.strftime('%Y%m%d')}.webm"  # noqa: E501

                # Build entry title
                title = "ED VISIT"
                if triage and triage.chief_complaint:
                    complaint = triage.chief_complaint[:50]
                    title = f"ED VISIT - {complaint.upper()}"
                elif soap_note and soap_note.assessment:
                    assessment = soap_note.assessment[:50]
                    title = f"ED VISIT - {assessment.upper()}"

                # Extract vitals
                vitals = None
                if triage and triage.vitals:
                    v = triage.vitals
                    vitals = {
                        "hr": v.get("hr") or v.get("heart_rate"),
                        "bp": f"{v.get('sbp') or v.get('systolic_bp')}/{v.get('dbp') or v.get('diastolic_bp')}"  # noqa: E501
                        if (v.get("sbp") or v.get("dbp"))
                        else None,
                        "temp": v.get("temp_c") or v.get("temperature"),
                        "rr": v.get("rr") or v.get("respiratory_rate"),
                        "o2": v.get("spo2") or v.get("oxygen_saturation"),
                    }
                    # Remove None values
                    vitals = {k: v for k, v in vitals.items() if v is not None}

                # Extract clinical data
                data: dict[str, Any] = {}
                if triage and triage.chief_complaint:
                    data["chief_complaint"] = triage.chief_complaint
                if vitals:
                    data["vitals"] = vitals
                if triage and triage.notes:
                    data["rfv"] = triage.notes
                if soap_note:
                    if soap_note.subjective:
                        data["subjective"] = soap_note.subjective
                    if soap_note.objective:
                        data["objective"] = soap_note.objective
                    if soap_note.assessment:
                        data["diagnosis"] = soap_note.assessment
                    if soap_note.plan:
                        data["plan"] = soap_note.plan
                if encounter.disposition:
                    data["disposition"] = encounter.disposition

                timeline_entries.append(
                    {
                        "id": str(encounter.id),
                        "date": encounter.arrival_ts.isoformat(),
                        "type": entry_type,
                        "title": title,
                        "source": {
                            "type": source_type,
                            "confidence": confidence,
                            "original_file": original_file,
                        },
                        "data": data,
                        "expanded": False,
                        "can_view_original": bool(original_file),
                    }
                )
            except Exception as e:
                logger.error(f"Error processing encounter {encounter.id}: {e}")
                continue

        # Process file assets as separate timeline entries
        for file_asset in file_assets:
            try:
                # Get extracted data if available
                extracted_data = file_asset.extraction_data or {}

                logger.info(
                    "[TIMELINE] Processing file asset %s: filename=%s, has_extraction_data=%s, extraction_status=%s",  # noqa: E501
                    file_asset.id,
                    file_asset.original_filename,
                    bool(extracted_data),
                    file_asset.extraction_status,
                )
                print(
                    f"[TIMELINE] Processing file asset {file_asset.id}: "
                    f"filename={file_asset.original_filename}, "
                    f"has_extraction_data={bool(extracted_data)}, "
                    f"extraction_status={file_asset.extraction_status}",
                    file=sys.stderr,
                    flush=True,
                )

                # If extraction_data is empty but we have raw_text, try to
                # extract basic info from raw_text
                if not extracted_data and file_asset.raw_text:
                    logger.warning(
                        "[TIMELINE] File asset %s has raw_text (%d chars) but "
                        "no extraction_data. Extracting basic info from raw_text.",
                        file_asset.id,
                        len(file_asset.raw_text),
                    )
                    print(
                        f"[TIMELINE] WARNING: File asset {file_asset.id} has raw_text ({len(file_asset.raw_text)} chars) but no extraction_data. Extracting basic info from raw_text.",  # noqa: E501
                        file=sys.stderr,
                        flush=True,
                    )

                    # Extract basic information from raw_text as fallback
                    raw_text_preview = file_asset.raw_text[
                        :2000
                    ]  # First 2000 chars for processing
                    extracted_data = self._extract_basic_info_from_text(
                        raw_text_preview, file_asset
                    )

                # Determine file type and title from extracted data or file type
                entry_type = "document"
                title = "DOCUMENT"

                visit_metadata = extracted_data.get("visit_metadata", {})
                visit_type = visit_metadata.get("visit_type", "").upper()
                chief_complaint_data = extracted_data.get("chief_complaint", {})
                chief_complaint_text = (
                    chief_complaint_data.get("text", "")
                    if isinstance(chief_complaint_data, dict)
                    else ""
                )

                if visit_type:
                    if "ED" in visit_type or "EMERGENCY" in visit_type:
                        entry_type = "visit"
                        if chief_complaint_text:
                            title = f"ED VISIT - {chief_complaint_text[:50].upper()}"
                        else:
                            title = f"{visit_type}"
                    elif "CLINIC" in visit_type:
                        entry_type = "visit"
                        title = (
                            f"CLINIC VISIT - {chief_complaint_text[:50].upper()}"
                            if chief_complaint_text
                            else visit_type
                        )
                    elif "PROCEDURE" in visit_type:
                        entry_type = "visit"
                        title = (
                            f"PROCEDURE - {chief_complaint_text[:50].upper()}"
                            if chief_complaint_text
                            else visit_type
                        )
                    elif "LAB" in visit_type:
                        entry_type = "lab"
                        title = f"LAB REPORT - {file_asset.document_type or 'MEDICAL REPORT'}"  # noqa: E501
                    else:
                        entry_type = "visit"
                        title = (
                            f"{visit_type} - {chief_complaint_text[:50].upper()}"
                            if chief_complaint_text
                            else visit_type
                        )
                else:
                    # Fallback to content type
                    if file_asset.content_type:
                        if "image" in file_asset.content_type:
                            entry_type = "lab"
                            title = f"LAB REPORT - {file_asset.document_type or 'MEDICAL REPORT'}"  # noqa: E501
                        elif "pdf" in file_asset.content_type:
                            entry_type = "visit"
                            if chief_complaint_text:
                                title = f"MEDICAL RECORD - {chief_complaint_text[:50].upper()}"  # noqa: E501
                            else:
                                title = (
                                    f"{file_asset.document_type or 'MEDICAL RECORD'}"
                                )
                        else:
                            title = f"DOCUMENT - {file_asset.document_type or 'UPLOADED FILE'}"  # noqa: E501
                    else:
                        title = f"DOCUMENT - {file_asset.original_filename or 'UPLOADED FILE'}"  # noqa: E501

                source_type = (
                    "uploaded_pdf"
                    if file_asset.content_type and "pdf" in file_asset.content_type
                    else "uploaded_image"
                )
                confidence = (
                    (file_asset.extraction_confidence * 100)
                    if file_asset.extraction_confidence
                    else (
                        (file_asset.confidence * 100) if file_asset.confidence else 92.0
                    )
                )

                # Build detailed data from extracted information
                data: dict[str, Any] = {
                    "document_type": file_asset.document_type,
                    "file_name": file_asset.original_filename,
                }

                # Add extracted data fields if available
                if extracted_data:
                    # Chief complaint
                    if chief_complaint_text:
                        data["chief_complaint"] = chief_complaint_text
                        if isinstance(
                            chief_complaint_data, dict
                        ) and chief_complaint_data.get("symptom_duration"):
                            data["symptom_duration"] = chief_complaint_data[
                                "symptom_duration"
                            ]

                    # Vital signs
                    vitals_data = extracted_data.get("vital_signs", {})
                    if vitals_data and isinstance(vitals_data, dict):
                        vitals = {}
                        for key, value in vitals_data.items():
                            if isinstance(value, dict) and "value" in value:
                                unit = value.get("unit", "")
                                val = value.get("value")
                                if val is not None:
                                    vitals[key] = f"{val} {unit}" if unit else str(val)
                            elif value is not None and value != "":
                                vitals[key] = str(value)
                        if vitals:
                            data["vitals"] = vitals

                    # Diagnoses
                    diagnoses = extracted_data.get("diagnoses", [])
                    if diagnoses and isinstance(diagnoses, list):
                        diag_list = []
                        for diag in diagnoses:
                            if isinstance(diag, dict):
                                diag_text = diag.get("diagnosis", "")
                                icd_code = diag.get("icd_code", "")
                                status = diag.get("status", "")
                                if diag_text:
                                    diag_str = diag_text
                                    if icd_code:
                                        diag_str += f" ({icd_code})"
                                    if status:
                                        diag_str += f" - {status}"
                                    diag_list.append(diag_str)
                        if diag_list:
                            data["diagnosis"] = ", ".join(diag_list)

                    # Medications
                    medications = extracted_data.get("medications", {})
                    if medications and isinstance(medications, dict):
                        med_list = []
                        for med_type in ["new", "continued", "discontinued"]:
                            meds = medications.get(med_type, [])
                            if meds and isinstance(meds, list):
                                for med in meds:
                                    if isinstance(med, dict):
                                        med_name = med.get("name", "")
                                        dose = med.get("dose", "")
                                        unit = med.get("unit", "")
                                        frequency = med.get("frequency", "")
                                        if med_name:
                                            med_str = med_name
                                            if dose:
                                                med_str += f" {dose}"
                                                if unit:
                                                    med_str += unit
                                            if frequency:
                                                med_str += f" {frequency}"
                                            med_list.append(med_str)
                        if med_list:
                            data["medications"] = ", ".join(med_list)

                    # Assessment
                    assessment = extracted_data.get("assessment", {})
                    if assessment and isinstance(assessment, dict):
                        clinical_impression = assessment.get("clinical_impression", "")
                        if clinical_impression:
                            data["assessment"] = clinical_impression

                    # Plan
                    plan = extracted_data.get("plan", {})
                    if plan and isinstance(plan, dict):
                        disposition = plan.get("disposition", "")
                        follow_up = plan.get("follow_up", "")
                        if disposition:
                            data["disposition"] = disposition
                        if follow_up:
                            data["follow_up"] = follow_up

                    # Visit metadata
                    if visit_metadata:
                        visit_date = visit_metadata.get("visit_date")
                        if visit_date:
                            data["visit_date"] = visit_date
                        hospital_name = visit_metadata.get("hospital_name")
                        if hospital_name:
                            data["hospital"] = hospital_name
                        doctor_name = visit_metadata.get("doctor_name")
                        if doctor_name:
                            data["doctor"] = doctor_name

                # Add raw text preview if available (truncated)
                if file_asset.raw_text and file_asset.raw_text.strip():
                    # Truncate to first 500 chars for preview
                    text_preview = (
                        file_asset.raw_text[:500] + "..."
                        if len(file_asset.raw_text) > 500
                        else file_asset.raw_text
                    )
                    data["text_preview"] = text_preview

                timeline_entries.append(
                    {
                        "id": str(file_asset.id),
                        "date": file_asset.created_at.isoformat(),
                        "type": entry_type,
                        "title": title.upper(),
                        "source": {
                            "type": source_type,
                            "confidence": confidence,
                            "original_file": file_asset.original_filename,
                        },
                        "data": data,
                        "expanded": False,
                        "can_view_original": True,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error processing file asset {file_asset.id}: {e}", exc_info=True
                )
                continue

        # Sort timeline by date (newest first)
        try:
            timeline_entries.sort(key=lambda x: x["date"], reverse=True)
        except Exception as e:
            logger.error(f"Error sorting timeline entries: {e}")
            # Continue with unsorted entries

        # Calculate years of history
        years_of_history = 0
        if timeline_entries:
            try:
                oldest_date_str = timeline_entries[-1]["date"]
                newest_date_str = timeline_entries[0]["date"]

                # Handle both ISO format with and without timezone
                def parse_date(date_str: str) -> datetime:
                    # Remove Z and add timezone if needed
                    if "Z" in date_str:
                        date_str = date_str.replace("Z", "+00:00")
                    # Parse ISO format
                    parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    # Ensure timezone-aware
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
                    return parsed

                oldest_date = parse_date(oldest_date_str)
                newest_date = parse_date(newest_date_str)

                # Ensure both are timezone-aware or both are naive
                if oldest_date.tzinfo is None and newest_date.tzinfo is not None:
                    oldest_date = oldest_date.replace(tzinfo=UTC)
                elif oldest_date.tzinfo is not None and newest_date.tzinfo is None:
                    newest_date = newest_date.replace(tzinfo=UTC)

                years_of_history = (newest_date - oldest_date).days / 365.25
            except Exception as e:
                logger.error(f"Error calculating years of history: {e}", exc_info=True)
                years_of_history = 0

        # Build structured output
        return {
            "patient": {
                "id": str(patient.id),
                "name": f"{patient.first_name} {patient.last_name}",
                "age": age,
                "patient_id": patient.mrn,
            },
            "alerts": alerts,
            "timeline": timeline_entries,
            "total_entries": len(timeline_entries),
            "years_of_history": round(years_of_history, 1),
            "last_updated": datetime.now().isoformat(),
        }

    def _extract_basic_info_from_text(self, text: str, file_asset) -> dict[str, Any]:
        """
        Extract basic information from raw text when extraction_data is not available.
        This is a fallback method that doesn't require Gemini API.

        Args:
            text: Raw text from the document.
            file_asset: File asset object.

        Returns:
            Dictionary with basic extracted information.
        """
        import re

        extracted = {
            "visit_metadata": {
                "visit_type": file_asset.document_type or "Medical Record",
            },
            "chief_complaint": {},
            "vital_signs": {},
            "diagnoses": [],
            "medications": {
                "new": [],
                "continued": [],
                "discontinued": [],
            },
            "assessment": {},
            "plan": {},
        }

        # Try to extract date from text
        date_patterns = [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # MM/DD/YYYY or DD/MM/YYYY
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",  # YYYY/MM/DD
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
            r"\d{1,2},?\s+\d{4}",  # Month DD, YYYY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Try to parse the date
                    match.group(0)
                    # Use file creation date if parsing fails
                    if file_asset.created_at:
                        extracted["visit_metadata"][
                            "visit_date"
                        ] = file_asset.created_at.date().isoformat()
                except Exception:
                    pass
                break

        if not extracted["visit_metadata"].get("visit_date") and file_asset.created_at:
            extracted["visit_metadata"][
                "visit_date"
            ] = file_asset.created_at.date().isoformat()

        # Try to extract vital signs using regex patterns
        vitals_patterns = {
            "pulse": r"(?:pulse|heart\s+rate|hr)[:\s]+(\d+)",
            "systolic_bp": r"(?:bp|blood\s+pressure|systolic)[:\s]+(\d+)[/\-]",
            "diastolic_bp": r"(?:bp|blood\s+pressure|diastolic)[:\s]+\d+[/\-](\d+)",
            "temperature": r"(?:temp|temperature|fever)[:\s]+(\d+\.?\d*)",
            "respiration_rate": r"(?:rr|respiratory\s+rate|respiration)[:\s]+(\d+)",
            "o2_saturation": r"(?:o2|spo2|oxygen|saturation)[:\s]+(\d+)",
        }

        for vital_name, pattern in vitals_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                if vital_name == "pulse":
                    extracted["vital_signs"]["pulse"] = {
                        "value": int(value),
                        "unit": "bpm",
                    }
                elif vital_name == "systolic_bp":
                    extracted["vital_signs"]["systolic_bp"] = {
                        "value": int(value),
                        "unit": "mmHg",
                    }
                elif vital_name == "diastolic_bp":
                    extracted["vital_signs"]["diastolic_bp"] = {
                        "value": int(value),
                        "unit": "mmHg",
                    }
                elif vital_name == "temperature":
                    extracted["vital_signs"]["temperature"] = {
                        "value": float(value),
                        "unit": "C",
                    }
                elif vital_name == "respiration_rate":
                    extracted["vital_signs"]["respiration_rate"] = {
                        "value": int(value),
                        "unit": "breaths/min",
                    }
                elif vital_name == "o2_saturation":
                    extracted["vital_signs"]["o2_saturation"] = {
                        "value": int(value),
                        "unit": "%",
                    }

        # Try to extract diagnosis/common medical terms
        diagnosis_keywords = [
            r"diagnosis[:\s]+([^\n\.]{10,100})",
            r"assessment[:\s]+([^\n\.]{10,100})",
            r"condition[:\s]+([^\n\.]{10,100})",
        ]
        for pattern in diagnosis_keywords:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                diag_text = match.group(1).strip()
                if diag_text and len(diag_text) > 5:
                    extracted["diagnoses"].append(
                        {"diagnosis": diag_text, "status": "Possible"}
                    )
                    break

        # Try to extract chief complaint
        complaint_patterns = [
            r"chief\s+complaint[:\s]+([^\n\.]{10,200})",
            r"cc[:\s]+([^\n\.]{10,200})",
            r"reason\s+for\s+visit[:\s]+([^\n\.]{10,200})",
        ]
        for pattern in complaint_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                complaint = match.group(1).strip()
                if complaint and len(complaint) > 5:
                    extracted["chief_complaint"]["text"] = complaint
                    break

        # If no chief complaint found, use first few lines as summary
        if not extracted["chief_complaint"].get("text"):
            lines = text.split("\n")[:5]
            summary = " ".join(line.strip() for line in lines if line.strip())[:200]
            if summary:
                extracted["chief_complaint"]["text"] = summary

        # Add text preview to assessment if no assessment found
        if not extracted["assessment"].get("clinical_impression"):
            preview = text[:500].replace("\n", " ").strip()
            if preview:
                extracted["assessment"]["clinical_impression"] = preview + (
                    "..." if len(text) > 500 else ""
                )

        return extracted
