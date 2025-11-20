"""
SOAP note generation with Gemini AI integration.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Protocol

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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
    AI-powered SOAP note generation using Gemini.
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

        logger.info(
            "Initializing ScribeEngine: api_key=%s, genai=%s",
            bool(gemini_api_key),
            genai is not None,
        )

        # FORCE GEMINI CHECK
        if not gemini_api_key:
            logger.warning(
                "⚠️ GEMINI_API_KEY is missing! Scribe will use STUB generator."
            )
        elif genai is None:
            logger.error(
                "❌ google-generativeai package not found, but API key provided!"
            )

        if gemini_api_key and genai is not None:
            try:
                genai.configure(api_key=gemini_api_key)
                self._model = genai.GenerativeModel(gemini_model)
                self._enabled = True
                logger.info("✅ ScribeEngine initialized with Gemini %s", gemini_model)
            except Exception as e:
                logger.error("Failed to initialize Gemini model: %s", e, exc_info=True)
                self._enabled = False
        else:
            # Only fall back if truly necessary, but log loudly
            if not gemini_api_key:
                logger.warning("Gemini API key not provided, using stub SOAP generator")
            if genai is None:
                logger.warning(
                    "google-generativeai not installed, using stub SOAP generator"
                )

    async def generate(
        self, payload: DialoguePayload, context: dict | None = None
    ) -> SoapGenerationResult:
        """
        Produce a SOAP note from dialogue text using Gemini AI.
        Falls back to stub if Gemini is unavailable.
        """
        if not self._enabled or self._model is None:
            logger.error(
                "❌ Gemini not enabled (_enabled=%s, _model=%s), using stub. API key present: %s. "  # noqa: E501
                "Please set GEMINI_API_KEY environment variable!",
                self._enabled,
                self._model is not None,
                bool(self.gemini_api_key),
            )
            return self._stub_generate(payload)

        transcript = payload.transcript.strip()
        logger.info(
            "Received transcript (length: %d chars): %s...",
            len(transcript),
            transcript[:100],
        )

        if not transcript:
            logger.error("Empty transcript provided, using stub")
            return self._stub_generate(payload)

        # Build prompt with context
        prompt = self._build_soap_prompt(transcript, context)
        logger.debug("Built prompt (length: %d chars)", len(prompt))

        try:
            # Run Gemini API call in thread pool to avoid blocking event loop
            def _run_gemini() -> str:
                try:
                    # Debug logging for API key
                    if not self.gemini_api_key:
                        logger.error("❌ Attempting to call Gemini without API key!")
                        raise ValueError("Gemini API key is missing")

                    response = self._model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.7,
                            "max_output_tokens": 4000,  # Increased to prevent
                            # truncation
                        },
                    )
                    return (response.text or "").strip()
                except Exception as e:
                    logger.error("Gemini API call failed: %s", e, exc_info=True)
                    raise

            logger.info("Calling Gemini API to generate SOAP note...")
            soap_text = await asyncio.to_thread(_run_gemini)
            if not soap_text:
                logger.error("Gemini returned empty SOAP note, falling back to stub")
                return self._stub_generate(payload)

            # Log the raw response for debugging
            logger.info(
                "✅ Gemini SOAP response received (length: %d chars)", len(soap_text)
            )
            logger.info("Gemini SOAP response preview: %s", soap_text[:300])

            # Parse the response into SOAP sections
            soap_sections = self._parse_soap_response(soap_text)

            logger.info(
                "✅ Generated AI SOAP note for encounter=%s using model=%s",
                payload.encounter_id,
                f"{self.model_version}+gemini",
            )
            return SoapGenerationResult(
                subjective=soap_sections.get("subjective", ""),
                objective=soap_sections.get("objective", ""),
                assessment=soap_sections.get("assessment", ""),
                plan=soap_sections.get("plan", ""),
                model_version=f"{self.model_version}+gemini",
            )
        except Exception as e:
            logger.error("Gemini SOAP generation failed: %s", e, exc_info=True)
            logger.error(
                "Falling back to stub generator. Check GEMINI_API_KEY and network connectivity."  # noqa: E501
            )
            return self._stub_generate(payload)

    def _build_soap_prompt(self, transcript: str, context: dict | None = None) -> str:
        """
        Build the prompt for Gemini to generate a SOAP note.
        """
        context_text = ""
        if context:
            if context.get("vitals"):
                vitals = context["vitals"]
                vitals_list = [
                    f"{k}: {v}"
                    for k, v in vitals.items()
                    if v and k not in ["ambulance_arrival", "seen_72h", "injury"]
                ]
                if vitals_list:
                    context_text += f"\nPatient Vitals: {', '.join(vitals_list)}\n"
            if context.get("chief_complaint"):
                context_text += f"\nChief Complaint: {context['chief_complaint']}\n"
            if context.get("age"):
                context_text += f"\nPatient Age: {context['age']}\n"

        prompt = f"""Act as an expert Medical Scribe and Clinical Documentation
Specialist.

Task: Convert the following patient-provider conversation transcript into a
formal medical SOAP note.

{context_text if context_text else ""}

Consultation Transcript:
{transcript}

Guidelines:

**Subjective:**
Extract the Chief Complaint (CC), History of Present Illness (HPI), relevant
Medical/Family/Social History, and Review of Systems (ROS). Use medical
terminology (e.g., convert "trouble breathing" to "dyspnea", "chest pain" to
"chest pain" or "angina" if appropriate).

**Objective:**
Only list vital signs or physical exam findings explicitly mentioned in the
transcript or context. If none are provided, state "Not mentioned in
transcript." Do not hallucinate or invent values.

**Assessment:**
Summarize the primary diagnosis or differential diagnoses discussed. Be
specific and clinically relevant.

**Plan:**
List diagnostic tests ordered, lifestyle modifications, medications prescribed
(if any), and follow-up instructions. Be specific and actionable.

**Tone:**
Professional, concise, and clinical. Remove all pleasantries, small talk, and
unrelated conversation.

**Format:**
Use clear section headers. Format your response EXACTLY as follows:

SUBJECTIVE:

[Chief Complaint, HPI, Medical/Family/Social History, ROS - use medical terminology]

OBJECTIVE:

[Vital signs and physical exam findings - only if mentioned, otherwise state
"Not mentioned in transcript"]

ASSESSMENT:

[Primary diagnosis or differential diagnoses]

PLAN:

[Diagnostic tests, medications, lifestyle modifications, follow-up instructions]

Now generate the complete SOAP note:"""

        return prompt

    def _parse_soap_response(self, response_text: str) -> dict[str, str]:
        """
        Parse Gemini's response into SOAP sections.
        Handles various formats including detailed subsections.
        """
        sections = {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": "",
        }

        # Normalize the response text
        text = response_text.strip()

        # Try to extract sections using regex with more flexible patterns
        patterns = {
            "subjective": r"(?i)^SUBJECTIVE:?\s*\n(.*?)(?=\n\s*(?:OBJECTIVE|ASSESSMENT|PLAN):|$)",  # noqa: E501
            "objective": r"(?i)^OBJECTIVE:?\s*\n(.*?)(?=\n\s*(?:ASSESSMENT|PLAN|SUBJECTIVE):|$)",  # noqa: E501
            "assessment": r"(?i)^ASSESSMENT:?\s*\n(.*?)(?=\n\s*(?:PLAN|SUBJECTIVE|OBJECTIVE):|$)",  # noqa: E501
            "plan": r"(?i)^PLAN:?\s*\n(.*?)(?=\n\s*(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT):|$)",  # noqa: E501
        }

        # First try: regex with DOTALL to match across newlines
        for section_name, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                sections[section_name] = match.group(1).strip()
                continue

        # Second try: line-by-line parsing (more flexible)
        if not all(sections.values()):
            lines = text.split("\n")
            current_section = None
            section_lines = []

            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    if current_section:
                        section_lines.append("")
                    continue

                # Check if this line starts a new section
                section_match = re.match(
                    r"^(?i)(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN):?\s*$", line_stripped
                )
                if section_match:
                    # Save previous section
                    if current_section and section_lines:
                        sections[current_section.lower()] = "\n".join(
                            section_lines
                        ).strip()
                    # Start new section
                    current_section = section_match.group(1).upper()
                    section_lines = []
                elif current_section:
                    # Continue current section
                    section_lines.append(line)
                else:
                    # Text before any section header - might be metadata, skip it
                    pass

            # Save last section
            if current_section and section_lines:
                sections[current_section.lower()] = "\n".join(section_lines).strip()

        # Third try: simple keyword-based extraction
        if not all(sections.values()):
            for section_name in ["subjective", "objective", "assessment", "plan"]:
                if sections[section_name]:
                    continue

                # Look for section header (case insensitive)
                pattern = rf"(?i)(?:^|\n)\s*{section_name.upper()}:?\s*\n(.*?)(?=\n\s*(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN):|$)"  # noqa: E501
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    sections[section_name] = match.group(1).strip()

        # Clean up sections - remove extra whitespace but preserve structure
        for key in sections:
            if sections[key]:
                # Remove leading/trailing whitespace from each line,
                # but keep line breaks
                lines = [line.strip() for line in sections[key].split("\n")]
                sections[key] = "\n".join(lines).strip()

        # Log parsing results for debugging
        if not all(sections.values()):
            logger.warning(
                "Some SOAP sections were not parsed. Found: %s",
                [k for k, v in sections.items() if v],
            )
            logger.debug("Response text (first 1000 chars): %s", text[:1000])

        return sections

    def _stub_generate(self, payload: DialoguePayload) -> SoapGenerationResult:
        """
        Fallback stub SOAP generation when Gemini is unavailable.
        Uses pattern matching and heuristics to extract meaningful information
        from transcript.
        """
        transcript = payload.transcript.strip()

        if not transcript:
            logger.warning("Empty transcript in stub generator")
            return SoapGenerationResult(
                subjective="No transcript available.",
                objective="Not documented.",
                assessment="Unable to assess without transcript.",
                plan="Document encounter details.",
                model_version=self.model_version,
            )

        # Extract subjective (patient complaints/symptoms)
        subjective_parts = []

        # Look for common complaint patterns
        complaint_patterns = [
            r"(?:I've|I have|I'm|I am|patient states?|patient reports?|"
            r"patient complains?).*?(?:pain|ache|discomfort|shortness of "
            r"breath|dyspnea|chest pain|difficulty breathing|cough|fever|"
            r"nausea|vomiting|dizziness|headache|fatigue|weakness)[^.]*",
            r"(?:experiencing|having|feeling|suffering from).*?[^.]*",
        ]

        for pattern in complaint_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE | re.DOTALL)
            for match in matches:
                complaint = match.group(0).strip()
                # Clean up the complaint
                complaint = re.sub(
                    r"^(?:Good morning|Hello|Hi|How are you)[,.]?\s*",
                    "",
                    complaint,
                    flags=re.IGNORECASE,
                ).strip()
                if len(complaint) > 20 and complaint not in subjective_parts:
                    subjective_parts.append(complaint[:200])  # Limit length

        # If no specific complaints found, use first few sentences
        if not subjective_parts:
            sentences = re.split(r"[.!?]+", transcript)
            for sentence in sentences[:3]:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    # Remove greetings
                    sentence = re.sub(
                        r"^(?:Good morning|Hello|Hi|How are you)[,.]?\s*",
                        "",
                        sentence,
                        flags=re.IGNORECASE,
                    ).strip()
                    if sentence:
                        subjective_parts.append(sentence[:200])
                        break

        # Join subjective parts
        subjective = (
            " ".join(subjective_parts) if subjective_parts else transcript[:200]
        )
        subjective = (
            f"Patient states: {subjective}"
            if not subjective.lower().startswith("patient")
            else subjective
        )

        # Extract objective (vital signs, physical exam)
        objective_parts = []

        # Look for vital signs mentioned
        vitals_patterns = {
            "pulse": r"(?:pulse|heart rate|hr)[:\s]+(\d+)",
            "bp": r"(?:blood pressure|bp)[:\s]+(\d+/\d+)",
            "temp": r"(?:temperature|temp|fever)[:\s]+(\d+\.?\d*)",
            "rr": r"(?:respiratory rate|rr|breathing rate)[:\s]+(\d+)",
            "o2": r"(?:oxygen|o2|spo2|saturation)[:\s]+(\d+)",
        }

        vitals_found = []
        for vital_name, pattern in vitals_patterns.items():
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                vitals_found.append(f"{vital_name.upper()}: {match.group(1)}")

        if vitals_found:
            objective_parts.append("Vitals: " + ", ".join(vitals_found))
        else:
            objective_parts.append("Vitals: See triage observations for details.")

        # Look for physical exam findings
        exam_patterns = [
            r"(?:exam|examination|assessment|findings?)[:\s]+([^.!?]{20,200})",
            r"(?:observed|noted|found)[:\s]+([^.!?]{20,200})",
        ]

        exam_findings = []
        for pattern in exam_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                finding = match.group(1).strip()
                if finding and len(finding) > 10:
                    exam_findings.append(finding[:150])

        if exam_findings:
            objective_parts.append("Physical exam: " + "; ".join(exam_findings[:2]))
        else:
            objective_parts.append("Physical exam: Not detailed in transcript.")

        objective = ". ".join(objective_parts)

        # Extract assessment (diagnosis/differential)
        assessment_parts = []

        # Look for diagnostic terms
        diagnosis_patterns = [
            r"(?:diagnosis|likely|suspected|possible|rule out|differential)[:\s]+([^.!?]{10,150})",  # noqa: E501
            r"(?:appears to be|suggests?|consistent with|indicates?)[:\s]+([^.!?]{10,150})",  # noqa: E501
        ]

        for pattern in diagnosis_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                diag = match.group(1).strip()
                if diag and len(diag) > 5:
                    assessment_parts.append(diag[:150])

        # If no specific diagnosis found, infer from symptoms
        if not assessment_parts:
            if re.search(r"chest pain|angina|cardiac", transcript, re.IGNORECASE):
                assessment_parts.append(
                    "Chest pain - cardiac vs non-cardiac causes to be evaluated"
                )
            elif re.search(
                r"shortness of breath|dyspnea|difficulty breathing",
                transcript,
                re.IGNORECASE,
            ):
                assessment_parts.append("Dyspnea - requires further evaluation")
            elif re.search(r"fever|infection", transcript, re.IGNORECASE):
                assessment_parts.append(
                    "Possible infectious process - requires evaluation"
                )
            else:
                assessment_parts.append(
                    "Provisional diagnosis pending further evaluation"
                )

        assessment = (
            "; ".join(assessment_parts[:2])
            if assessment_parts
            else "Provisional diagnosis pending further evaluation."
        )

        # Extract plan (tests, medications, follow-up)
        plan_parts = []

        # Look for ordered tests
        test_patterns = [
            r"(?:order|ordered|request|schedule|perform)[:\s]+(?:a |an |the )?([^.!?]{10,100})",  # noqa: E501
            r"(?:test|lab|labs|imaging|x-ray|ct|mri|ekg|ecg|blood work)[^.!?]*",
        ]

        tests_mentioned = []
        for pattern in test_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                test = (
                    match.group(1).strip() if match.groups() else match.group(0).strip()
                )
                if test and len(test) > 5:
                    tests_mentioned.append(test[:100])

        if tests_mentioned:
            plan_parts.append("Order: " + ", ".join(tests_mentioned[:3]))

        # Look for medications
        med_patterns = [
            r"(?:prescribe|prescription|medication|med|start|take|given)[:\s]+([^.!?]{10,100})",
            r"(?:mg|mcg|tablet|capsule|dose)[^.!?]*",
        ]

        meds_mentioned = []
        for pattern in med_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                med = (
                    match.group(1).strip() if match.groups() else match.group(0).strip()
                )
                if med and len(med) > 5:
                    meds_mentioned.append(med[:100])

        if meds_mentioned:
            plan_parts.append("Medications: " + ", ".join(meds_mentioned[:2]))

        # Look for follow-up
        followup_patterns = [
            r"(?:follow up|follow-up|return|come back|recheck)[:\s]+([^.!?]{10,100})",
            r"(?:in \d+ (?:days?|weeks?|months?))",
        ]

        followup_mentioned = []
        for pattern in followup_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                followup = (
                    match.group(1).strip() if match.groups() else match.group(0).strip()
                )
                if followup:
                    followup_mentioned.append(followup[:100])

        if followup_mentioned:
            plan_parts.append("Follow-up: " + ", ".join(followup_mentioned[:2]))

        # Default plan if nothing found
        if not plan_parts:
            plan_parts.append("Document encounter")
            plan_parts.append("Review triage observations")
            plan_parts.append("Consider further diagnostic workup as indicated")

        plan = ". ".join(plan_parts) + "."

        result = SoapGenerationResult(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            model_version=self.model_version,
        )
        logger.warning(
            "⚠️ Generated STUB SOAP note for encounter=%s using model=%s. This means Gemini was NOT used!",  # noqa: E501
            payload.encounter_id,
            self.model_version,
        )
        logger.warning(
            "⚠️ Check logs above for why Gemini failed (empty transcript, API error, etc.)"  # noqa: E501
        )
        logger.info(
            "Stub SOAP note generated with extracted information from transcript (length: %d chars)",  # noqa: E501
            len(transcript),
        )
        return result
