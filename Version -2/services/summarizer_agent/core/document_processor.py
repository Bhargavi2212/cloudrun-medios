"""
Document processing engine with multi-step Gemini pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessingResult:
    """Result of document processing."""

    success: bool
    cleaned_text: str | None = None
    extracted_data: dict[str, Any] | None = None
    deidentified_data: dict[str, Any] | None = None
    confidence_scores: dict[str, float] | None = None
    overall_confidence: float = 0.0
    errors: list[str] | None = None
    warnings: list[str] | None = None


class DocumentProcessor:
    """
    Multi-step document processing engine using Gemini.
    """

    def __init__(
        self, *, api_key: str | None = None, model_name: str = "gemini-2.0-flash-exp"
    ) -> None:
        """
        Initialize document processor.

        Args:
            api_key: Google Gemini API key.
            model_name: Gemini model name to use.
        """
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

        if genai is None:
            logger.warning(
                "google-generativeai not installed, document processing will be disabled"  # noqa: E501
            )
            return

        if api_key:
            genai.configure(api_key=api_key)
            try:
                self._model = genai.GenerativeModel(model_name)
                logger.info("Document processor initialized with model: %s", model_name)
            except Exception as e:
                logger.error("Failed to initialize Gemini model: %s", e)
                self._model = None
        else:
            logger.warning(
                "Gemini API key not provided, document processing will be disabled"
            )

    def _call_gemini(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Call Gemini API with prompt.

        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction.

        Returns:
            Response text from Gemini.
        """
        if self._model is None:
            raise RuntimeError("Gemini model not initialized")

        try:
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            # Combine system instruction with prompt if provided
            # (some Gemini API versions don't support system_instruction parameter)
            final_prompt = prompt
            if system_instruction:
                final_prompt = f"{system_instruction}\n\n{prompt}"

                response = self._model.generate_content(
                    final_prompt,
                    generation_config=generation_config,
                )

            return response.text
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            raise

    async def process_document(
        self, raw_text: str, metadata: dict[str, Any] | None = None
    ) -> DocumentProcessingResult:
        """
        Process document through 4-step Gemini pipeline.

        Args:
            raw_text: Raw extracted text from OCR/PDF.
            metadata: Optional metadata (filename, document_type, etc.).

        Returns:
            DocumentProcessingResult with all processing steps.
        """
        metadata = metadata or {}
        result = DocumentProcessingResult(
            success=False,
            errors=[],
            warnings=[],
        )

        if self._model is None:
            result.errors.append("Gemini model not initialized")
            return result

        try:
            # Step 1: Text cleaning & normalization
            cleaned_text = await self._step1_clean_text(raw_text, metadata)
            if not cleaned_text:
                result.errors.append("Step 1 (text cleaning) failed")
                return result

            # Step 2: Medical data extraction
            extracted_data = await self._step2_extract_data(cleaned_text)
            if not extracted_data:
                result.errors.append("Step 2 (data extraction) failed")
                return result

            # Step 3: De-identification
            deidentified_data = await self._step3_deidentify(extracted_data)
            if not deidentified_data:
                result.warnings.append(
                    "Step 3 (de-identification) failed, using original data"
                )
                deidentified_data = extracted_data

            # Step 4: Confidence scoring
            confidence_result = await self._step4_score_confidence(
                deidentified_data, cleaned_text, raw_text
            )
            if not confidence_result:
                result.warnings.append("Step 4 (confidence scoring) failed")
                confidence_result = {"overall_confidence": 0.5, "field_scores": {}}

            result.success = True
            result.cleaned_text = cleaned_text
            result.extracted_data = extracted_data
            result.deidentified_data = deidentified_data
            result.confidence_scores = confidence_result.get("field_scores", {})
            result.overall_confidence = confidence_result.get("overall_confidence", 0.0)

            return result

        except Exception as e:
            logger.exception("Document processing failed: %s", e)
            result.errors.append(str(e))
            return result

    async def _step1_clean_text(
        self, raw_text: str, metadata: dict[str, Any]
    ) -> str | None:
        """Step 1: Clean and normalize text."""
        system_instruction = (
            """You are a medical document text cleaning expert. Clean OCR/PDF """
            """text, fix errors, identify sections, and normalize formatting. """
            """Return only the cleaned text, no JSON."""
        )
        prompt = f"""Clean this medical document text:

{raw_text[:5000]}

Instructions:
1. Fix OCR errors (recognize medical terms)
2. Identify document structure (dates, vitals, tests, medications)
3. Remove headers, footers, page numbers, stamps
4. Normalize formatting (consistent line breaks, spacing)
5. Mark unclear sections with [UNCLEAR: description] if confidence <70%

Return the cleaned text only."""

        try:
            cleaned = self._call_gemini(prompt, system_instruction)
            return cleaned.strip()
        except Exception as e:
            logger.error("Step 1 failed: %s", e)
            return None

    async def _step2_extract_data(self, cleaned_text: str) -> dict[str, Any] | None:
        """Step 2: Extract structured medical data."""
        system_instruction = (
            """You are a medical data extraction expert. Extract structured """
            """medical information from clinical documents. Return valid JSON """
            """only."""
        )
        prompt = f"""Extract medical data from this cleaned text:

{cleaned_text[:5000]}

Return JSON in this exact format:
{{
  "visit_metadata": {{
    "visit_date": "YYYY-MM-DD",
    "visit_type": "ED Visit | Clinic | Procedure | Lab",
    "hospital_name": "[Hospital Name]",
    "doctor_name": "[Doctor Name]"
  }},
  "chief_complaint": {{
    "text": "Patient's main reason for visit",
    "symptom_duration": "2 hours | 2 weeks | etc"
  }},
  "vital_signs": {{
    "pulse": {{"value": 110, "unit": "bpm"}},
    "systolic_bp": {{"value": 145, "unit": "mmHg"}},
    "diastolic_bp": {{"value": 95, "unit": "mmHg"}},
    "temperature": {{"value": 37.2, "unit": "C"}},
    "respiration_rate": {{"value": 24, "unit": "breaths/min"}},
    "o2_saturation": {{"value": 94, "unit": "%"}},
    "pain_score": {{"value": 8, "unit": "0-10"}}
  }},
  "diagnoses": [
    {{"diagnosis": "Unstable Angina", "icd_code": "I20.0",
      "status": "Primary | Rule out | Possible"}}
  ],
  "medications": {{
    "new": [{{"name": "Aspirin", "dose": "325", "unit": "mg",
              "frequency": "1x daily"}}],
    "continued": [],
    "discontinued": []
  }},
  "assessment": {{
    "clinical_impression": "Patient likely has unstable angina"
  }},
  "plan": {{
    "disposition": "Discharged",
    "follow_up": "Cardiology in 1 week"
  }}
}}

Return only valid JSON, no markdown formatting."""

        try:
            response = self._call_gemini(prompt, system_instruction)
            # Try to extract JSON from response
            json_text = response.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            extracted = json.loads(json_text)
            return extracted
        except Exception as e:
            logger.error("Step 2 failed: %s", e)
            return None

    async def _step3_deidentify(
        self, extracted_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Step 3: De-identify PHI."""
        system_instruction = """You are a PHI de-identification expert. Remove all personally identifiable information from medical data. Return valid JSON only."""  # noqa: E501
        prompt = f"""De-identify this medical data by replacing PHI with placeholders:

{json.dumps(extracted_data, indent=2)}

Replace:
- Hospital names → [Hospital]
- Doctor/staff names → [Doctor]
- Patient names → [Patient]
- Patient IDs/MRN → [ID]
- Phone numbers → [Phone]
- Specific locations → [Location]

Keep all clinical data (dates, diagnoses, medications, vitals, etc.).

Return the same JSON structure with PHI replaced."""

        try:
            response = self._call_gemini(prompt, system_instruction)
            json_text = response.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            deidentified = json.loads(json_text)
            return deidentified
        except Exception as e:
            logger.error("Step 3 failed: %s", e)
            return None

    async def _step4_score_confidence(
        self, deidentified_data: dict[str, Any], cleaned_text: str, raw_text: str
    ) -> dict[str, Any] | None:
        """Step 4: Score confidence for each field."""
        system_instruction = """You are a medical data quality expert. Score confidence (0-100%) for extracted fields. Return valid JSON only."""  # noqa: E501
        prompt = f"""Score confidence for each extracted field:

De-identified Data:
{json.dumps(deidentified_data, indent=2)[:2000]}

For each field, estimate confidence:
- 95-100%: Clear, unambiguous
- 85-95%: Clear, minor uncertainty
- 70-85%: Reasonably clear, some ambiguity
- 50-70%: Unclear, needs review
- <50%: Very unclear, should not use

Return JSON:
{{
  "overall_confidence": 92,
  "field_scores": {{
    "pulse": {{"value": 110, "confidence": 98}},
    "diagnosis": {{"value": "Unstable Angina", "confidence": 72}}
  }},
  "flags": ["Diagnosis marked as 'possible' - flag for confirmation"]
}}

Return only valid JSON."""

        try:
            response = self._call_gemini(prompt, system_instruction)
            json_text = response.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            scored = json.loads(json_text)
            return scored
        except Exception as e:
            logger.error("Step 4 failed: %s", e)
            return None


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file.

    Args:
        file_path: Path to PDF file.

    Returns:
        Extracted text.
    """
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        # pdfplumber not available, try PyPDF2
        try:
            import PyPDF2

            text_parts = []
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            logger.error(
                "PDF extraction failed: Neither pdfplumber nor PyPDF2 is installed"
            )
            logger.error(
                "Please install one of: poetry add pdfplumber  OR  poetry add PyPDF2"
            )
            return ""
        except Exception as e:
            logger.error("PDF extraction failed with PyPDF2: %s", e)
            return ""
    except Exception as e:
        logger.error("PDF extraction failed: %s", e)
        return ""
