"""Document processing pipeline for uploaded records and timeline enrichment."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

from ..database import crud
from ..database.models import (
    DocumentProcessingStatus,
    FileAsset,
    TimelineEventStatus,
    TimelineEventType,
)
from ..database.session import get_session
from .ai_models import AIModelsService
from .notifier import NotificationService, notification_service as default_notification_service
from .storage import StorageService

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.65

DATE_PATTERNS = [
    r"\b(\d{4}-\d{2}-\d{2})\b",
    r"\b(\d{2}/\d{2}/\d{4})\b",
    r"\b(\d{2}-\d{2}-\d{4})\b",
]


class DocumentProcessingError(Exception):
    """Raised when a document cannot be processed."""


@dataclass
class DocumentExtraction:
    text: str
    pages: List[str]
    page_count: int
    content_type: str
    document_type: str
    confidence: float
    is_handwritten: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class TimelineEntryPayload:
    title: str
    summary: str
    event_type: TimelineEventType
    event_date: datetime
    status: TimelineEventStatus
    confidence: float
    data: Dict[str, object] = field(default_factory=dict)


@dataclass
class DocumentProcessingResult:
    success: bool
    status: DocumentProcessingStatus
    confidence: float
    summary: str
    timeline_event_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    needs_review: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "status": self.status.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "timeline_event_ids": self.timeline_event_ids,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
            "needs_review": self.needs_review,
        }


class DocumentProcessingService:
    """Processes uploaded documents into structured timeline events and summaries."""

    def __init__(
        self,
        *,
        session_factory: Optional[Any] = None,
        storage_service: Optional[StorageService] = None,
        ai_models: Optional[AIModelsService] = None,
        notification_service: Optional[NotificationService] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize DocumentProcessingService.
        
        Args:
            session_factory: Optional session factory for testing. If not provided,
                           uses the default get_session dependency.
            storage_service: Optional storage service. If not provided, creates a new one.
            ai_models: Optional AI models service. If not provided, creates a new one.
            notification_service: Optional notification service. If not provided, uses default.
            confidence_threshold: Confidence threshold for document processing.
        """
        self._session_factory = session_factory or get_session
        self.storage_service = storage_service or StorageService()
        self.ai_models = ai_models or AIModelsService()
        self.notification_service = notification_service or default_notification_service
        self.confidence_threshold = confidence_threshold

    async def process_document(
        self,
        file_id: str,
        *,
        patient_id: Optional[str] = None,
        consultation_id: Optional[str] = None,
    ) -> DocumentProcessingResult:
        errors: List[str] = []
        warnings: List[str] = []
        resolved_patient_id = patient_id
        resolved_consultation_id = consultation_id
        try:
            with self._session_factory() as session:
                asset = crud.get_file_asset(session, file_id)
                if asset is None:
                    raise DocumentProcessingError(f"File asset {file_id} not found.")

                resolved_patient_id, resolved_consultation_id = self._resolve_patient_consultation(
                    session,
                    asset,
                    patient_id,
                    consultation_id,
                )

                crud.update_file_asset(
                    session,
                    asset,
                    status=DocumentProcessingStatus.PROCESSING,
                    processed_at=None,
                    processing_notes=None,
                    last_error=None,
                )

                file_path = self.storage_service.resolve_file_asset_path(asset)
                content_type = asset.content_type or self._guess_content_type(asset)
                original_filename = asset.original_filename or os.path.basename(file_path)

            extraction = await self._extract_document(file_path, content_type, original_filename)
            warnings.extend(extraction.warnings)

            summarizer_metadata = {
                "filename": original_filename,
                "document_type": extraction.document_type,
                "page_count": extraction.page_count,
            }
            summary_result = await self.ai_models.summarize_document(extraction.text, summarizer_metadata)
            summary_text = summary_result.get("summary", "").strip()
            highlights = summary_result.get("highlights", [])
            summary_confidence = float(summary_result.get("confidence") or 0.0)
            if summary_result.get("warning"):
                warnings.append(str(summary_result["warning"]))
            if not summary_result.get("success"):
                warnings.append("Document summarizer reported failure; marking for review.")

            overall_confidence = min(summary_confidence, extraction.confidence)
            needs_review = (
                overall_confidence < self.confidence_threshold
                or extraction.is_handwritten
                or summary_result.get("is_stub", False)
                or not summary_result.get("success", True)
            )

            file_status = (
                DocumentProcessingStatus.NEEDS_REVIEW if needs_review else DocumentProcessingStatus.COMPLETED
            )
            event_status = (
                TimelineEventStatus.NEEDS_REVIEW if needs_review else TimelineEventStatus.COMPLETED
            )

            timeline_entries = self._build_timeline_entries(
                file_id,
                summary_text,
                highlights,
                event_status,
                overall_confidence,
                extraction,
                original_filename,
            )

            metadata_payload = {
                "file_id": file_id,
                "patient_id": resolved_patient_id,
                "consultation_id": resolved_consultation_id,
                "page_count": extraction.page_count,
                "document_type": extraction.document_type,
                "extraction_confidence": extraction.confidence,
                "summary_confidence": summary_confidence,
                "highlights": highlights,
            }
            if warnings:
                metadata_payload["warnings"] = warnings
            processing_notes = "; ".join(warnings) if warnings else None

            timeline_event_ids: List[str] = []
            processed_at = datetime.now(timezone.utc)

            with self._session_factory() as session:
                asset = crud.get_file_asset(session, file_id)
                if asset is None:
                    raise DocumentProcessingError("File asset disappeared before completion.")
                updated_asset = crud.update_file_asset(
                    session,
                    asset,
                    status=file_status,
                    confidence=overall_confidence,
                    document_type=extraction.document_type,
                    processing_metadata=metadata_payload,
                    processing_notes=processing_notes,
                    processed_at=processed_at,
                    last_error=None,
                )
                session.flush()

                # Always create at least one timeline event for the document
                if not timeline_entries:
                    # Fallback: create a basic timeline event even if no highlights were extracted
                    base_date = datetime.now(timezone.utc)
                    timeline_entries = [
                        TimelineEntryPayload(
                            title=f"Document uploaded: {original_filename}",
                            summary=summary_text or "Document uploaded and processed.",
                            event_type=TimelineEventType.DOCUMENT,
                            event_date=base_date,
                            status=event_status,
                            confidence=overall_confidence,
                            data={
                                "document_id": file_id,
                                "source_filename": original_filename,
                                "page_count": extraction.page_count,
                                "auto_generated": True,
                            },
                        )
                    ]

                for entry in timeline_entries:
                    event = crud.create_timeline_event(
                        session,
                        patient_id=resolved_patient_id,
                        consultation_id=resolved_consultation_id,
                        source_file_id=updated_asset.id,
                        event_type=entry.event_type,
                        event_date=entry.event_date,
                        title=entry.title,
                        summary=entry.summary,
                        data=entry.data,
                        status=entry.status,
                        confidence=entry.confidence,
                    )
                    session.flush()
                    timeline_event_ids.append(event.id)
                
                # Session will commit on context exit

            self._publish_update(
                file_id=file_id,
                patient_id=resolved_patient_id,
                consultation_id=resolved_consultation_id,
                status=file_status,
                processed_at=processed_at,
                timeline_event_ids=timeline_event_ids,
                confidence=overall_confidence,
                needs_review=needs_review,
                summary=summary_text or "",
                metadata=metadata_payload,
            )

            return DocumentProcessingResult(
                success=True,
                status=file_status,
                confidence=overall_confidence,
                summary=summary_text or "Document processed but no summary generated.",
                timeline_event_ids=timeline_event_ids,
                metadata=metadata_payload,
                errors=errors,
                warnings=warnings,
                needs_review=needs_review,
            )

        except DocumentProcessingError as exc:
            logger.error("Document processing failed: %s", exc)
            errors.append(str(exc))
            self._publish_failure(
                file_id=file_id,
                patient_id=resolved_patient_id,
                consultation_id=resolved_consultation_id,
                error=str(exc),
            )
            await self._mark_failed(file_id, errors[-1])
            return DocumentProcessingResult(
                success=False,
                status=DocumentProcessingStatus.FAILED,
                confidence=0.0,
                summary="",
                errors=errors,
                warnings=warnings,
                needs_review=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected document processing failure for %s", file_id)
            errors.append(str(exc))
            self._publish_failure(
                file_id=file_id,
                patient_id=resolved_patient_id,
                consultation_id=resolved_consultation_id,
                error=str(exc),
            )
            await self._mark_failed(file_id, str(exc))
            return DocumentProcessingResult(
                success=False,
                status=DocumentProcessingStatus.FAILED,
                confidence=0.0,
                summary="",
                errors=errors,
                warnings=warnings,
                needs_review=True,
            )

    async def _mark_failed(self, file_id: str, error: str) -> None:
        with get_session() as session:
            asset = crud.get_file_asset(session, file_id)
            if asset is None:
                return
            
            # Resolve patient and consultation context
            resolved_patient_id, resolved_consultation_id = self._resolve_patient_consultation(
                session,
                asset,
                None,
                None,
            )
            
            crud.update_file_asset(
                session,
                asset,
                status=DocumentProcessingStatus.FAILED,
                processing_notes=error,
                processed_at=datetime.now(timezone.utc),
                last_error=error,
            )
            session.flush()
            
            # Create a timeline event even when processing fails
            # so the document appears in the patient summary
            if resolved_patient_id:
                try:
                    base_date = datetime.now(timezone.utc)
                    event = crud.create_timeline_event(
                        session,
                        patient_id=resolved_patient_id,
                        consultation_id=resolved_consultation_id,
                        source_file_id=asset.id,
                        event_type=TimelineEventType.DOCUMENT,
                        event_date=base_date,
                        title=f"Document uploaded: {asset.original_filename or 'Unknown'}",
                        summary=f"Document uploaded but processing failed: {error}",
                        data={
                            "document_id": file_id,
                            "source_filename": asset.original_filename,
                            "processing_failed": True,
                            "error": error,
                        },
                        status=TimelineEventStatus.NEEDS_REVIEW,
                        confidence=0.0,
                    )
                    session.flush()
                    session.commit()
                except Exception as exc:
                    logger.warning("Failed to create timeline event for failed document %s: %s", file_id, exc)
                    session.rollback()

    def _publish_update(
        self,
        *,
        file_id: str,
        patient_id: Optional[str],
        consultation_id: Optional[str],
        status: DocumentProcessingStatus,
        processed_at: datetime,
        timeline_event_ids: List[str],
        confidence: float,
        needs_review: bool,
        summary: str,
        metadata: Dict[str, object],
    ) -> None:
        if not patient_id:
            return
        event_payload = {
            "file_id": file_id,
            "patient_id": patient_id,
            "consultation_id": consultation_id,
            "status": status.value,
            "processed_at": processed_at.isoformat(),
            "timeline_event_ids": timeline_event_ids,
            "confidence": confidence,
            "needs_review": needs_review,
            "summary": summary,
            "metadata": metadata,
        }
        for channel in self._channels(patient_id, consultation_id):
            self.notification_service.publish(channel, event_payload)

    def _publish_failure(
        self,
        *,
        file_id: str,
        patient_id: Optional[str],
        consultation_id: Optional[str],
        error: str,
    ) -> None:
        if not patient_id:
            return
        event_payload = {
            "file_id": file_id,
            "patient_id": patient_id,
            "consultation_id": consultation_id,
            "status": DocumentProcessingStatus.FAILED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        for channel in self._channels(patient_id, consultation_id):
            self.notification_service.publish(channel, event_payload)

    @staticmethod
    def _channels(patient_id: str, consultation_id: Optional[str]) -> List[str]:
        channels = [f"documents:patient:{patient_id}"]
        if consultation_id:
            channels.append(f"documents:consultation:{consultation_id}")
        return channels

    def _resolve_patient_consultation(
        self,
        session,
        asset: FileAsset,
        patient_id: Optional[str],
        consultation_id: Optional[str],
    ) -> tuple[str, Optional[str]]:
        resolved_consultation_id = consultation_id
        resolved_patient_id = patient_id

        if asset.owner_type == "consultation":
            resolved_consultation_id = resolved_consultation_id or asset.owner_id
            consultation = (
                crud.get_consultation(session, resolved_consultation_id) if resolved_consultation_id else None
            )
            if consultation:
                resolved_patient_id = consultation.patient_id
        elif asset.owner_type == "patient":
            resolved_patient_id = resolved_patient_id or asset.owner_id

        if not resolved_patient_id:
            raise DocumentProcessingError(
                f"Unable to determine patient context for file {asset.id} (owner_type={asset.owner_type}).",
            )

        return resolved_patient_id, resolved_consultation_id

    async def _extract_document(
        self,
        file_path: str,
        content_type: Optional[str],
        filename: str,
    ) -> DocumentExtraction:
        if self._is_pdf(file_path, content_type):
            if PdfReader is None:
                raise DocumentProcessingError(
                    "PDF support requires the 'pypdf' package. Install it and retry.",
                )
            return await asyncio.to_thread(self._extract_pdf_text, file_path, filename)

        if self._is_text(content_type, filename):
            return await asyncio.to_thread(self._extract_text_file, file_path, content_type or "")

        # Fallback for images or unknown formats
        warnings = [
            "Document appears to be an image or unsupported format. OCR required; flagging for manual review."
        ]
        return DocumentExtraction(
            text="",
            pages=[""],
            page_count=1,
            content_type=content_type or "application/octet-stream",
            document_type="image",
            confidence=0.2,
            is_handwritten=True,
            warnings=warnings,
        )

    def _extract_pdf_text(self, file_path: str, filename: str) -> DocumentExtraction:
        reader = PdfReader(file_path)
        pages: List[str] = []
        warnings: List[str] = []
        for index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - pypdf runtime issues
                logger.warning("Failed to extract text from %s page %s: %s", filename, index + 1, exc)
                page_text = ""
                warnings.append(f"Page {index + 1}: text extraction failed.")
            pages.append(page_text.strip())
        combined_text = "\n\n".join(filter(None, pages))
        confidence = 0.9 if combined_text else 0.3
        is_handwritten = self._detect_handwritten(pages)
        if is_handwritten:
            warnings.append("Low text density detected; likely handwritten content.")
            confidence = min(confidence, 0.4)
        return DocumentExtraction(
            text=combined_text,
            pages=pages,
            page_count=len(pages),
            content_type="application/pdf",
            document_type="pdf",
            confidence=confidence,
            is_handwritten=is_handwritten,
            warnings=warnings,
        )

    def _extract_text_file(self, file_path: str, content_type: str) -> DocumentExtraction:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        confidence = 0.7 if text.strip() else 0.1
        return DocumentExtraction(
            text=text,
            pages=[text],
            page_count=1,
            content_type=content_type or "text/plain",
            document_type="text",
            confidence=confidence,
            warnings=[],
        )

    def _build_timeline_entries(
        self,
        file_id: str,
        summary_text: str,
        highlights: List[str],
        status: TimelineEventStatus,
        confidence: float,
        extraction: DocumentExtraction,
        filename: str,
    ) -> List[TimelineEntryPayload]:
        entries: List[TimelineEntryPayload] = []
        base_date = datetime.now(timezone.utc)

        if highlights:
            for index, highlight in enumerate(highlights[:5]):
                event_date = self._find_date(highlight) or base_date
                title = self._title_from_text(highlight)
                entries.append(
                    TimelineEntryPayload(
                        title=title,
                        summary=highlight,
                        event_type=TimelineEventType.DOCUMENT,
                        event_date=event_date,
                        status=status,
                        confidence=confidence,
                        data={
                            "document_id": file_id,
                            "page_index": index,
                            "source_filename": filename,
                            "page_count": extraction.page_count,
                        },
                    )
                )
        else:
            fallback_summary = summary_text or "Document requires manual review; no automated summary available."
            entries.append(
                TimelineEntryPayload(
                    title=self._title_from_text(fallback_summary),
                    summary=fallback_summary,
                    event_type=TimelineEventType.DOCUMENT,
                    event_date=base_date,
                    status=status,
                    confidence=confidence,
                    data={
                        "document_id": file_id,
                        "source_filename": filename,
                        "page_count": extraction.page_count,
                        "auto_generated": True,
                    },
                )
            )

        return entries

    @staticmethod
    def _detect_handwritten(pages: List[str]) -> bool:
        if not pages:
            return True
        short_pages = sum(1 for page in pages if len(page.strip()) < 50)
        return short_pages / max(len(pages), 1) >= 0.5

    @staticmethod
    def _find_date(text: str) -> Optional[datetime]:
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, text)
            if not match:
                continue
            date_str = match.group(1)
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _title_from_text(text: str) -> str:
        words = text.split()
        if not words:
            return "Document Update"
        return " ".join(words[:8]).rstrip(",.;:") or "Document Update"

    @staticmethod
    def _guess_content_type(asset: FileAsset) -> str:
        filename = asset.original_filename or ""
        extension = os.path.splitext(filename)[1].lower()
        if extension == ".pdf":
            return "application/pdf"
        if extension in {".txt", ".log"}:
            return "text/plain"
        if extension in {".rtf"}:
            return "application/rtf"
        if extension in {".png", ".jpg", ".jpeg", ".heic"}:
            return f"image/{extension.strip('.')}"
        return "application/octet-stream"

    @staticmethod
    def _is_pdf(file_path: str, content_type: Optional[str]) -> bool:
        return file_path.lower().endswith(".pdf") or (content_type or "").lower() == "application/pdf"

    @staticmethod
    def _is_text(content_type: Optional[str], filename: str) -> bool:
        if content_type and content_type.startswith("text/"):
            return True
        return filename.lower().endswith((".txt", ".md", ".csv"))

