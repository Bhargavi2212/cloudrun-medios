"""
Document processing service for file assets.
"""

from __future__ import annotations

import logging
from datetime import UTC
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from database.crud import AsyncCRUDRepository
from database.models import FileAsset, TimelineEvent
from services.summarizer_agent.core.document_processor import (
    DocumentProcessor,
    extract_text_from_pdf,
)

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Handles document processing and timeline event creation.
    """

    def __init__(
        self,
        session: AsyncSession,
        document_processor: DocumentProcessor,
        storage_root: Path | str,
    ) -> None:
        """
        Initialize document service.

        Args:
            session: Database session.
            document_processor: Document processor engine.
            storage_root: Root directory for file storage.
        """
        self.session = session
        self.document_processor = document_processor
        self.storage_root = Path(storage_root)
        self.file_repository = AsyncCRUDRepository[FileAsset](session, FileAsset)
        self.timeline_repository = AsyncCRUDRepository[TimelineEvent](
            session, TimelineEvent
        )

    async def get_file_asset(self, file_id: UUID) -> FileAsset | None:
        """Get file asset by ID."""
        return await self.file_repository.get(file_id)

    async def process_document(self, file_id: UUID) -> dict[str, Any]:
        """
        Process a document through the multi-step pipeline.

        Args:
            file_id: File asset ID.

        Returns:
            Processing result dictionary.
        """
        # Get file asset
        asset = await self.get_file_asset(file_id)
        if asset is None:
            raise ValueError(f"File asset {file_id} not found")

        # Get file path
        file_path = self.storage_root / asset.storage_path
        if not file_path.exists():
            raise ValueError(f"File not found at {file_path}")

        # Extract text from file
        raw_text = ""
        if asset.content_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_path)
        elif asset.raw_text:
            raw_text = asset.raw_text
        else:
            # For images, we'd need OCR here (Google Vision API)
            # For now, assume text is already extracted
            raw_text = asset.raw_text or ""

        if not raw_text:
            raise ValueError("No text extracted from document")

        # Process document
        metadata = {
            "filename": asset.original_filename,
            "document_type": asset.document_type,
        }
        result = await self.document_processor.process_document(raw_text, metadata)

        # Update file asset
        update_data = {
            "raw_text": result.cleaned_text or raw_text,
            "extraction_status": "completed" if result.success else "failed",
            "extraction_confidence": result.overall_confidence,
            "extraction_data": result.deidentified_data or result.extracted_data,
            "processing_metadata": {
                "step1_cleaned": result.cleaned_text is not None,
                "step2_extracted": result.extracted_data is not None,
                "step3_deidentified": result.deidentified_data is not None,
                "step4_scored": result.confidence_scores is not None,
                "errors": result.errors or [],
                "warnings": result.warnings or [],
            },
        }

        # Determine confidence tier
        if result.overall_confidence >= 0.9:
            update_data["confidence_tier"] = "high"
            update_data["needs_manual_review"] = False
        elif result.overall_confidence >= 0.75:
            update_data["confidence_tier"] = "medium"
            update_data["needs_manual_review"] = False
        elif result.overall_confidence >= 0.5:
            update_data["confidence_tier"] = "low"
            update_data["needs_manual_review"] = True
            update_data["status"] = "needs_review"
        else:
            update_data["confidence_tier"] = "very_low"
            update_data["needs_manual_review"] = True
            update_data["status"] = "needs_review"

        if result.success and result.overall_confidence >= 0.75:
            update_data["status"] = "processed"
        elif not result.success:
            update_data["status"] = "failed"
            update_data["last_error"] = "; ".join(result.errors or [])

        updated_asset = await self.file_repository.update(asset, update_data)
        await self.session.commit()
        await self.session.refresh(updated_asset)

        # Create timeline event if processing was successful
        timeline_event = None
        if result.success and result.deidentified_data and asset.patient_id:
            timeline_event = await self._create_timeline_event(
                asset=updated_asset,
                extracted_data=result.deidentified_data,
                confidence=result.overall_confidence,
            )

        return {
            "file_id": str(file_id),
            "success": result.success,
            "overall_confidence": result.overall_confidence,
            "confidence_tier": update_data.get("confidence_tier"),
            "needs_review": update_data.get("needs_manual_review", False),
            "timeline_event_id": str(timeline_event.id) if timeline_event else None,
            "errors": result.errors or [],
            "warnings": result.warnings or [],
        }

    async def _create_timeline_event(
        self,
        asset: FileAsset,
        extracted_data: dict,
        confidence: float,
    ) -> TimelineEvent:
        """
        Create timeline event from extracted document data.

        Args:
            asset: File asset.
            extracted_data: De-identified extracted data.
            confidence: Overall confidence score.

        Returns:
            Created timeline event.
        """
        visit_metadata = extracted_data.get("visit_metadata", {})
        visit_date_str = visit_metadata.get("visit_date")
        # Parse date or use current date
        from datetime import datetime

        try:
            if visit_date_str:
                event_date = datetime.fromisoformat(
                    visit_date_str.replace("Z", "+00:00")
                )
            else:
                event_date = datetime.now(UTC)
        except Exception:
            event_date = datetime.now(UTC)

        # Determine event type
        visit_type = visit_metadata.get("visit_type", "Unknown")
        event_type = "ed_visit" if "ED" in visit_type.upper() else "clinic_visit"

        # Create title and summary
        chief_complaint = extracted_data.get("chief_complaint", {}).get(
            "text", "No chief complaint"
        )
        title = f"{visit_type}: {chief_complaint[:50]}"

        summary_parts = []
        if chief_complaint:
            summary_parts.append(f"Chief Complaint: {chief_complaint}")
        diagnoses = extracted_data.get("diagnoses", [])
        if diagnoses:
            diag_list = [
                d.get("diagnosis", "") for d in diagnoses if d.get("diagnosis")
            ]
            if diag_list:
                summary_parts.append(f"Diagnoses: {', '.join(diag_list)}")
        plan = extracted_data.get("plan", {})
        if plan.get("disposition"):
            summary_parts.append(f"Disposition: {plan['disposition']}")

        summary = (
            ". ".join(summary_parts)
            if summary_parts
            else "Document processed and data extracted."
        )

        payload = {
            "patient_id": asset.patient_id,
            "encounter_id": asset.encounter_id,
            "source_type": "uploaded_document",
            "source_file_asset_id": asset.id,
            "event_type": event_type,
            "event_date": event_date,
            "title": title,
            "summary": summary,
            "data": extracted_data,
            "status": "completed" if confidence >= 0.75 else "pending",
            "confidence": confidence,
            "extraction_confidence": confidence,
            "extraction_metadata": {
                "document_type": asset.document_type,
                "upload_method": asset.upload_method,
            },
        }

        event = await self.timeline_repository.create(payload)
        await self.session.commit()
        await self.session.refresh(event)
        return event
