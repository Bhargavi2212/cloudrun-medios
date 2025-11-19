"""
Document processing endpoints.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_session
from services.summarizer_agent.config import SummarizerAgentSettings
from services.summarizer_agent.core.document_processor import DocumentProcessor
from services.summarizer_agent.dependencies import get_document_processor, get_settings
from services.summarizer_agent.services.document_service import DocumentService

router = APIRouter(prefix="/summarizer", tags=["documents"])


def get_document_service(
    session: AsyncSession = Depends(get_session),
    processor: DocumentProcessor = Depends(get_document_processor),
    settings: SummarizerAgentSettings = Depends(get_settings),
) -> DocumentService:
    """Dependency to get document service."""
    return DocumentService(session, processor, settings.storage_root)


@router.post(
    "/documents/{file_id}/process",
    status_code=status.HTTP_200_OK,
    summary="Process document",
    description="Process an uploaded document through the multi-step Gemini pipeline.",
)
async def process_document(
    file_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> dict:
    """
    Process a document through the multi-step pipeline.

    Args:
        file_id: File asset ID.
        document_service: Document service.

    Returns:
        Processing result.
    """
    try:
        result = await document_service.process_document(file_id)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {e!s}",
        ) from e
