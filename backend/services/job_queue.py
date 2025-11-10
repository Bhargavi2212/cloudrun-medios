"""Asynchronous job queue service backed by the job_queue table."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..database import crud
from ..database.models import JobQueue
from ..database.session import get_session
from .config import get_settings
from .document_processing import DocumentProcessingService
from .make_agent import MakeAgentService
from .storage import StorageError, StorageService
from .telemetry import record_service_metric


class JobQueueService:
    def __init__(
        self,
        *,
        make_agent_service: Optional[MakeAgentService] = None,
        storage_service: Optional[StorageService] = None,
        document_processing_service: Optional[DocumentProcessingService] = None,
    ) -> None:
        self.make_agent_service = make_agent_service or MakeAgentService()
        self.storage_service = storage_service or StorageService()
        self.document_processing_service = (
            document_processing_service or DocumentProcessingService()
        )

    async def enqueue_audio_processing(self, audio_id: str) -> JobQueue:
        payload = {"audio_id": audio_id}
        with get_session() as session:
            job = crud.create_job(
                session, task_type="scribe.audio_pipeline", payload=payload
            )
            session.refresh(job)
        asyncio.create_task(self._process_job(job.id))
        return job

    async def enqueue_document_processing(
        self,
        file_id: str,
        *,
        patient_id: str,
        consultation_id: Optional[str] = None,
    ) -> JobQueue:
        payload = {
            "file_id": file_id,
            "patient_id": patient_id,
            "consultation_id": consultation_id,
        }
        with get_session() as session:
            job = crud.create_job(
                session, task_type="documents.process", payload=payload
            )
            session.refresh(job)
        asyncio.create_task(self._process_job(job.id))
        return job

    async def enqueue_retention_cleanup(self, batch_size: int = 100) -> JobQueue:
        """Enqueue a retention cleanup job.

        Args:
            batch_size: Maximum number of files to process in one run

        Returns:
            JobQueue instance
        """
        payload = {"batch_size": batch_size}
        with get_session() as session:
            job = crud.create_job(
                session, task_type="storage.retention_cleanup", payload=payload
            )
            session.refresh(job)
        asyncio.create_task(self._process_job(job.id))
        return job

    def get_job(self, job_id: str) -> Optional[JobQueue]:
        with get_session() as session:
            return crud.get_job_by_id(session, job_id)

    async def _process_job(self, job_id: str) -> None:
        start_time = datetime.now(timezone.utc)
        with get_session() as session:
            job = crud.get_job_by_id(session, job_id)
            if not job:
                return
            job = crud.update_job(
                session,
                job,
                status="in_progress",
                attempts=job.attempts + 1,
                started_at=start_time,
            )
            payload = dict(job.payload or {})
            task_type = job.task_type

        try:
            if task_type == "scribe.audio_pipeline":
                audio_id = payload.get("audio_id")
                if not audio_id:
                    raise ValueError("audio_id missing from job payload")

                audio_file = self.storage_service.get_audio_file(audio_id)
                if not audio_file:
                    raise StorageError(f"Audio file {audio_id} not found")

                file_path = self.storage_service.resolve_file_path(audio_file)
                result = await self.make_agent_service.process_audio_pipeline(
                    file_path,
                    consultation_id=audio_file.consultation_id,
                    author_id=audio_file.uploaded_by,
                )
                # Handle both dict and model responses
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = result if isinstance(result, dict) else {}

                # Determine success from result
                success = result_dict.get("success", True)
                if not success:
                    # Check for errors
                    errors = result_dict.get("errors", [])
                    if errors:
                        success = False
                    elif result_dict.get("stage_completed") == "failed":
                        success = False

                payload["result"] = result_dict
                status = "completed" if success else "failed"
                error_message = (
                    result_dict.get("error")
                    or "; ".join(result_dict.get("errors", []))
                    or None
                )
                metric_service = "scribe"
            elif task_type == "documents.process":
                file_id = payload.get("file_id")
                patient_id = payload.get("patient_id")
                if not file_id or not patient_id:
                    raise ValueError(
                        "file_id and patient_id required for document processing jobs"
                    )
                result = await self.document_processing_service.process_document(
                    file_id,
                    patient_id=patient_id,
                    consultation_id=payload.get("consultation_id"),
                )
                payload["result"] = result.to_dict()
                success = result.success
                status = "completed" if success else "failed"
                error_message = "; ".join(result.errors) if result.errors else None
                metric_service = "documents"
            elif task_type == "storage.retention_cleanup":
                # Retention cleanup job
                batch_size = payload.get("batch_size", 100)
                cleanup_stats = self.storage_service.cleanup_expired_files(
                    batch_size=batch_size
                )
                payload["result"] = cleanup_stats
                success = True
                status = "completed"
                error_message = None
                metric_service = "storage"
            else:
                raise ValueError(f"Unknown job task_type '{task_type}'")
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = str(exc)
            payload.setdefault("errors", []).append(error_message)
            metric_service = "job_queue"
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        if status == "failed" and error_message:
            errors_list = payload.setdefault("errors", [])
            if error_message not in errors_list:
                errors_list.append(error_message)

        with get_session() as session:
            job = crud.get_job_by_id(session, job_id)
            if not job:
                return
            crud.update_job(
                session,
                job,
                status=status,
                completed_at=end_time,
                error_message=error_message if status == "failed" else None,
                payload=payload,
            )

        record_service_metric(
            service_name=metric_service,
            metric_name="async_pipeline_seconds",
            metric_value=duration,
            metadata={"job_id": job_id, "status": status},
        )
