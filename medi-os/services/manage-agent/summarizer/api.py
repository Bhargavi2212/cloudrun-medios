"""
FastAPI application exposing the summariser service.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from summarizer.config import Settings, load_settings
from summarizer.errors import PatientNotFoundError, SummarizerError
from summarizer.metrics import MetricsRecorder
from summarizer.summarizer import SummarizerService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = FastAPI(title="AI Medical Summarizer", version="0.1.0")

try:
    _settings = load_settings()
    service_dependency = SummarizerService(
        settings=_settings,
        metrics=MetricsRecorder(slow_threshold=_settings.slow_request_threshold),
    )
except SummarizerError as exc:
    logger.error("Failed to initialise summarizer service: %s", exc)
    raise


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/patients")
def list_patients(
    limit: int = Query(20, ge=1, le=200),
) -> dict:
    try:
        patient_ids = service_dependency.loader.list_patient_ids(limit=limit)
        return {"patients": patient_ids}
    except SummarizerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/summarize/{subject_id}")
def summarize_patient(
    subject_id: int,
    visit_limit: Optional[int] = Query(None, ge=1, le=200),
) -> dict:
    try:
        result = service_dependency.summarize(subject_id, visit_limit=visit_limit)
        return {
            "subject_id": result.subject_id,
            "summary_markdown": result.markdown_summary,
            "timeline": result.structured_timeline,
            "metrics": result.metrics,
        }
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SummarizerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict:
    try:
        return service_dependency.metrics.snapshot()
    except SummarizerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = ["app"]

