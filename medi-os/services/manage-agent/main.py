"""
Manage Agent FastAPI Application.

The Orchestrator service that handles patient triage and prioritization.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Manage Agent",
    description="Patient triage and orchestration service",
    version="1.0.0",
)


@app.get("/")
async def root() -> dict:
    """
    Root endpoint to check service status.

    Returns:
        dict: Status confirmation message.
    """
    return {"status": "ok"}


@app.get("/health")
async def health() -> JSONResponse:
    """
    Health check endpoint.

    Returns:
        JSONResponse: HTTP 200 status code.
    """
    return JSONResponse(content={"status": "healthy"}, status_code=200)

