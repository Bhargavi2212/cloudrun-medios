"""
AI Scribe Agent FastAPI Application.

The Documenter service that handles audio transcription and SOAP note generation.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI Scribe Agent",
    description="Audio transcription and SOAP note generation service",
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

