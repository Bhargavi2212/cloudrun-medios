"""
AI Summarizer Agent FastAPI Application.

The Historian service that handles medical record summarization.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="AI Summarizer Agent",
    description="Medical record summarization service",
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

