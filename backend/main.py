from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api import api_router
from backend.api.error_handlers import register_exception_handlers
from backend.services.config import get_settings
from backend.services.error_response import StandardResponse
from backend.services.model_manager import initialize_models
from backend.services.logging import configure_logging, get_logger
from backend.services.middleware import AccessLogMiddleware, RequestIDMiddleware

settings = get_settings()
configure_logging(settings.app_env)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure models are ready on startup."""
    success, error = await asyncio.to_thread(initialize_models)
    if not success:
        logger.error("Model initialisation failed: %s", error)
    yield


app = FastAPI(title="MediOS AI Scribe", version="2.0.0", lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

register_exception_handlers(app)
app.include_router(api_router)


@app.get("/", response_model=StandardResponse)
async def root() -> StandardResponse:
    return StandardResponse(success=True, data={"message": "MediOS AI Scribe v2"})


@app.get("/health", response_model=StandardResponse)
async def health() -> StandardResponse:
    return StandardResponse(success=True, data={"status": "healthy"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

