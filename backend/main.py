from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api import api_router
from backend.api.error_handlers import register_exception_handlers
from backend.database.init import initialize_demo_data
from backend.services.config import get_settings
from backend.services.error_response import StandardResponse
from backend.services.logging import configure_logging, get_logger
from backend.services.middleware import AccessLogMiddleware, RequestIDMiddleware
from backend.services.model_manager import initialize_models

settings = get_settings()
configure_logging(settings.app_env)
logger = get_logger(__name__)
allowed_origins = settings.cors_allow_origins or [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models in background, don't block startup."""
    logger.info("Starting MediOS AI Scribe - models will initialize in background")

    # Start initialization tasks in background (don't await them)
    asyncio.create_task(initialize_database_background())
    asyncio.create_task(initialize_models_background())

    yield
    logger.info("Shutting down MediOS AI Scribe")


async def initialize_models_background():
    """Initialize models in background without blocking startup."""
    try:
        logger.info("Starting background model initialization...")
        success, error = await asyncio.to_thread(initialize_models)
        if not success:
            logger.error("Model initialization failed: %s", error)
        else:
            logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Unexpected error during model initialization: {e}")


async def initialize_database_background():
    """Seed roles and demo users without blocking startup."""
    try:
        logger.info("Starting background database initialization...")
        await asyncio.to_thread(initialize_demo_data)
        logger.info("Database initialization complete")
    except Exception as exc:
        logger.error("Unexpected error during database initialization: %s", exc)


app = FastAPI(title="MediOS AI Scribe", version="2.0.0", lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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

    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
