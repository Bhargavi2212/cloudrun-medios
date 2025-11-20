"""
FastAPI application factory utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from fastapi import APIRouter, FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from database.session import dispose_engine, init_engine
from shared.logging import configure_logging


class ServiceSettings(Protocol):
    """
    Protocol describing the configuration required by the app factory.
    """

    environment: str
    debug: bool
    log_level: str
    database_url: str
    cors_allow_origins: list[str]


def create_service_app(
    *,
    service_name: str,
    version: str,
    settings: ServiceSettings,
    routers: Iterable[APIRouter],
    enable_database: bool = True,
) -> FastAPI:
    """
    Build a FastAPI application with shared middleware and logging.

    Args:
        service_name: Name of the service (used in logging).
        version: Semantic version string.
        settings: Configuration instance implementing ServiceSettings.
        routers: Iterable of APIRouter instances to include.
        enable_database: Whether to initialise the shared async database engine.
    """

    configure_logging(service_name, settings.log_level)
    app = FastAPI(
        title=f"Medi OS {service_name}",
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Debug CORS configuration
    import logging

    cors_logger = logging.getLogger("shared.fastapi")
    cors_logger.info(f"Setting up CORS with origins: {settings.cors_allow_origins}")

    if settings.cors_allow_origins:
        # Check if wildcard is in the list
        allow_all_origins = "*" in settings.cors_allow_origins

        # Force CORS headers on ALL responses, including errors
        class ForceCORSMiddleware(BaseHTTPMiddleware):
            """Middleware to force CORS headers on all responses."""

            async def dispatch(self, request: Request, call_next):
                origin = request.headers.get("origin")
                should_allow = allow_all_origins or (
                    origin and origin in settings.cors_allow_origins
                )

                if should_allow:
                    try:
                        response = await call_next(request)
                        # Force CORS headers on response
                        if allow_all_origins:
                            response.headers["Access-Control-Allow-Origin"] = "*"
                        elif origin:
                            response.headers["Access-Control-Allow-Origin"] = origin
                        response.headers["Access-Control-Allow-Credentials"] = "true"
                        response.headers[
                            "Access-Control-Allow-Methods"
                        ] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                        response.headers["Access-Control-Allow-Headers"] = "*"
                        return response
                    except Exception as e:
                        # Force CORS headers even on exceptions
                        import logging

                        from fastapi.responses import JSONResponse

                        logger = logging.getLogger("shared.fastapi")
                        logger.error("Exception in request: %s", e, exc_info=True)
                        response = JSONResponse(
                            status_code=500,
                            content={"detail": "Internal server error"},
                        )
                        if allow_all_origins:
                            response.headers["Access-Control-Allow-Origin"] = "*"
                        elif origin:
                            response.headers["Access-Control-Allow-Origin"] = origin
                        response.headers["Access-Control-Allow-Credentials"] = "true"
                        response.headers[
                            "Access-Control-Allow-Methods"
                        ] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                        response.headers["Access-Control-Allow-Headers"] = "*"
                        return response
                else:
                    return await call_next(request)

        # Add force CORS middleware FIRST (outermost)
        app.add_middleware(ForceCORSMiddleware)

        # Add standard CORS middleware for preflight requests
        # If wildcard is present, use it; otherwise use the list
        cors_origins = ["*"] if allow_all_origins else settings.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        cors_logger.info(
            f"CORS middleware added with origins: {settings.cors_allow_origins} "
            f"(allow_all={allow_all_origins})"
        )
    else:
        cors_logger.warning("CORS middleware NOT added - no origins configured")

    for router in routers:
        app.include_router(router)

    # Add exception handlers AFTER routers to ensure they catch all exceptions
    if settings.cors_allow_origins:
        from fastapi.responses import JSONResponse
        from starlette.exceptions import HTTPException

        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Ensure CORS headers are added to HTTP exception responses."""
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )
            # Add CORS headers manually
            origin = request.headers.get("origin")
            allow_all = "*" in settings.cors_allow_origins
            if allow_all:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
            elif origin and origin in settings.cors_allow_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Ensure CORS headers are added to general exception responses."""
            import logging

            logger = logging.getLogger("shared.fastapi")
            logger.error("Unhandled exception: %s", exc, exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
            # Add CORS headers manually
            origin = request.headers.get("origin")
            allow_all = "*" in settings.cors_allow_origins
            if allow_all:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
            elif origin and origin in settings.cors_allow_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
            return response

    if enable_database:

        @app.on_event("startup")
        async def _startup() -> None:
            init_engine(
                database_url=settings.database_url,
                echo=settings.debug,
            )

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            await dispose_engine()

    return app
