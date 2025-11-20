"""
FastAPI application factory utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from fastapi import APIRouter, FastAPI
from starlette.middleware.cors import CORSMiddleware

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

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for router in routers:
        app.include_router(router)

    if enable_database:
        from database.session import dispose_engine, init_engine

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
