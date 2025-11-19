"""
Application entry point for the manage-agent service.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from database.session import get_session_factory, init_engine
from federation.client import FederationClient
from services.manage_agent.config import ManageAgentSettings
from services.manage_agent.core.nurse_triage import NurseTriageEngine
from services.manage_agent.core.triage import TriageEngine
from services.manage_agent.handlers import api_router
from services.manage_agent.services.check_in_service import CheckInService
from services.manage_agent.services.federated_scheduler import FederatedTrainingLoop
from services.manage_agent.services.federated_sync_service import FederatedSyncService
from services.manage_agent.services.federated_trainer import FederatedTrainer
from services.manage_agent.services.orchestrator_client import OrchestratorClient
from shared.config import get_settings
from shared.fastapi import create_service_app

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests for debugging."""

    async def dispatch(self, request: Request, call_next):
        import sys

        # Log request details
        client_host = request.client.host if request.client else "unknown"
        query_params = str(request.query_params) if request.query_params else ""
        print(
            f"[REQUEST] INCOMING: {request.method} {request.url.path} from {client_host} {query_params}",  # noqa: E501
            file=sys.stderr,
            flush=True,
        )
        logger.info(
            "[REQUEST] Incoming request: %s %s from %s %s",
            request.method,
            request.url.path,
            client_host,
            query_params,
        )

        # Log request body for POST/PUT/PATCH (but not for file uploads)
        if request.method in [
            "POST",
            "PUT",
            "PATCH",
        ] and "multipart/form-data" not in request.headers.get("content-type", ""):
            try:
                body = await request.body()
                if body:
                    body_str = body.decode("utf-8", errors="ignore")[
                        :500
                    ]  # Limit to 500 chars
                    print(
                        f"[REQUEST] Request body: {body_str}",
                        file=sys.stderr,
                        flush=True,
                    )
                    logger.debug("Request body: %s", body_str)

                # Re-create request with body for downstream handlers
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive
            except Exception as e:
                logger.warning("Could not read request body: %s", e)

        try:
            response = await call_next(request)
            print(
                f"[REQUEST] RESPONSE: {request.method} {request.url.path} - Status {response.status_code}",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )
            logger.info(
                "[REQUEST] Response: %s %s - Status %d",
                request.method,
                request.url.path,
                response.status_code,
            )
            return response
        except Exception as e:
            print(
                f"[REQUEST] ERROR in request: {request.method} {request.url.path} - {e}",  # noqa: E501
                file=sys.stderr,
                flush=True,
            )
            logger.error(
                "[REQUEST] Error in request: %s %s - %s",
                request.method,
                request.url.path,
                e,
                exc_info=True,
            )
            raise


def create_app(settings: ManageAgentSettings | None = None) -> FastAPI:
    """
    Build and configure the FastAPI application.
    """

    loaded_settings = settings or get_settings(ManageAgentSettings)

    # Debug: Log CORS configuration
    logger.info(f"CORS origins configured: {loaded_settings.cors_allow_origins}")

    app = create_service_app(
        service_name="manage-agent",
        version=loaded_settings.version,
        settings=loaded_settings,
        routers=[api_router],
        enable_database=True,
    )

    # Add request logging middleware to see all incoming requests
    app.add_middleware(RequestLoggingMiddleware)
    app.state.settings = loaded_settings
    app.state.triage_engine = TriageEngine(model_version=loaded_settings.model_version)
    app.state.nurse_triage_engine = NurseTriageEngine()
    app.state.check_in_service = CheckInService(
        base_url=loaded_settings.dol_base_url,
        shared_secret=loaded_settings.dol_shared_secret,
        hospital_id=loaded_settings.dol_hospital_id,
    )
    orchestrator_client: OrchestratorClient | None = None
    if (
        loaded_settings.orchestrator_base_url
        and loaded_settings.orchestrator_shared_secret
    ):
        orchestrator_client = OrchestratorClient(
            base_url=loaded_settings.orchestrator_base_url,
            shared_secret=loaded_settings.orchestrator_shared_secret,
            hospital_id=loaded_settings.dol_hospital_id,
        )
        app.state.orchestrator_client = orchestrator_client

        if loaded_settings.manage_public_url:
            registration_payload: dict[str, Any] = {
                "hospital_id": loaded_settings.dol_hospital_id,
                "name": loaded_settings.hospital_name,
                "manage_url": loaded_settings.manage_public_url,
                "scribe_url": loaded_settings.scribe_public_url,
                "summarizer_url": loaded_settings.summarizer_public_url,
                "dol_url": loaded_settings.dol_public_url,
                "capabilities": ["triage", "documents"],
            }

            @app.on_event("startup")
            async def register_with_orchestrator() -> None:
                await orchestrator_client.register(registration_payload)

        @app.on_event("shutdown")
        async def close_orchestrator_client() -> None:
            await orchestrator_client.close()

    init_engine(
        database_url=loaded_settings.database_url,
        echo=loaded_settings.debug,
    )
    session_factory = get_session_factory()
    app.state.federated_sync_service = FederatedSyncService(
        session_factory=session_factory,
        hospital_id=loaded_settings.dol_hospital_id,
        client=orchestrator_client,
    )
    federation_client: FederationClient | None = None
    if loaded_settings.federation_base_url and loaded_settings.federation_shared_secret:
        federation_client = FederationClient(
            base_url=loaded_settings.federation_base_url,
            shared_secret=loaded_settings.federation_shared_secret,
            hospital_id=loaded_settings.dol_hospital_id,
        )
        app.state.federation_client = federation_client

    trainer = FederatedTrainer(
        settings=loaded_settings,
        federation_client=federation_client,
    )
    app.state.federated_trainer = trainer

    if loaded_settings.federated_training_enabled and federation_client is not None:
        scheduler = FederatedTrainingLoop(
            trainer=trainer,
            triage_engine=app.state.triage_engine,
            federation_client=federation_client,
            interval_seconds=loaded_settings.federated_training_interval_seconds,
        )
        app.state.federated_scheduler = scheduler

        @app.on_event("startup")
        async def start_federated_training_loop() -> None:
            scheduler.start()

        @app.on_event("shutdown")
        async def stop_federated_training_loop() -> None:
            await scheduler.shutdown()

    return app


app = create_app()
