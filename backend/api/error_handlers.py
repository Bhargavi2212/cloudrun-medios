from __future__ import annotations

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from backend.services.error_response import StandardResponse
from backend.services.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app):
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        logger.warning(
            "HTTPException",
            extra={
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
            },
        )
        payload = StandardResponse(success=False, error=str(exc.detail), is_stub=False)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        logger.warning(
            "Validation error",
            extra={"errors": exc.errors(), "path": request.url.path},
        )
        payload = StandardResponse(
            success=False, error="Validation failed.", data=exc.errors()
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        payload = StandardResponse(
            success=False,
            error="Internal server error.",
            is_stub=True,
        )
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump()
        )


__all__ = ["register_exception_handlers"]
